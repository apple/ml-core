#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import TanhGaussianPolicy, FCNet, ObservationModel, DoubleCritic, ContrastivePrediction
from world_model import WorldModel
import numpy as np
from replay_buffer import SequenceReplayBuffer
import torch.optim
import argparse
import yaml
import os
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils import write_video_mp4, torchify, add_weight_decay, generate_expt_id, crop_image_tensor, get_parameter_list
from environments import make_environment
from copy import deepcopy
import sys


class Trainer(object):
    """ Trainer for all models. """
    def __init__(self, config, device, debug):
        self.config = config
        self.device = device
        self.debug = debug

        # Where should artifacts be written out.
        artifact_dir = config.get('artifact_dir', os.environ.get('BOLT_ARTIFACT_DIR', 'artifacts'))
        if not debug:
            self.log_dir = os.path.join(artifact_dir, config['expt_id'])
            os.makedirs(self.log_dir, exist_ok=True)
            config_filename = os.path.join(self.log_dir, 'config.yaml')
            with open(config_filename, 'w') as f:
                yaml.dump(config, f, sort_keys=True)
            print('Results will be logged to {}'.format(self.log_dir))
            self.tb_writer = SummaryWriter(self.log_dir)

        self.num_envs = self.config['num_envs']
        self.num_val_envs = self.config['num_val_envs']
        seed = self.config['seed']
        self.train_env_containers = [make_environment(self.config['env'], train=True, seed=seed+i) for i in range(self.num_envs)]
        seed += self.num_envs
        self.val_env_containers = [make_environment(self.config['env'], train=False, seed=seed+i) for i in range(self.num_val_envs)]
        env = self.train_env_containers[0]
        self.action_repeat = env.get_action_repeat()
        action_dims = env.get_action_dims()
        self.obs_channels, self.obs_height, self.obs_width = env.get_obs_chw()
        self.obs_other_dims = env.get_obs_other_dims()

        # Setup the observation encoder.
        self.crop_height = config['crop_height']
        self.crop_width = config['crop_width']
        self.same_crop_across_time = self.config.get('same_crop_across_time', False)
        self.random_crop_padding = self.config.get('random_crop_padding', 0)
        chw = (self.obs_channels, self.crop_height, self.crop_width)
        self.observation_model = ObservationModel(config, chw, self.obs_other_dims)
        obs_dims = self.observation_model.output_dims

        # Setup Contrastive Prediction.
        self.contrastive_prediction = ContrastivePrediction(config['contrastive'], obs_dims)

        # Setup the recurrent dynamics model.
        if 'wm' in config:
            self.model = WorldModel(config['wm'], obs_dims, action_dims)
            state_dims = self.model.state_dims
            self.exclude_wm_loss = config.get('exclude_wm_loss', False)
        else:  # For models like plain SAC.
            self.model = None
            state_dims = obs_dims
            self.exclude_wm_loss = True

        # Setup Actor and Critic.
        self.actor = TanhGaussianPolicy(config['actor'], state_dims, action_dims)
        self.critic = DoubleCritic(config['critic'], state_dims, action_dims)
        self.log_alpha = nn.parameter.Parameter(torch.tensor([float(self.config['initial_log_alpha'])], device=device))
        self.target_entropy = self.config.get('target_entropy', -action_dims)

        # Initialization.
        if 'initial_model_path' in config:
            model_path = os.path.join(artifact_dir, config['initial_model_path'])
            self.load(model_path)

        if self.model is not None:
            self.model = self.model.to(device)
        self.observation_model = self.observation_model.to(device)
        self.contrastive_prediction = self.contrastive_prediction.to(device)
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)

        self.has_momentum_encoder = config.get('momentum_encoder', False)
        
        # Set up optimizers.
        # Model optimizer.
        params = add_weight_decay(self.observation_model, config['weight_decay'])
        if self.model is not None:
            params.extend(add_weight_decay(self.model, config['weight_decay']))
        contrastive_pred_params = list(self.contrastive_prediction.parameters())
        params.append({'params': contrastive_pred_params, 'lr': config['lr_inverse_temp'], 'weight_decay': 0.0})
        self.optimizer = torch.optim.Adam(params, lr=config['lr'], betas=(0.9, 0.999))

        # Actor optimizer.
        actor_params = add_weight_decay(self.actor, config['weight_decay'])
        self.optimizer_actor = torch.optim.Adam(actor_params, lr=config['lr_actor'], betas=(config.get('momentum_actor', 0.9), 0.999))

        # Critic optimizer.
        critic_params = add_weight_decay(self.critic, config['weight_decay'])
        if config.get('include_model_params_in_critic', False):  # Include model params in critic.
            critic_params.extend(params)
            self.sac_detach_states = False
        else:
            self.sac_detach_states = True
        self.optimizer_critic = torch.optim.Adam(critic_params, lr=config['lr_critic'], betas=(config.get('momentum_critic', 0.9), 0.999))
        self.critic_optimizer_parameter_list = get_parameter_list(self.optimizer_critic)
        self.target_critic = deepcopy(self.critic).to(device)
        
        # Adaptive temperature optimizer for SAC.
        self.optimizer_alpha = torch.optim.Adam([self.log_alpha], lr=config['lr_alpha'],
                                                betas=(config.get('momentum_alpha', 0.9), 0.999))
    
        if self.has_momentum_encoder:
            self.target_encoder = deepcopy(self.observation_model.encoder)
            self.moco_dims = self.target_encoder.output_dims
            self.moco_W = nn.Parameter(torch.rand(self.moco_dims, self.moco_dims, device=self.device))
            curl_params = list(self.observation_model.encoder.parameters())
            curl_params.append(self.moco_W)
            self.curl_optimizer = torch.optim.Adam(curl_params, lr=config['lr_curl'], betas=(0.9, 0.999))
        
        self.optimizer_parameter_list = get_parameter_list(self.optimizer)

        # Whether SAC uses the mean or sampled state.
        self.sac_deterministic_state = config.get('sac_deterministic_state', False)

        # Whether to decode from the prior or posterior.
        self.recon_from_prior = self.config.get('recon_from_prior', False)

        # For saving the best model.
        self.best_loss = None

    def save(self, step, loss):
        """ Save a checkpoint. """
        if self.debug:
            return
        checkpoint_dir = os.path.join(self.log_dir, 'checkpoint_%08d' % step)
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        filename = os.path.join(checkpoint_dir, 'model.pt')
        print('Saved model to {}'.format(filename))
        info = {
            'observation_model_state_dict': self.observation_model.state_dict(),
            'contrastive_prediction_state_dict': self.contrastive_prediction.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'log_alpha': self.log_alpha.item(),
            'loss': loss,
            'step': step
        }
        if self.model is not None:
            info['model_state_dict'] = self.model.state_dict()
        torch.save(info, filename)
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            filename = os.path.join(self.log_dir, 'best_model.pt')
            print('Saved model to {}'.format(filename))
            torch.save(info, filename)

    def load(self, filename, skip_actor_critic=False):
        info = torch.load(filename, map_location=torch.device('cpu'))
        missing, unexpected = self.observation_model.load_state_dict(info['observation_model_state_dict'], strict=False)
        print('Missing keys', missing)
        print('Unexpected keys', unexpected)
        if 'contrastive_prediction_state_dict' in info:
            self.contrastive_prediction.load_state_dict(info['contrastive_prediction_state_dict'])
        if not skip_actor_critic:
            self.actor.load_state_dict(info['actor_state_dict'])
            self.critic.load_state_dict(info['critic_state_dict'])
            self.log_alpha[0] = info['log_alpha']
        if self.model is not None:
            self.model.load_state_dict(info['model_state_dict'])

    def normalize(self, obs):
        return obs.float() / 255

    def unnormalize(self, obs):
        obs = obs[..., -3:, :, :]  # Select the last three channels. Sometimes we have stacked frames.
        return torch.clamp(obs * 255, 0, 255).to(torch.uint8)

    def forward_prop(self, batch, decoding_for_viz=False):
        """
        Fprop the batch through the model.
        """
        outputs = self.observation_model(batch)  # (T, B, dims)
        batch['obs_features'] = outputs['obs_features']

        # Fprop through the world model.
        outputs.update(self.model(batch))

        if self.model.decoder is not None:  # If the world model uses decoding.
            if self.recon_from_prior:
                obs_features_recon = outputs['obs_features_recon_prior']
            else:
                obs_features_recon = outputs['obs_features_recon_post']
            outputs['obs_features_recon'] = obs_features_recon

            if self.observation_model.decoder is not None:  # If decoding all the way to pixels.
                if self.config.get('detach_pixel_decoder', False):
                    obs_features_recon = obs_features_recon.detach()
                obs_recon = self.observation_model.decode(obs_features_recon)
                outputs['obs_recon'] = obs_recon
                if decoding_for_viz:
                    # We want to visualize recon from both prior and posterior latent state.
                    # One of them is computed above, so the other is computed here.
                    if self.recon_from_prior:
                        outputs['obs_recon_prior'] = obs_recon
                        obs_features_recon = outputs['obs_features_recon_post'].detach()
                        outputs['obs_recon_post'] = self.observation_model.decode(obs_features_recon)
                    else:
                        outputs['obs_recon_post'] = obs_recon
                        obs_features_recon = outputs['obs_features_recon_prior'].detach()
                        outputs['obs_recon_prior'] = self.observation_model.decode(obs_features_recon)
        return outputs

    def loss_reward(self, batch, outputs, loss, metrics):
        loss_scales = self.config['loss_scales']
        if loss_scales['eta_r'] == 0:
            metrics['loss_reward'] = 0.
        else:
            reward_prediction = outputs['reward_prediction'][1:]  # The reward is the reward at time t.
            reward = batch['reward'][:-1]  # reward at index t corresponds to state t+1.
            loss_reward = nn.functional.smooth_l1_loss(reward_prediction, reward).mean()
            metrics['loss_reward'] = loss_reward.item()
            loss = loss + loss_reward * loss_scales['eta_r']
        return loss, metrics

    def loss_fwd_dynamics(self, batch, outputs, loss, metrics):
        loss_scales = self.config['loss_scales']
        if loss_scales['eta_fwd'] == 0:
            metrics['loss_fwd'] = 0.
        else:
            posterior_detached = {k: v.detach() for k, v in outputs['posterior'].items()}

            # skip t=0, because prior is uninformative there. No reason why posterior should match that.
            loss_fwd = self.model.dynamics.compute_forward_dynamics_loss(posterior_detached, outputs['prior'])[1:]
            loss_fwd = loss_fwd.mean()

            if 'eta_q' in loss_scales and loss_scales['eta_q'] > 0.0:
                eta_q = loss_scales['eta_q']
                prior_detached = {k: v.detach() for k, v in outputs['prior'].items()}
                loss_fwd_q = self.model.dynamics.compute_forward_dynamics_loss(outputs['posterior'], prior_detached)[1:]
                loss_fwd_q = loss_fwd_q.mean()
                loss_fwd = (1 - eta_q) * loss_fwd + loss_scales['eta_q'] * loss_fwd_q
            metrics['loss_fwd'] = loss_fwd.item()
            loss = loss + loss_fwd * loss_scales['eta_fwd']
        return loss, metrics

    def loss_contrastive(self, batch, outputs, train, loss, metrics):
        loss_scales = self.config['loss_scales']
        eta_s = loss_scales['eta_s']
        if eta_s > 0:
            x = outputs['obs_features']
            y = outputs['obs_features_recon']
            if self.recon_from_prior:  # Skip t=0 when reconstructing from prior, because prior knows nothing at t=0.
                x = x[1:]
                y = y[1:]
            loss_contrastive = self.contrastive_prediction(x, y, train=train)
            loss = loss + loss_contrastive * eta_s
            metrics['loss_contrastive'] = loss_contrastive.item()
        return loss, metrics

    def loss_observation_recon(self, batch, outputs, loss, metrics):
        if 'obs_recon' not in outputs:
            return loss, metrics
        obs_recon = outputs['obs_recon']
        obs = batch.get('obs_image_clean', batch['obs_image'])
        if self.recon_from_prior:  # Skip t=0.
            obs_recon = obs_recon[1:]
            obs = obs[1:]
        loss_obs = nn.functional.smooth_l1_loss(obs_recon, obs).mean()
        loss_scales = self.config['loss_scales']
        metrics['loss_obs'] = loss_obs.item()
        loss = loss + loss_scales['eta_x'] * loss_obs
        return loss, metrics

    def loss_inv_dynamics(self, batch, outputs, loss, metrics):
        if 'action_prediction' not in outputs:
            return loss, metrics
        loss_scales = self.config['loss_scales']
        if loss_scales['eta_inv'] == 0:
            metrics['loss_inverse_dynamics'] = 0.
        else:
            action_prediction = outputs['action_prediction']  # (T-1, B, a_dims) tanh valued.
            action = batch['action'][:-1]
            loss_inverse_dynamics = 0.5 * ((action - action_prediction) ** 2).sum(dim=-1).mean()
            loss = loss + loss_scales['eta_inv'] * loss_inverse_dynamics
            metrics['loss_inverse_dynamics'] = loss_inverse_dynamics.item()
        return loss, metrics

    def compute_loss(self, batch, train, decoding_for_viz=False):
        outputs = self.forward_prop(batch, decoding_for_viz=decoding_for_viz)
        metrics = {}
        loss = 0
        loss, metrics = self.loss_reward(batch, outputs, loss, metrics)
        loss, metrics = self.loss_inv_dynamics(batch, outputs, loss, metrics)
        loss, metrics = self.loss_fwd_dynamics(batch, outputs, loss, metrics)
        loss, metrics = self.loss_contrastive(batch, outputs, train, loss, metrics)
        loss, metrics = self.loss_observation_recon(batch, outputs, loss, metrics)
        metrics['loss_total'] = loss.item()
        return loss, metrics, outputs

    def update_curl(self, batch, step, heavy_logging=False):
        with torch.no_grad():
            f_k, _ = self.observation_model.encode_pixels(batch['obs_image_2'], encoder=self.target_encoder)
            f_k = f_k.detach().view(-1, self.moco_dims)
        f_q, _ = self.observation_model.encode_pixels(batch['obs_image'])
        f_q = f_q.view(-1, self.moco_dims)
        f_proj = torch.matmul(f_k, self.moco_W)
        logits = torch.matmul(f_q, f_proj.T)
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(log_probs.diagonal().mean())
        metrics = {'moco_loss' : loss.item()}
        
        self.curl_optimizer.zero_grad()
        loss.backward()
        self.curl_optimizer.step()

        # Do momentum update.
        tau = self.config.get('update_target_encoder_tau', 1)
        self.update_target(self.target_encoder, self.observation_model.encoder, tau)

        return metrics

    def update_world_model(self, batch, step, heavy_logging=False):
        """
        Update the world model.
        batch : Dict containing keys ('action', 'obs_image', 'reward', etc)
            'action' : (T, B action_dims)
            'obs_image' : (T, B, C, H, W)
            'reward': (T, B)
        """
        loss, metrics, outputs = self.compute_loss(batch, train=True, decoding_for_viz=heavy_logging)
        self.optimizer.zero_grad()
        loss.backward()
        if 'max_grad_norm_wm' in self.config:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.optimizer_parameter_list, self.config['max_grad_norm_wm'])
            metrics['grad_norm_wm'] = grad_norm.item()
        self.optimizer.step()

        if step % self.config['print_every'] == 0:
            loss_str = ' '.join(['{}: {:.2f}'.format(k, v) for k, v in sorted(metrics.items())])
            print('Step {} {}'.format(step, loss_str))
        if not self.debug and self.tb_writer is not None:
            for k, v in metrics.items():
                self.tb_writer.add_scalar('metrics/{}'.format(k), v, step)
            if heavy_logging:
                max_B = 16
                self.tb_writer.add_video('obs/input',
                                         self.unnormalize(batch['obs_image']).transpose(0, 1)[:max_B], step)
                if self.observation_model.decoder is not None:
                    self.tb_writer.add_video('obs/recon',
                                             self.unnormalize(outputs['obs_recon']).transpose(0, 1)[:max_B], step)
                    self.tb_writer.add_video('obs/recon_post',
                                             self.unnormalize(outputs['obs_recon_post']).transpose(0, 1)[:max_B], step)
                    self.tb_writer.add_video('obs/recon_prior',
                                             self.unnormalize(outputs['obs_recon_prior']).transpose(0, 1)[:max_B], step)
        return metrics, outputs

    def log_video(self, video_tag, frames, step):
        """
        Log a video to disk.
        Args:
            frames : List of (B, T, C, H, W)
            step: training step.
            video_tag: tag used for logging into tensorboard and as dir name for disk.
        """
        self.tb_writer.add_video(video_tag, frames, step)

        B, T, C, H, W = list(frames.shape)
        frames = frames.permute(1, 2, 3, 0, 4).contiguous().view(T, C, H, B*W)  # Stack batch along width.
        video_dir = os.path.join(self.log_dir, video_tag)
        os.makedirs(video_dir, exist_ok=True)
        filename = os.path.join(video_dir, 'video_%08d.mp4' % step)
        write_video_mp4(filename, frames)

    def validate(self, step):
        self.observation_model.eval()
        if self.model is not None:
            self.model.eval()
        self.actor.eval()
        self.critic.eval()
        tic = time.time()
        metrics = {}
        # Collect data. One episode in each val environment.
        replay_buffer = SequenceReplayBuffer()
        num_episodes_per_val_env_for_reward = self.config.get('num_episodes_per_val_env_for_reward', 10)
        sample_policy = self.config.get('val_stochastic_policy', False)
        if sample_policy:
            print('Using stochastic policy for val')
        episode_reward = self.collect_data_from_actor(replay_buffer,
                                                      num_episodes_per_env=num_episodes_per_val_env_for_reward,
                                                      train=False, sample_policy=sample_policy)
        metrics['episode_reward'] = episode_reward

        # Take the first few episodes for computing the rest of the metrics. They are expensive to compute.
        num_episodes_for_model = self.config.get('num_episodes_val_for_model', 5)
        batch = replay_buffer.sample(num_episodes_for_model)
        batch = self.prep_batch(batch, random_crop=False)
        steps_per_episode = self.config['episode_steps'] // self.action_repeat

        if not self.exclude_wm_loss:
            with torch.no_grad():
                loss, model_metrics, outputs = self.compute_loss(batch, train=False, decoding_for_viz=True)
                metrics.update(model_metrics)
                # Generate rollout from prior.
                if self.observation_model.decoder is not None:
                    init_t = self.config['rollout_prior_init_t']
                    assert 0 < init_t < steps_per_episode - 1
                    init_state = dict([(k, v[init_t-1]) for k, v in outputs['posterior'].items()])
                    prior = self.model.dynamics.rollout_prior(init_state, batch['action'][init_t:, ...], deterministic=False)
                    # Decode to images.
                    latent = self.model.dynamics.get_state(prior, deterministic=False)
                    obs_recon_imagined = self.observation_model.decode(self.model.decoder(latent))
                    # Add the first init_t images from the posterior.
                    obs_recon_imagined = torch.cat([outputs['obs_recon_prior'][:init_t, :], obs_recon_imagined], dim=0)
 
        elif self.observation_model.use_gating_network:  # Even if model is None, we want outputs to have gating.
            with torch.no_grad():
                outputs = self.observation_model(batch)  # (T, B, dims)  # Used to visualize gating.

        toc = time.time()
        metrics.update({
            'timing': toc - tic,
        })

        loss_str = ' '.join(['{}: {:.2f}'.format(k, v) for k, v in sorted(metrics.items())])
        print('Val Iter {} {}'.format(step, loss_str))
        if not self.debug and self.tb_writer is not None:
            for k, v in metrics.items():
                self.tb_writer.add_scalar('val_metrics/{}'.format(k), v, step)
            obs = self.unnormalize(batch['obs_image']).transpose(0, 1)  # (B, T, C, H, W)
            if self.observation_model.use_gating_network:
                obs_gating = outputs['obs_gating'].transpose(0, 1)  # (B, T, F, 1, H, W)
                obs_gating = obs_gating[:, :, -1, :, :, :]  # The gating for the last frame.
                obs_gating = (obs_gating * 255).to(torch.uint8)
                obs_gating = obs_gating.expand_as(obs).contiguous()  # replicate along RGB.
                obs = torch.cat([obs, obs_gating], dim=3)
            if self.model is not None and self.observation_model.decoder is not None:
                obs_recon = self.unnormalize(outputs['obs_recon']).transpose(0, 1)
                obs_recon_post = self.unnormalize(outputs['obs_recon_post']).transpose(0, 1)
                obs_recon_prior = self.unnormalize(outputs['obs_recon_prior']).transpose(0, 1)
                obs_recon_imagined = self.unnormalize(obs_recon_imagined).transpose(0, 1)
                obs = torch.cat([obs, obs_recon, obs_recon_post, obs_recon_prior, obs_recon_imagined], dim=3)
            self.log_video('obs/val', obs, step)
        return -episode_reward

    def collect_data_random_policy(self, replay_buffer, num_episodes_per_env=1, train=True):
        steps_per_episode = self.config['episode_steps'] // self.action_repeat
        env_containers = self.train_env_containers if train else self.val_env_containers
        total_reward = 0
        for env_container in env_containers:
            action_low, action_high = env_container.get_action_limits()
            action_dims = env_container.get_action_dims()
            for _ in range(num_episodes_per_env):
                obs = env_container.reset()
                seq = []
                for _ in range(steps_per_episode):
                    action = np.random.uniform(action_low, action_high, action_dims)
                    next_obs, reward, _, _ = env_container.step(action)
                    seq.append(dict(obs=obs, action=action, reward=reward))
                    obs = next_obs
                    total_reward += reward
                replay_buffer.add(seq)
        avg_reward = total_reward / (num_episodes_per_env * len(env_containers))
        return avg_reward

    def prep_batch(self, batch, random_crop=False):
        """ Prepare batch of data for input to the model.
        Inputs:
            batch : Dict containing 'obs', etc.
        Returns:
            batch: Same dict, but with images randomly cropped, moved to GPU, normalized.
        """
        for key in batch.keys():
            batch[key] = batch[key].to(self.device)
        obs_image_cropped = crop_image_tensor(batch['obs_image'], self.crop_height, self.crop_width,
                                               random_crop=random_crop,
                                               same_crop_across_time=self.same_crop_across_time,
                                               padding=self.random_crop_padding)
        if self.has_momentum_encoder:
            batch['obs_image_2'] = crop_image_tensor(batch['obs_image'], self.crop_height, self.crop_width,
                                                   random_crop=random_crop,
                                                   same_crop_across_time=self.same_crop_across_time,
                                                   padding=self.random_crop_padding)

        if 'obs_image_clean' in batch:  # When we have paired distraction-free and distracting obs.
            batch['obs_image_clean'] = crop_image_tensor(batch['obs_image_clean'], self.crop_height, self.crop_width, random_crop=False, same_crop_across_time=True, padding=0)
        else:
            batch['obs_image_clean'] = crop_image_tensor(batch['obs_image'], self.crop_height, self.crop_width, random_crop=False, same_crop_across_time=True, padding=0)

        batch['obs_image'] = obs_image_cropped
        if len(batch['obs_image'].shape) == 5:  # (B, T, C, H, W) -> (T, B, C, H, W)
            swap_first_two_dims = True
        else:  # (B, C, H, W) -> no change.
            swap_first_two_dims = False
        for key in batch.keys():
            if swap_first_two_dims:
                batch[key] = batch[key].transpose(0, 1)
            batch[key] = batch[key].contiguous().float().detach()
        batch['obs_image'] = self.normalize(batch['obs_image'])
        if 'obs_image_clean' in batch:
            batch['obs_image_clean'] = self.normalize(batch['obs_image_clean'])
        if 'obs_imaage_2' in batch:
            batch['obs_image_2'] = self.normalize(batch['obs_image_2'])
        return batch

    def collect_data_from_actor(self, replay_buffer, num_episodes_per_env=1, train=True, sample_policy=True):
        steps_per_episode = self.config['episode_steps'] // self.action_repeat
        self.observation_model.eval()
        if self.model is not None:
            self.model.eval()
        self.actor.eval()
        reward_total = 0
        env_containers = self.train_env_containers if train else self.val_env_containers
        num_env = len(env_containers)
        
        for _ in range(num_episodes_per_env):
            seq_list = []
            obs_list = []
            for env_container in env_containers:
                obs = env_container.reset()
                seq_list.append(list())
                obs_list.append(dict(obs=obs))
            posterior = None
            action = None
            for _ in range(steps_per_episode):
                # Find the action to take for a batch of environments.
                batch = torchify(obs_list)  # Dict of (B, ...)
                batch = self.prep_batch(batch, random_crop=False)
                outputs = self.observation_model(batch)
                obs_features = outputs['obs_features']
                if self.model is not None:  # If using a dynamics model.
                    latent, posterior = self.model.forward_one_step(obs_features, posterior, action,
                                                                    deterministic_latent=self.sac_deterministic_state)
                else:
                    latent = obs_features
                action, _, _ = self.actor(latent, sample=sample_policy)
                action_npy = action.detach().cpu().numpy()  # (B, a_dims)

                # Step each environment with the computed action.
                for i, env_container in enumerate(env_containers):
                    current_action = action_npy[i]
                    obs, reward, _, _ = env_container.step(current_action)
                    seq_list[i].append(dict(obs=obs_list[i]['obs'], action=current_action, reward=reward))
                    obs_list[i]['obs'] = obs
                    reward_total += reward
            for seq in seq_list:
                replay_buffer.add(seq)
        episode_reward = reward_total / (num_env * num_episodes_per_env)
        return episode_reward

    def update_target(self, target, critic, tau):
        target_params_dict = dict(target.named_parameters())
        for n, p in critic.named_parameters():
            target_params_dict[n].data.copy_(
                (1 - tau) * target_params_dict[n] + tau * p
            )

    def update_actor_critic_sac(self, batch, step, heavy_logging=False):
        """
        Inputs:
            batch : Dict containing keys ('action', 'obs', 'reward')
                'action' : (T, B, action_dims)
                'obs' : (T, B, C, H, W)
                'reward': (T, B)
        """
        metrics = {}

        outputs = self.observation_model(batch)  # (T, B, dims)
        obs_features = outputs['obs_features']
        batch['obs_features'] = obs_features
        if self.model is not None:
            outputs = self.model(batch)  # Dict containing prior (stoch, logits), posterior(..)
            states = self.model.dynamics.get_state(outputs['posterior'],
                                                   deterministic=self.sac_deterministic_state)
        else:
            states = obs_features

        # Update critic (potentially including the encoder).
        current_states = states[:-1]
        if self.sac_detach_states:
            current_states = current_states.detach()
        current_actions = batch['action'][:-1]
        reward = batch['reward'][:-1]  # (T-1, B)
        next_states = states[1:].detach()
        alpha = torch.exp(self.log_alpha).detach()
        gamma = self.config['gamma']
        with torch.no_grad():
            if torch.isnan(next_states).any():
                raise Exception('Next states contains nan')
            next_actions, next_action_log_probs, _ = self.actor(next_states)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_v = torch.min(target_q1, target_q2) - alpha * next_action_log_probs
            target_q = reward + gamma * target_v
        q1, q2 = self.critic(current_states, current_actions)  # (T-1, B)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        if 'max_grad_norm_critic' in self.config:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config['max_grad_norm_critic'])
            metrics['grad_norm_critic'] = grad_norm.item()
        self.optimizer_critic.step()

        # Update actor.
        current_states_detached = current_states.detach()  # Actor loss does not backpropagate into encoder or dynamics.
        policy_actions, policy_action_log_probs, policy_action_std = self.actor(current_states_detached)  # (T-1, B, action_dims)
        q1, q2 = self.critic(current_states_detached, policy_actions)
        q = torch.min(q1, q2)
        q_loss = -q.mean()
        entropy_loss = policy_action_log_probs.mean()
        entropy_loss_wt = torch.exp(self.log_alpha).detach()
        actor_loss = q_loss + entropy_loss_wt * entropy_loss
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        if 'max_grad_norm_actor' in self.config:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config['max_grad_norm_actor'])
            metrics['grad_norm_actor'] = grad_norm.item()
        self.optimizer_actor.step()

        # Update alpha (adaptive entropy loss wt)
        alpha_loss = -(torch.exp(self.log_alpha) * (self.target_entropy + entropy_loss.detach()))
        self.optimizer_alpha.zero_grad()
        alpha_loss.backward()
        if 'max_grad_norm_log_alpha' in self.config:
            grad_norm = torch.nn.utils.clip_grad_norm_([self.log_alpha], self.config['max_grad_norm_log_alpha'])
            metrics['grad_norm_log_alpha'] = grad_norm.item()
        self.optimizer_alpha.step()
        if 'max_log_alpha' in self.config:
            with torch.no_grad():
                self.log_alpha.clamp_(max=self.config['max_log_alpha'])

        if step % self.config['update_target_critic_after'] == 0:
            tau = self.config.get('update_target_critic_tau', 1)
            self.update_target(self.target_critic, self.critic, tau)

        metrics.update({
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_loss': q_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'log_alpha': self.log_alpha.item(),
        })
        if not self.debug and self.tb_writer is not None:
            for k, v in metrics.items():
                self.tb_writer.add_scalar('rl_metrics/{}'.format(k), v, step)
            if heavy_logging:
                self.tb_writer.add_histogram('rl_metrics/reward', reward.view(-1), step)
                self.tb_writer.add_histogram('rl_metrics/q_targets', target_q.view(-1), step)
                self.tb_writer.add_histogram('rl_metrics/critic_scores', q.view(-1), step)
                self.tb_writer.add_histogram('rl_metrics/action', policy_actions.view(-1), step)
                self.tb_writer.add_histogram('rl_metrics/action_log_probs', policy_action_log_probs.view(-1), step)
                self.tb_writer.add_histogram('rl_metrics/action_std', policy_action_std.view(-1), step)
        return metrics

    def train(self):
        """ Train the model."""
        # Setup replay buffer.
        steps_per_episode = self.config['episode_steps'] // self.action_repeat
        replay_buffer_size = self.config['replay_buffer_size']
        num_episodes_in_replay_buffer = replay_buffer_size // steps_per_episode
        replay_buffer = SequenceReplayBuffer(size=num_episodes_in_replay_buffer)

        # Find out how many data collection iterations to do use.
        max_steps = self.config['max_steps'] // self.action_repeat
        num_iters = max_steps // (self.num_envs * steps_per_episode)

        # How many gradients updates per iteration.
        num_updates_per_iter = int(steps_per_episode * self.config.get('update_frequency_factor', 1.0))

        random_crop = self.config.get('random_crop', False)
        B = self.config['batch_size']
        T = self.config['dynamics_seq_len']
        train_step = 0

        # Initial data collection.
        initial_episodes_per_env = self.config['initial_data_steps'] // (self.num_envs * steps_per_episode)  # Used to delay both world model and rl training.
        start_rl_training_after = self.config['start_rl_training_after']  # Used to delay rl training until world model has updated for a bit.

        for ii in range(num_iters):
            if ii % self.config['validate_every_iters'] == 0:
                loss = self.validate(ii)
                if ii == 0:
                    print('Completed validation')
                self.save(ii, loss)

            # Collect data. One episode in each environment.
            with torch.no_grad():
                if ii < initial_episodes_per_env or train_step < start_rl_training_after:
                    episode_reward = self.collect_data_random_policy(replay_buffer, num_episodes_per_env=1, train=True)
                else:
                    episode_reward = self.collect_data_from_actor(replay_buffer, num_episodes_per_env=1, train=True,
                                                                  sample_policy=True)
                if not self.debug and self.tb_writer is not None:
                    self.tb_writer.add_scalar('rl_metrics/episode_reward', episode_reward, ii)

            if ii < initial_episodes_per_env:  # No updates until a few episodes have been collected.
                continue
            self.observation_model.train()
            if self.model is not None:
                self.model.train()
            self.actor.train()
            self.critic.train()
            for i in range(num_updates_per_iter):
                # Train world model.
                tic = time.time()
                train_step += 1
                batch = replay_buffer.sample(B, T)  # Dict of (B, T, ..)
                batch = self.prep_batch(batch, random_crop=random_crop)
                tic1 = time.time()
                if not self.exclude_wm_loss:  # Skip for model-free variants, like SAC, RSAC.
                    self.update_world_model(batch, train_step, heavy_logging=(i == 0))
                tic2 = time.time()
                if self.has_momentum_encoder:
                    self.update_curl(batch, train_step, heavy_logging=(i == 0))
                tic3 = time.time()
                if train_step >= start_rl_training_after:
                    self.update_actor_critic_sac(batch, train_step, heavy_logging=(i == 0))
                toc = time.time()
                timing_metrics = {
                    'time_data_prep': tic1 - tic,
                    'time_wm_update': tic2 - tic1,
                    'time_curl_update': tic3 - tic2,
                    'time_ac_update': toc - tic3,
                    'time_per_update': toc - tic,
                }
                if not self.debug and self.tb_writer is not None:
                    for k, v in timing_metrics.items():
                        self.tb_writer.add_scalar('timing_metrics/{}'.format(k), v, train_step)
                if train_step == 1:
                    print('Completed one step')


def argument_parser(argument):
    """ Argument parser """
    parser = argparse.ArgumentParser(description='Binder Network.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('-c', '--config', default='', type=str, help='Training config')
    parser.add_argument('--debug', action='store_true', help='Debug mode. Disable logging.')
    args = parser.parse_args(argument)
    return args


def main():
    args = argument_parser(None)
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Running on GPU {}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device('cpu')
        print('Running on CPU')

    try:
       with open(args.config) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error opening specified config yaml at: {}. "
              "Please check filepath and try again.".format(args.config))
        sys.exit(1)

    config = config['parameters']
    config['expt_id'] = generate_expt_id()
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trainer = Trainer(config, device, args.debug)
    trainer.train()


if __name__ == '__main__':
    main()

