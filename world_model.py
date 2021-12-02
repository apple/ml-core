#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import stack_tensor_dict_list
from modules import FCNet, CNN, TransposeCNN, GRUCell, weight_init


def get_pairwise_smooth_l1(x):
    dims = x.shape[0]
    x_shape = list(x.shape)
    x1 = x.unsqueeze(0).expand(dims, *x_shape)
    x2 = x.unsqueeze(1).expand(dims, *x_shape)
    return F.smooth_l1_loss(x1, x2, reduction='none')


def get_pairwise_l2(x):
    dims = x.shape[0]
    x_shape = list(x.shape)
    x1 = x.unsqueeze(0).expand(dims, *x_shape)
    x2 = x.unsqueeze(1).expand(dims, *x_shape)
    return F.mse_loss(x1, x2, reduction='none')


def sample_softmax(probs):
    dist = torch.distributions.one_hot_categorical.OneHotCategorical(probs=probs)
    stoch = dist.sample() + probs - probs.detach()  # Trick to send straight-through gradients into probs.
    return stoch


class Dynamics(nn.Module):
    def __init__(self, config, obs_embed_dims, action_dims):
        super().__init__()
        self.config = config
        self.action_dims = action_dims
        self.obs_embed_dims = obs_embed_dims

        self.forward_dynamics_loss = config['forward_dynamics_loss']
        assert self.forward_dynamics_loss in ['neg_log_prob', 'kl']
        self.discrete = config.get('discrete', False)
        if self.discrete:
            self.num_softmaxes = config['num_softmaxes']
            self.dims_per_softmax = config['dims_per_softmax']
            self.latent_dims = self.dims_per_softmax * self.num_softmaxes
            self.suff_stats_dims = self.latent_dims  # dims needed to specify a distribution over the latent state.
        else:
            self.latent_dims = config['latent_dims']
            self.suff_stats_dims = 2 * self.latent_dims  # mu, log_std

        self.recurrent = config.get('recurrent', False)
        if self.recurrent:
            self.rnn_state_dims = config['rnn_state_dims']
        else:
            self.rnn_state_dims = 0
        self.state_dims = self.latent_dims + self.rnn_state_dims

        if self.recurrent:
            # Input to recurrent model.
            self.rnn_input_net = FCNet(config['rnn_input'], self.latent_dims + action_dims)  # Inputs: Prev latent + act.
            num_features = self.rnn_input_net.output_dims
            print('RNN Input Net input_dims: {} output_dims {}'.format(self.latent_dims + action_dims, num_features))

            # Recurrent model.
            print('GRUCell input_dims {} hidden dims {}'.format(num_features, self.rnn_state_dims))
            self.cell = GRUCell(num_features, self.rnn_state_dims, norm=True)  # h_t = cell([z_{t-1},a_{t-1}], h_{t-1})

            # Prior model (Transition function) P(z_t | h_t)
            print('Prior Net input dims {} output dims {}'.format(self.rnn_state_dims, self.suff_stats_dims))
            self.prior_net = FCNet(config['prior'], self.rnn_state_dims, out_features=self.suff_stats_dims)
        else:
            # Prior model (Transition function) P(z_t | h_{t-1}, a_{t-1})
            print('Prior Net --------')
            print('Prior Net input dims {} output dims {}'.format(self.latent_dims + action_dims, self.suff_stats_dims))
            self.prior_net = FCNet(config['prior'], self.latent_dims + action_dims, self.suff_stats_dims)

        # Posterior model (Representation model). P(z_t | x_t, h_t)
        print('Posterior Net input dims {} output dims {}'.format(self.obs_embed_dims + self.rnn_state_dims,
              self.suff_stats_dims))
        self.posterior_net = FCNet(config['posterior'], self.obs_embed_dims + self.rnn_state_dims,
                                   out_features=self.suff_stats_dims)

        # Initial_state.
        if self.discrete:
            initial_logits = torch.zeros(self.num_softmaxes, self.dims_per_softmax)
            self.register_buffer('initial_logits', initial_logits)
        else:
            initial_mu = torch.zeros(self.latent_dims)
            initial_log_std = torch.zeros(self.latent_dims)
            self.register_buffer('initial_mu', initial_mu)
            self.register_buffer('initial_log_std', initial_log_std)
        if self.recurrent:
            initial_rnn_state = torch.zeros(self.rnn_state_dims)
            self.register_buffer('initial_rnn_state', initial_rnn_state)

        self.elementwise_bisim = False
        self.free_kl = config.get('free_kl', 0.0)

    def initial_state(self, batch_size, deterministic=False):
        if self.discrete:
            logits = self.initial_logits.unsqueeze(0).expand(batch_size, -1, -1)
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                state = probs
            else:
                state = sample_softmax(probs)
            state = dict(logits=logits, probs=probs.detach(), state=state.detach())
        else:
            mu = self.initial_mu.unsqueeze(0).expand(batch_size, -1)
            log_std = self.initial_log_std.unsqueeze(0).expand(batch_size, -1)
            if deterministic:
                state = mu
            else:
                state = mu + torch.exp(log_std) * torch.randn_like(log_std)
            state = dict(log_std=log_std, mu=mu, state=state.detach())
        if self.recurrent:
            state['rnn_state'] = self.initial_rnn_state.unsqueeze(0).expand(batch_size, -1)
        return state

    def get_state(self, state_dict, deterministic=False):
        """
        Get the latent vector by concatenating rnn state and flattened latent state.
        Args:
            state_dict: Dict containing 'state', 'rnn_state', 'mu', 'log_std', (OR 'logits')
        Returns:
            state: (B, D)
        """
        if deterministic:
            if self.discrete:
                state = state_dict['probs'].flatten(-2, -1)
            else:
                state = state_dict['mu']
        else:
            if self.discrete:
                state = state_dict['state'].flatten(-2, -1)
            else:
                state = state_dict['state']
        if self.recurrent:
            state = torch.cat([state, state_dict['rnn_state']], dim=-1)
        return state

    def compute_kl(self, state_p, state_q):
        """
        p log (p/q)
        """
        if self.discrete:
            logits_p = state_p['logits']
            p = state_p['probs']
            logits_q = state_q['logits']
            log_p = nn.functional.log_softmax(logits_p, dim=-1)
            log_q = nn.functional.log_softmax(logits_q, dim=-1)
            kld = (p * (log_p - log_q)).sum(dim=-1)
        else:
            mu_1 = state_p['mu']
            mu_2 = state_q['mu']
            log_std_1 = state_p['log_std']
            log_std_2 = state_q['log_std']
            var_1 = torch.exp(2 * log_std_1)
            var_2 = torch.exp(2 * log_std_2)
            kld = log_std_2 - log_std_1 + (var_1 + (mu_1 - mu_2) ** 2) / (2 * var_2) - 0.5
            kld = kld.sum(dim=-1)
        if self.free_kl > 0.0:
            kld = kld.clamp_min(self.free_kl)
        return kld

    def compute_neg_log_prob(self, state_p, state_q):
        """
        Negative Log prob of mean of p under the q distribution.
        """
        if self.discrete:
            raise NotImplementedError
        else:
            x = state_p['mu']
            mu = state_q['mu']
            log_std = state_q['log_std'].clamp(-10., 10.)
            var = torch.exp(2 * log_std)
            neg_log_prob = log_std + ((x - mu) ** 2) / (2 * var)
            neg_log_prob = neg_log_prob.mean(dim=-1)
        return neg_log_prob

    def compute_forward_dynamics_loss(self, state_p, state_q):
        if self.forward_dynamics_loss == 'kl':
            return self.compute_kl(state_p, state_q)
        elif self.forward_dynamics_loss == 'neg_log_prob':
            return self.compute_neg_log_prob(state_p, state_q)
        else:
            raise Exception('Unknown forward dynamics loss')

    def process_suff_stats(self, suff_stats):
        """
        Process the sufficient statistics (obtained from the output of posterior or prior networks).
        If discrete, the output is interpreted as logits.
        Otherwise, the output is chunked into two : mu and sigma.
        """
        # Get the prior state.
        res = {}
        if self.discrete:
            logits = suff_stats.view(-1, self.num_softmaxes, self.dims_per_softmax)
            probs = nn.functional.softmax(logits, dim=-1)
            state = self.sample_softmax(probs)
            res['logits'] = logits
            res['probs'] = probs
        else:
            mu, log_std = suff_stats.chunk(2, -1)
            log_std = log_std.clamp(-10.0, 10.0)
            state = mu + torch.exp(log_std) * torch.randn_like(log_std)
            res['mu'] = mu
            res['log_std'] = log_std
        res['state'] = state
        return res

    def imagine_step(self, prev_state_dict, prev_action, deterministic=False):
        """
        Args:
            prev_state_dict : Dict containing 'state', 'mu', etc (B, num_softmaxes, dims_per_softmax).
            prev_action: (B, action_dims) or None
            deterministic: If True, sample from the prev_state distribution, otherwise use the mode.
        Returns:
            prior: Dict containing 'stoch', 'deter', 'logits'
        """

        # prev_action is None at the first time step where we don't know what action led to the first state.
        # In this case, we just pass on the prior as out best guess.
        if prev_action is None:
            return prev_state_dict

        if deterministic:
            if self.discrete:
                prev_state = prev_state_dict['probs']
            else:
                prev_state = prev_state_dict['mu']
        else:
            prev_state = prev_state_dict['state']
        if self.discrete:
            prev_state = prev_state.flatten(-2, -1)
        x = torch.cat([prev_state, prev_action], dim=-1)

        res = {}
        if self.recurrent:
            # Step the RNN to get rnn state.
            prev_rnn_state = prev_state_dict['rnn_state']
            x = self.rnn_input_net(x)
            x = self.cell(x, prev_rnn_state)
            res['rnn_state'] = x

        # Get the prior state.
        x = self.prior_net(x)
        res.update(self.process_suff_stats(x))
        return res

    def forward(self, prev_state_dict, prev_action, obs_embed, deterministic=False):
        """
        Take one step of the dynamics, and return the updated prior and posterior.
        Args:
            prev_state_dict : Dict containing 'state' (B, num_softmaxes, dims_per_softmax), 'rnn_state' (B, deter_dims)
            prev_action: (B, action_dims)
            obs_embed: (B, obs_embed_dims)
            deterministic: If false, estimate the prior using samples from the prev_state distribution, otherwise use the mode.
        Returns:
            prior: Dict containing 'state', 'rnn_state', ('logits', OR 'mu', 'log_std')
            posterior: Dict containing 'state', 'rnn_state', ('logits', OR 'mu', 'log_std')
        """
        prior = self.imagine_step(prev_state_dict, prev_action, deterministic=deterministic)
        posterior = {}
        x = obs_embed
        if self.recurrent:
            rnn_state = prior['rnn_state']
            posterior['rnn_state'] = rnn_state
            x = torch.cat([x, rnn_state], dim=-1)
        x = self.posterior_net(x)
        posterior.update(self.process_suff_stats(x))
        return prior, posterior

    def rollout_prior(self, initial_state, actions, deterministic=False):
        """
        Rollout the dynamics given actions, starting from an initial state.
        Args:
            initial_state: Dict containing 'state', 'rnn_state', ('logits', OR 'mu', 'log_std') (B, ...)
            actions: (T, B, action_dims)
            deterministic: If true, use samples from the state distribution over time, otherwise use the mode.
        Return:
            state: Dict containing 'state', 'rnn_state', ('logits', OR 'mu', 'log_std') (T, B, ...)
        """
        T = actions.shape[0]
        state = initial_state
        state_list = []
        for t in range(T):
            state = self.imagine_step(state, actions[t], deterministic=deterministic)
            state_list.append(state)
        state = stack_tensor_dict_list(state_list)
        return state


class WorldModel(nn.Module):
    def __init__(self, config, obs_dims, action_dims):
        super().__init__()

        # Obs encoder.
        if 'obs_encoder' in config:
            self.encoder = FCNet(config['obs_encoder'], obs_dims)
            obs_embed_dims = self.encoder.output_dims
            print('Observation feature encoder ---- Input dims {} Output dims {}'.format(obs_dims, obs_embed_dims))
        else:
            self.encoder = None
            obs_embed_dims = obs_dims

        # Dynamics Model.
        self.dynamics = Dynamics(config['dynamics'], obs_embed_dims, action_dims)
        state_dims = self.dynamics.state_dims
        print('Dynamics ---- obs dims {} action dims {} state dims {}'.format(obs_embed_dims, action_dims, state_dims))

        # Image predictor P(x_t | z_t, h_t)
        if 'obs_decoder' in config:
            self.decoder = FCNet(config['obs_decoder'], state_dims, out_features=obs_dims)
            print('Observation feature decoder ---- input dims {} output dims {}'.format(state_dims, obs_dims))
        else:
            self.decoder = None

        print('Reward predictor ---- input dims {}'.format(state_dims))
        # Reward predictor P(r | z_t, h_t)
        self.reward_net = FCNet(config['reward_net'], state_dims)
        self.reward_prediction_from_prior = config['reward_prediction_from_prior']

        if 'inverse_dynamics' in config:
            print('Inverse dynamics model ------ input dims 2 * {} action dims {}'.format(state_dims, action_dims))
            self.inv_dynamics = FCNet(config['inverse_dynamics'], 2 * state_dims, action_dims)
        else:
            self.inv_dynamics = None

        self.propagate_deterministic = config['propagate_deterministic']
        self.decode_deterministic = config['decode_deterministic']
        self.state_dims = state_dims

        if self.dynamics.forward_dynamics_loss == 'neg_log_prob':
            assert self.propagate_deterministic, "Posterior variance is not trained, propagate_deterministic should be True"
        self.apply(weight_init)

    def forward_one_step(self, obs_encoding, posterior=None, action=None, deterministic_latent=True):
        """
        Inputs:
            batch: Dict containing
        """
        if posterior is None:
            B = obs_encoding.size(0)
            posterior = self.dynamics.initial_state(B)
        obs_embed = self.encoder(obs_encoding)
        _, posterior = self.dynamics(posterior, action, obs_embed, deterministic=self.propagate_deterministic)
        latent = self.dynamics.get_state(posterior, deterministic=deterministic_latent)
        return latent, posterior

    def forward(self, batch):
        """
        Args:
            batch: Dict containing
                'obs': (T, B, C, H, W)
                'obs_features': (T, B, feat_dims)
                'action': (T, B, action_dims)
                'reward': (T, B)
        Return:
            output: Dict containing 'obs_recon'
        """
        obs_features = batch['obs_features']
        action = batch['action']
        T, B, _ = action.shape

        if self.encoder is not None:
            obs_features = self.encoder(obs_features)  # (T, B, obs_embed_dims)
        prev_state = self.dynamics.initial_state(B)
        prior_list = []
        posterior_list = []
        for t in range(T):
            prev_action = action[t-1] if t > 0 else None
            current_obs = obs_features[t]
            prior, posterior = self.dynamics(prev_state, prev_action, current_obs, deterministic=self.propagate_deterministic)
            prior_list.append(prior)
            posterior_list.append(posterior)
            prev_state = posterior
        prior = stack_tensor_dict_list(prior_list)
        posterior = stack_tensor_dict_list(posterior_list)
        latent_post = self.dynamics.get_state(posterior, deterministic=self.decode_deterministic)  # T, B, D
        latent_prior = self.dynamics.get_state(prior, deterministic=False)
        if self.reward_prediction_from_prior:
            reward_prediction = self.reward_net(latent_prior).squeeze(-1)  # DBC uses latent prior.
        else:
            reward_prediction = self.reward_net(latent_post).squeeze(-1)
        outputs = dict(prior=prior,
                       posterior=posterior,
                       reward_prediction=reward_prediction)
        if self.decoder is not None:
            outputs['obs_features_recon_post'] = self.decoder(latent_post)  # (T, B, dims)
            outputs['obs_features_recon_prior'] = self.decoder(latent_prior)  # (T, B, dims)
        if self.inv_dynamics is not None:
            paired_states = torch.cat([latent_post[:-1], latent_post[1:]], dim=-1)
            action_prediction = self.inv_dynamics(paired_states)
            outputs['action_prediction'] = action_prediction
        return outputs
