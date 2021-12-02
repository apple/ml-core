#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

""" Wraps Distracting Control Suite in a Gym-like wrapper."""
import numpy as np
from distracting_control import suite as distracting_suite
from collections import deque
from PIL import Image
import argparse
import yaml
import matplotlib.pyplot as plt
plt.ion()


class EnvironmentContainerDCS(object):
    """
    Wrapper around DCS.
    """
    def __init__(self, config, train=True, seed=None):
        self.domain = config['domain']

        self.get_other_obs = False

        # The standard task and action_repeat for each domain.
        task_info = {
            'ball_in_cup': ('catch', 4),
            'cartpole': ('swingup', 8),
            'cheetah': ('run', 4),
            'finger': ('spin', 2),
            'reacher': ('easy', 4),
            'walker': ('walk', 2),
        }
        self.task, self.action_repeat = task_info[self.domain]

        self.difficulty = config['difficulty']
        if self.difficulty in ['none', 'None']:
            self.difficulty = None

        self.dynamic = config['dynamic']
        self.background_dataset_path = config.get('background_dataset_path', 'DAVIS/JPEGImages/480p')
        allow_color_distraction = config.get('allow_color_distraction', True)
        allow_background_distraction = config.get('allow_background_distraction', True)
        allow_camera_distraction = config.get('allow_camera_distraction', True)
        if seed is not None:
            task_kwargs = {'random': seed}
        else:
            task_kwargs = None
        self.env = distracting_suite.load(
            self.domain, self.task, difficulty=self.difficulty, dynamic=self.dynamic,
            background_dataset_path=self.background_dataset_path,
            background_dataset_videos='train' if train else 'val',
            pixels_only=False,
            task_kwargs=task_kwargs,
            allow_color_distraction=allow_color_distraction,
            allow_camera_distraction=allow_camera_distraction,
            allow_background_distraction=allow_background_distraction)

        action_spec = self.env.action_spec()
        self.action_dims = len(action_spec.minimum)
        self.action_low = action_spec.minimum
        self.action_high = action_spec.maximum
        self.num_frames_to_stack = config.get('num_frames_to_stack', 1)
        if self.num_frames_to_stack > 1:
            self.frame_queue = deque([], maxlen=self.num_frames_to_stack)
        self.config = config
        self.image_height, self.image_width = self.config['image_height'], self.config['image_width']
        self.num_channels = 3 * self.num_frames_to_stack
        self.other_dims = 0

    def get_action_dims(self):
        return self.action_dims

    def get_action_repeat(self):
        return self.action_repeat

    def get_action_limits(self):
        return self.action_low, self.action_high

    def get_obs_chw(self):
        return self.num_channels, self.image_height, self.image_width

    def get_obs_other_dims(self):
        return self.other_dims

    def reset(self):
        time_step = self.env.reset()
        if self.num_frames_to_stack > 1:
            self.frame_queue.clear()
        obs = self._get_image(time_step)  # C, H, W.
        obs_dict = {'image': obs}
        return obs_dict

    def step(self, action):
        reward = 0
        for _ in range(self.action_repeat):
            time_step = self.env.step(action)
            reward += time_step.reward
        obs = self._get_image(time_step)
        done = False
        info = {}
        obs_dict = {'image': obs}
        return obs_dict, reward, done, info

    def _get_image(self, time_step):
        image_height, image_width = self.config['image_height'], self.config['image_width']
        obs = time_step.observation['pixels'][:, :, 0:3]  # (240, 320, 3).
        # Resize to image_height, image_width
        obs = Image.fromarray(obs).resize((image_width, image_height), resample=Image.BILINEAR)
        #obs = cv2.resize(obs, dsize=(image_width, image_height))
        obs = np.asarray(obs)
        obs = obs.transpose((2, 0, 1)).copy()  # (C, H, W)
        obs = self._stack_images(obs)
        return obs

    def _stack_images(self, obs):
        if self.num_frames_to_stack > 1:
            if len(self.frame_queue) == 0:  # Just after reset.
                for _ in range(self.num_frames_to_stack):
                    self.frame_queue.append(obs)
            else:
                self.frame_queue.append(obs)
            obs = np.concatenate(list(self.frame_queue), axis=0)
        return obs


class EnvironmentContainerDCS_DMC_paired(EnvironmentContainerDCS):
    def __init__(self, config, train=True, seed=1):
        super().__init__(config, train=train, seed=seed)
        config_dmc = config.copy()
        config_dmc['difficulty'] = 'none'
        self.dmc = EnvironmentContainerDCS(config_dmc, train=train, seed=seed)

    def reset(self):
        obs_dict_dmc  = self.dmc.reset()
        obs_dict = super().reset()
        obs_dict['image_clean'] = obs_dict_dmc['image']
        return obs_dict

    def step(self, action):
        obs_dict_dmc, _, _, _ = self.dmc.step(action)
        obs_dict, reward, done, info = super().step(action)
        obs_dict['image_clean'] = obs_dict_dmc['image']
        return obs_dict, reward, done, info


def argument_parser(argument):
    """ Argument parser """
    parser = argparse.ArgumentParser(description='Binder Network.')
    parser.add_argument('-c', '--config', default='', type=str, help='Training config')
    args = parser.parse_args(argument)
    return args


def test():
    args = argument_parser(None)
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error opening specified config yaml at: {}. "
              "Please check filepath and try again.".format(args.config))

    config = config['parameters']
    seed = config['seed']
    np.random.seed(seed)
    env = EnvironmentContainerDCS_DMC_paired(config['env'], train=True, seed=config['seed'])
    plt.figure(1)
    action_low, action_high = env.get_action_limits()
    action_dims = env.get_action_dims()
    for _ in range(1):
        env.reset()
        for _ in range(1):
            action = np.random.uniform(action_low, action_high, action_dims)
            obs_dict, reward, done, info = env.step(action)
            obs_dcs = obs_dict['image'].transpose((1, 2, 0))
            obs_dmc = obs_dict['image_clean'].transpose((1, 2, 0))
            plt.clf()
            obs = np.concatenate([obs_dcs, obs_dmc], axis=1)
            plt.imshow(obs)
            plt.pause(0.001)
            filename = '/Users/nitish/Desktop/binder_figures/sample_0.png'
            plt.savefig(filename)


if __name__ == '__main__':
    test()
