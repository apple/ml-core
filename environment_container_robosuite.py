#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import argparse
import os
import numpy as np
import yaml
from collections import deque
import robosuite as suite
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS
from mujoco_py import MujocoException
from robosuite.wrappers.domain_randomization_wrapper import DomainRandomizationWrapper, DEFAULT_CAMERA_ARGS,\
    DEFAULT_COLOR_ARGS, DEFAULT_LIGHTING_ARGS
import matplotlib.pyplot as plt
plt.ion()


def dict_merge(default, user):
    d = default.copy()
    d.update(user)
    return d


def load_robosuite_controller_config(controller):
    if controller in set(ALL_CONTROLLERS):
        # This is a default controller
        controller_config = load_controller_config(default_controller=controller)
    else:
        # This is a string to the custom controller
        controller_config = load_controller_config(custom_fpath=controller)
    return controller_config


class EnvironmentContainerRobosuite(object):
    def __init__(self, config, train=True, seed=None):
        super().__init__()
        self.config = config
        robosuite_config = config['robosuite_config']
        self.crop_image = config.get('crop_image', False)
        if self.crop_image:
            self.image_height = config['crop_height']
            self.image_width = config['crop_width']
            self.crop_center_xy = config['crop_center_xy']
            cx, cy = self.crop_center_xy
            self.crop_left = cx - self.image_width // 2
            self.crop_top = cy - self.image_height // 2
        else:
            self.image_height = robosuite_config['camera_heights']
            self.image_width = robosuite_config['camera_widths']
        
        self.dr = config.get('domain_randomize', False)
        dr_config = config.get('domain_randomization_config', None)
        controller = config['controller']
        robosuite_config['controller_configs'] = load_robosuite_controller_config(controller)

        self.env = suite.make(**robosuite_config)
        self.robosuite_config = robosuite_config
        self.env_name = self.robosuite_config['env_name']

        self.image_key = config['image_key']
        if 'other_key' in config:
            self.other_key = config['other_key']
            self.get_other_obs = True
            ob_dict = self.env.reset()
            assert self.other_key in ob_dict
            self.other_dims = len(ob_dict[self.other_key])
        else:
            self.get_other_obs = False
            self.other_dims = 0

        low, high = self.env.action_spec
        self.action_dims = len(low)
        self.action_repeat = 1

        if self.dr:
            dr_config['color_randomization_args'] = dict_merge(DEFAULT_COLOR_ARGS,
                                                               dr_config.get('color_randomization_args', {}))
            dr_config['camera_randomization_args'] = dict_merge(DEFAULT_CAMERA_ARGS,
                                                                dr_config.get('camera_randomization_args', {}))
            dr_config['lighting_randomization_args'] = dict_merge(DEFAULT_LIGHTING_ARGS,
                                                                  dr_config.get('lighting_randomization_args', {}))
            self.env = DomainRandomizationWrapper(self.env, seed=seed, **dr_config)

        self.num_frames_to_stack = config.get('num_frames_to_stack', 1)
        if self.num_frames_to_stack > 1:
            self.frame_queue = deque([], maxlen=self.num_frames_to_stack)
        self.num_channels = 3 * self.num_frames_to_stack
        self.other_dims = self.other_dims * self.num_frames_to_stack
        self.ob_dict = None  # To hold the last raw observation.

    def get_action_dims(self):
        return self.action_dims

    def get_action_repeat(self):
        return self.action_repeat

    def get_action_limits(self):
        low, high = self.env.action_spec
        return low, high

    def get_obs_chw(self):
        return self.num_channels, self.image_height, self.image_width

    def get_obs_other_dims(self):
        return self.other_dims

    def preprocess_image(self, img):
        # If this is an RGB image, make it C, H, W.
        # This is done here rather than just before fprop so that
        # if/when depth is added, we can concatenate it as a channel.
        if len(img.shape) == 3:  # RGB image
            if self.crop_image:
                x, y = self.crop_left, self.crop_top
                img = img[y:y + self.image_height, x:x + self.image_width, :]
        elif len(img.shape) == 2:  # Depth image
            if self.crop_image:
                x, y = self.crop_left, self.crop_top
                img = img[y:y + self.image_height, x:x + self.image_width]
        if len(img.shape) >= 2:
            img = img[::-1, ...].copy()  # The frontview image is upside-down.
            img = img.transpose(2, 0, 1)  # CHW
        return img

    def _get_obs(self, obs_dict, verbose=False):
        assert self.image_key in obs_dict, "key {} not found in obs".format(self.image_key)
        img = self.preprocess_image(obs_dict[self.image_key])
        res = dict(image=img)
        if self.get_other_obs:
            assert self.other_key in obs_dict, "key {} not found in obs".format(self.other_key)
            res['other'] = obs_dict[self.other_key]
        if self.num_frames_to_stack > 1:
            res = self._get_stacked_obs(res)
        return res

    def _get_stacked_obs(self, obs):
        if len(self.frame_queue) == 0:
            for _ in range(self.num_frames_to_stack):
                self.frame_queue.append(obs)
        else:
            self.frame_queue.append(obs)
        keys = obs.keys()
        res = {}
        for key in keys:
            res[key] = np.concatenate([frame[key] for frame in self.frame_queue])
        return res

    def reset(self):
        if self.num_frames_to_stack > 1:
            self.frame_queue.clear()
        self.ob_dict = self.env.reset()
        #if self.dr:   # Fix for resetting bug. Domain is not getting randmized for the obs coming from reset.
        action = np.zeros(self.action_dims)
        self.ob_dict, _, _, _ = self.env.step(action)
        obs = self._get_obs(self.ob_dict)
        return obs

    def step(self, action):
        try:
            ob_dict, reward, done, info = self.env.step(action)
        except MujocoException as e:
            print('MujocoException', e)
            print('Will skip this action')
            if self.ob_dict is not None:
                ob_dict = self.ob_dict
            reward = 0
            done = False
            info = {}
        self.ob_dict = ob_dict
        # Additional reward shaping.
        #if self.env_name == 'Door':
        #    # Additional reward shaping for door angle.
        #    if reward < 1:
        #        hinge_qpos = self.env.sim.data.qpos[self.env.hinge_qpos_addr]
        #        reward += np.clip(0.5 * hinge_qpos / 0.3, 0, 0.5)
        obs = self._get_obs(ob_dict)
        return obs, reward, done, info

    def render(self, mode, **kwargs):
        if mode == 'rgb_array':
            image_list = []
            for (cam_name, cam_w, cam_h, cam_d) in \
                    zip(self.env.camera_names, self.env.camera_widths, self.env.camera_heights, self.env.camera_depths):

                # Add camera observations to the dict
                camera_obs = self.env.sim.render(
                    camera_name=cam_name,
                    width=cam_w,
                    height=cam_h,
                    depth=cam_d
                )
                if cam_d:
                    img, depth = camera_obs
                    camera_obs = np.concatenate([img, depth[:, :, None]], axis=2)
                image_list.append(camera_obs)
            image = np.concatenate(image_list, axis=1)
            return image  # return RGB frame suitable for video
        elif mode == 'human':
            self.env.render()  # pop up a window and render
        else:
            raise NotImplementedError

def argument_parser(argument):
    """ Argument parser """
    parser = argparse.ArgumentParser(description='Binder Network.')
    parser.add_argument('-c', '--config', default='', type=str, help='Training config')
    args = parser.parse_args(argument)
    return args



def test2():
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
    env = EnvironmentContainerRobosuite(config['env'])
    obs_dict = env.reset()
    action_low, action_high = env.get_action_limits()
    action_dims = env.get_action_dims()
    plt.figure(1)
    obs_list = []
    for ii in range(2):
        obs = obs_dict['image'].transpose((1, 2, 0))
        obs_list.append(obs)
        plt.clf()
        plt.imshow(obs)
        plt.suptitle('Image {}'.format(ii))
        plt.pause(0.5)
        action = np.random.uniform(action_low, action_high, action_dims)
        obs_dict, reward, done, info = env.step(action)
    obs_list.append(np.abs(obs_list[-1] - obs_list[-2]))
    obs = np.concatenate(obs_list, axis=1)
    plt.imshow(obs)
    plt.axis('off')
    plt.show()
    input('Press enter')


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
    plt.figure(1)
    randomize_settings = [(False, False),  (False, True), (True, True)]
    obs_list = []
    for robot in ['Panda', 'Jaco']:
        for randomize_camera, randomize_other in randomize_settings:
            config['env']['domain_randomize'] = True
            config['env']['robosuite_config']['robots'] = [robot]
            config['env']['domain_randomization_config']['randomize_camera'] = randomize_camera
            config['env']['domain_randomization_config']['randomize_color'] = randomize_other
            config['env']['domain_randomization_config']['randomize_lighting'] = randomize_other
            env = EnvironmentContainerRobosuite(config['env'], seed=seed)
            env.reset()
            action_low, action_high = env.get_action_limits()
            action_dims = env.get_action_dims()
            action = np.random.uniform(action_low, action_high, action_dims)
            obs_dict, reward, done, info = env.step(action)
            obs = obs_dict['image'].transpose((1, 2, 0))
            plt.clf()
            plt.imshow(obs)
            plt.pause(0.001)
            obs_list.append(obs)
    obs1 = np.concatenate(obs_list[:3], axis=0)
    obs2 = np.concatenate(obs_list[3:], axis=0)
    obs = np.concatenate([obs1, obs2], axis=1)
    plt.draw()
    plt.imshow(obs)
    plt.axis('off')
    plt.show()
    plt.savefig('/Users/nitish/Desktop/binder_figures/robosuite_supp_new.png', bbox_inches='tight')

if __name__ == '__main__':
    test()
