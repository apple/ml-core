#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import random
import string
import time
import torch
from torchvision.transforms import Resize, RandomCrop, CenterCrop
import subprocess as sp
import numpy as np
import os
import imageio


def torchify(seq):
    """
    Convert list of dict of numpy arrays/floats/dicts to dict of tensors.
    Args:
        seq : List of dicts.
    Returns:
        batch : Dict of tensors of shape (T, ..).
    """
    keys = seq[0].keys()
    batch = {}
    for key in keys:
        value = seq[0][key]
        if isinstance(value, np.ndarray):
            batch[key] = torch.stack([torch.from_numpy(frame[key]) for frame in seq])  # (T, ..)
        elif isinstance(value, float) or isinstance(value, int):
            batch[key] = torch.tensor([frame[key] for frame in seq])
        elif isinstance(value, dict):
            sub_batch = torchify([frame[key] for frame in seq])
            for sub_key, val in sub_batch.items():
                batch[key + '_' + sub_key] = val
        else:
            raise Exception('Unknown type of value in torchify for key ', key)
    return batch


class FreezeParameters:
    def __init__(self, parameters):
        self.parameters = parameters
        self.param_states = [p.requires_grad for p in self.parameters]

    def __enter__(self):
        for param in self.parameters:
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(self.parameters):
            param.requires_grad = self.param_states[i]


def freeze_parameters(parameters):
    for param in parameters:
        param.requires_grad = False


def add_weight_decay(net, l2_value, skip_list=(), exclude_both_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad or name in exclude_both_list:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


def get_random_string(length=12):
    # choose from all lowercase letter
    letters = string.ascii_lowercase + string.digits
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def generate_expt_id():
    task_id = get_random_string()
    return time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + task_id


def features_as_image(x, x_min=None, x_max=None):
    """
    x : (B, D) or (1, B, D)
    """
    assert len(x.shape) == 2 or (len(x.shape) == 3 and x.shape[0] == 1)
    if x_max is None:
        x_max = x.max()
    if x_min is None:
        x_min = x.min()
    x = (x - x_min) / (x_max - x_min)
    if len(x.shape) == 2:
        x = x.unsqueeze(0)  # (1, H, W)
    return x


def probs_as_images(probs, W=64):
    """
    Inputs:
        probs: (..., D)
    Outputs:
        vid : (..., C, H, W), C = 3, W = 32
    """
    orig_shape = list(probs.shape)
    D = orig_shape[-1]
    H = D // W
    assert D % W == 0
    probs = (probs * 255).to(torch.uint8)
    C = 3
    probs = probs.view(-1, 1, H, W).expand(-1, C, -1, -1).contiguous()
    probs = probs.view(*orig_shape[:-1], C, H, W)
    return probs


def resize_image(images, height, width):
    """
    Resize images.
    Inputs:
        images: (..., C, H, W)
    Outputs:
        images: (..., C, height, width)
    """
    orig_shape = list(images.shape)
    C, H, W = orig_shape[-3:]
    images = images.view(-1, C, H, W)
    resize_op = Resize((height, width))
    images = resize_op(images)
    images = images.view(*orig_shape[:-2], height, width)
    return images


def crop_image_tensor(obs, crop_height, crop_width, random_crop=False, same_crop_across_time=False, padding=0):
    """
    Crop a tensor of images.
    Args:
        obs: (B, T, C, H, W), or (B, C, H, W) or (C, H, W).
        crop_height: Height of the cropped image.
        crop_width: Width of the cropped image.
        random_crop: If true, crop random patch. Otherwise the center crop is returned.
        same_crop_across_time: Maintain the same crop across time for temporal sequences.
        padding: How much edge padding to add.
    Returns:
        cropped_obs: (B, T, C, crop_height, crop_width)
    """
    assert len(obs.shape) >= 3
    channels, height, width = obs.shape[-3], obs.shape[-2], obs.shape[-1]
    if random_crop:
        transform = RandomCrop((crop_height, crop_width), padding=padding, padding_mode='edge')
        orig_shape = list(obs.shape[:-2])
        if same_crop_across_time and len(obs.shape) >= 5:
            T = obs.shape[-4]
            channels = channels * T
        obs = obs.view(-1, channels, height, width)
        cropped_obs = torch.zeros(obs.size(0), channels, crop_height, crop_width, dtype=obs.dtype, device=obs.device)
        for i in range(obs.size(0)):
            cropped_obs[i, ...] = transform(obs[i, ...])
        cropped_obs = cropped_obs.view(*orig_shape, crop_height, crop_width)
    else:
        transform = CenterCrop((crop_height, crop_width))
        cropped_obs = transform(obs)
    return cropped_obs


def get_parameter_list(optimizer):
    params_list = []
    for group in optimizer.param_groups:
        params_list.extend(list(group['params']))
    return params_list


def stack_tensor_dict_list(tensor_dict_list):
    """ Stack tensors in a list of dictionaries. """
    keys = tensor_dict_list[0].keys()
    res = {}
    for key in keys:
        res[key] = torch.stack([d[key] for d in tensor_dict_list])
    return res


def write_video_mp4_command(filename, frames):
    """
    frames : T, C, H, W
    """
    if isinstance(frames, np.ndarray):
        T, channels, height, width = frames.shape
    elif isinstance(frames, torch.Tensor):
        T, channels, height, width = frames.shape
        frames = frames.detach().cpu().numpy()
    elif isinstance(frames, list):
        channels, height, width = frames[0].shape
        assert channels >= 3
        frames = np.stack([frame[:3, ...] for frame in frames], axis=0)
    else:
        raise Exception('Unknown frame specification.')
    frames = frames.astype(np.uint8)
    frames = frames.transpose((0, 2, 3, 1))
    print('Writing video {}'.format(filename))
    # Write out as a mp4 video.
    command = ['ffmpeg',
               '-y',  # (optional) overwrite output file if it exists
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-s', '{}x{}'.format(width, height),  # size of one frame
               '-pix_fmt', 'rgb24',
               '-r', '20',  # frames per second
               '-an',  # Tells FFMPEG not to expect any audio
               '-i', '-',  # The input comes from a pipe
               '-vcodec', 'libx264',
               '-pix_fmt', 'yuv420p',
               filename]
    print(' '.join(command))
    proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
    outs, errs = proc.communicate(input=frames.tobytes())


def write_video_mp4(filename, frames):
    """
    frames : T, C, H, W
    """
    if isinstance(frames, np.ndarray):
        T, channels, height, width = frames.shape
    elif isinstance(frames, torch.Tensor):
        T, channels, height, width = frames.shape
        frames = frames.detach().cpu().numpy()
    elif isinstance(frames, list):
        channels, height, width = frames[0].shape
        assert channels >= 3
        frames = np.stack([frame[:3, ...] for frame in frames], axis=0)
    else:
        raise Exception('Unknown frame specification.')
    frames = frames.astype(np.uint8)
    frames = frames.transpose((0, 2, 3, 1))
    print('Writing video {}'.format(filename))
    # Write out as a mp4 video.
    writer = imageio.get_writer(filename, fps=20)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def test_write_video_mp4():
    filename = 'test.mp4'
    frames = np.random.rand(100, 3, 128, 128) * 255
    write_video_mp4(filename,  frames)


if __name__ == '__main__':
    test_write_video_mp4()
