#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch import distributions as pyd
import math


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def act_func(act_str):
    act = None
    if act_str == 'relu':
        act = nn.ReLU
    elif act_str == 'elu':
        act = nn.ELU
    elif act_str == 'identity':
        act = nn.Identity
    elif act_str == 'tanh':
        act = nn.Tanh
    elif act_str == 'sigmoid':
        act = nn.Sigmoid
    else:
        raise ValueError('Unknown activation function.')
    return act


class GRUCell(nn.Module):
    """
    GRUCell with optional layer norm.
    """
    def __init__(self, input_dims, hidden_dims, norm=False, update_bias=-1):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.linear = nn.Linear(input_dims + hidden_dims, 3 * hidden_dims, bias=not norm)
        self.norm = norm
        self.update_bias = update_bias
        if norm:
            self.norm_layer = nn.LayerNorm(3 * hidden_dims)

    def forward(self, inputs, state):
        gate_inputs = self.linear(torch.cat([inputs, state], dim=-1))
        if self.norm:
            gate_inputs = self.norm_layer(gate_inputs)
        reset, cand, update = gate_inputs.chunk(3, -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update + self.update_bias)
        output = update * cand + (1 - update) * state
        return output


class FCNet(nn.Module):
    """ MLP with fully-connected layers."""
    def __init__(self, config, in_features, out_features=None):
        super().__init__()
        layers = []
        fc_act = act_func(config['fc_activation'])
        self.input_dims = in_features
        num_hids = config['fc_hiddens']
        if out_features:
            num_hids.append(out_features)
        for i, num_hid in enumerate(num_hids):
            fc_layer = nn.Linear(in_features=in_features, out_features=num_hid)
            layers.append(fc_layer)
            if i < len(num_hids) - 1:
                layers.append(fc_act())
                if config.get('fc_batch_norm', False):
                    layers.append(nn.BatchNorm1d(num_hid))
            else:
                output_act = act_func(config['output_activation'])
                layers.append(output_act())
            in_features = num_hid
        if config.get('layer_norm_output', False):
            layers.append(nn.LayerNorm([in_features], elementwise_affine=config.get('layer_norm_affine', False)))
        self.fc_net = nn.Sequential(*layers)
        self.output_dims = in_features

    def forward(self, x):
        assert x.shape[-1] == self.input_dims, "Last dim is {} but should be {}".format(x.shape[-1], self.input_dims)
        orig_shape = list(x.shape)
        x = x.view(-1, self.input_dims)
        x = self.fc_net(x)
        orig_shape[-1] = self.output_dims
        x = x.view(*orig_shape)
        return x


class CNN(nn.Module):
    """ CNN, optionally followed by fully connected layers."""
    def __init__(self, config, chw):
        super().__init__()
        self.config = config
        channels, height, width = chw
        cnn_layers, channels, height, width = self.make_conv_layers((channels, height, width), config['conv_filters'])
        self.conv_net = nn.Sequential(*cnn_layers)
        num_features = channels * height * width
        if 'fc_hiddens' in config:
            fc_layers, num_features = self.make_fc_layers(num_features)
            self.fc_net = nn.Sequential(*fc_layers)
        else:
            self.fc_net = None
        self.output_dims = num_features

    def make_conv_layers(self, input_chw, conv_filters):
        channels, height, width = input_chw
        conv_act = act_func(self.config['conv_activation'])
        base_num_hid = self.config['base_num_hid']
        layers = []
        for i, filter_spec in enumerate(conv_filters):
            if len(filter_spec) == 3:  # Padding is not specified.
                num_filters, kernel_size, stride = filter_spec
                padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
            elif len(filter_spec) == 4:
                num_filters, kernel_size, stride, padding = filter_spec
            num_filters = num_filters * base_num_hid
            conv_layer = nn.Conv2d(channels, out_channels=num_filters, kernel_size=kernel_size, stride=stride,
                                   padding=padding)
            height = (height - kernel_size[0] + 2 * padding[0]) // stride + 1
            width = (width - kernel_size[1] + 2 * padding[1]) // stride + 1
            layers.append(conv_layer)
            layers.append(conv_act())
            if self.config['conv_batch_norm']:
                bn = nn.BatchNorm2d(num_filters)
                layers.append(bn)
            channels = num_filters
        return layers, channels, height, width

    def make_fc_layers(self, in_features):
        fc_act = act_func(self.config['fc_activation'])
        base_num_hid = self.config['base_num_hid']
        layers = []
        for i, num_hid in enumerate(self.config['fc_hiddens']):
            num_hid = num_hid * base_num_hid
            fc_layer = nn.Linear(in_features=in_features, out_features=num_hid)
            layers.append(fc_layer)
            if i < len(self.config['fc_hiddens']) - 1:
                layers.append(fc_act())
                if self.config.get('fc_batch_norm', False):
                    layers.append(nn.BatchNorm1d(num_hid))
            else:
                if self.config.get('layer_norm_output', False):
                    layers.append(nn.LayerNorm([num_hid], elementwise_affine=False))
                output_act = act_func(self.config['output_activation'])
                layers.append(output_act())
            in_features = num_hid
        return layers, in_features

    def forward(self, x):
        """
        Args:
            x : (..., C, H, W)
        Output:
            x: (..., D)
        """
        orig_shape = list(x.size())
        x = x.view(-1, *orig_shape[-3:])
        x = self.conv_net(x)
        if self.fc_net is not None:
            x = x.view(x.size(0), -1)  # Flatten.
            x = self.fc_net(x)
            x = x.view(*orig_shape[:-3], x.shape[-1])
        else:
            x = x.view(*orig_shape[:-3], *x.shape[-3:])
        return x


class TransposeCNN(nn.Module):
    """Decode images from a vector.
    """

    def __init__(self, config, input_size):
        """Initializes a ConvDecoder instance.

        Args:
            input_size (int): Input size, usually feature size output from
                RSSM.
            depth (int): Number of channels in the first conv layer
            act (Any): Activation for Encoder, default ReLU
            shape (List): Shape of observation input
        """
        super().__init__()

        layers = []
        in_features = input_size
        fc_act = act_func(config['fc_activation'])
        base_num_hid = config['base_num_hid']
        for i, num_hid in enumerate(config['fc_hiddens']):
            num_hid = num_hid * base_num_hid
            fc_layer = nn.Linear(in_features=in_features, out_features=num_hid)
            layers.append(fc_layer)
            layers.append(fc_act())
            if config.get('fc_batch_norm', False):
                layers.append(nn.BatchNorm1d(num_hid))
            in_features = num_hid
        self.fc_net = nn.Sequential(*layers)

        filters = config['conv_filters']
        act = act_func(config['conv_activation'])
        output_act = act_func(config['output_activation'])
        bn = config.get('conv_batch_norm', False)
        in_channels = in_features
        layers = []
        in_size = [1, 1]
        for i, (out_channels, kernel, stride) in enumerate(filters):
            if i < len(filters) - 1:
                out_channels = out_channels * base_num_hid
            layer = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride)
            layers.append(layer)
            out_size = [kernel[0] + (in_size[0] - 1) * stride, kernel[1] + (in_size[1] - 1) * stride]
            if i < len(filters) - 1:  # Don't put batch norm in the last layer, potentially different activation func.
                if act is not None:
                    layers.append(act())
                if bn:
                    layers.append(nn.BatchNorm2d(out_channels))
            else:
                layers.append(output_act())
            in_channels = out_channels
            in_size = out_size

        self.conv_transpose = nn.Sequential(*layers)
        self.out_size = in_size
        self.out_channels = in_channels

    def forward(self, x):
        """
        Args:
            x: (..., D)
        Output:
            x : (..., C, H, W)
        """
        orig_shape = list(x.size())
        x = x.view(-1, orig_shape[-1])
        x = self.fc_net(x)
        C = x.shape[-1]
        x = x.view(-1, C, 1, 1)
        x = self.conv_transpose(x)
        out_shape = list(x.size())
        res_shape = orig_shape[:-1] + out_shape[1:]
        x = x.view(*res_shape)
        return x


class ObservationModel(nn.Module):
    """
    Module that encapsulates the observation encoder (and optionally, decoder).
    """
    def __init__(self, config, image_chw, other_dims=0):
        """
        Inputs:
            image_chw: (channels, height, width) for the image observations.
            other_dims : int, any extra dimensions, e.g. proprioceptive state.
        """
        super().__init__()

        # Gating network.
        self.use_gating_network = config.get('use_gating_network', False)
        if self.use_gating_network:
            self.gating_net = CNN(config['encoder_gating'], image_chw)
            self.gating_net_channels = image_chw[0]

        # Image encoder.
        self.encoder = CNN(config['encoder'], image_chw)
        self.output_dims = self.encoder.output_dims

        # (Optional) Proprioceptive state encoder.
        if 'other_obs_model' in config:
            print('Proprioceptive state has {} dims'.format(other_dims))
            self.other_model = FCNet(config['other_obs_model'], other_dims)
            self.output_dims += self.other_model.output_dims
        else:
            self.other_model = None

        if 'decoder' in config:
            # Image predictor P(x_t | z_t, h_t)
            self.decoder = TransposeCNN(config['decoder'], self.output_dims)
        else:
            self.decoder = None

        self.apply(weight_init)

    def encode_pixels(self, x, encoder=None):
        if self.use_gating_network:
            orig_shape = list(x.shape)
            input_channels = orig_shape[-3]
            num_frames = input_channels // self.gating_net_channels
            # 9 input_channels = 3 frames * 3 (RGB) channels.
            # Gating net is shared across stacked frames.
            x = x.view(*orig_shape[:-3], num_frames, self.gating_net_channels, *orig_shape[-2:])
            g = torch.sigmoid(self.gating_net(x))
            assert g.shape[-3] == 1  # Gating should give one scalar per location.
            x = x * g
            x = x.view(*orig_shape)
        else:
            g = None
        
        if encoder is None:
            x = self.encoder(x)
        else:
            x = encoder(x)
        return x, g

    def encode(self, batch):
        """
        Encode the observations.
        Inputs:
            batch: dict containing 'obs_image' and optionally, other observations.
        Returns:
            outputs: dict containing 'obs_encoding', and optionally 'obs_gating'.
        """
        outputs = {}
        x, g = self.encode_pixels(batch['obs_image'])
        if self.use_gating_network:
            outputs['obs_gating'] = g
        
        if self.other_model is not None:
            other_obs = batch['obs_other']
            other_encoding = self.other_model(other_obs)
            x = torch.cat([x, other_encoding], dim=-1)
        outputs['obs_features'] = x
        return outputs

    def decode(self, x):
        assert self.decoder is not None
        return self.decoder(x)

    def forward(self, x):
        return self.encode(x)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        try:
            self.base_dist = pyd.Normal(loc, scale)
        except Exception as e:
            print(e)
            print('Loc mean: ', loc.mean())
            print('Loc: ', loc)
            raise e
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class TanhGaussianPolicy(nn.Module):
    def __init__(self, config, state_dims, action_dims):
        super().__init__()
        self.fc_net = FCNet(config, state_dims, action_dims * 2)
        self.log_std_min = config.get("policy_min_logstd", -10)
        self.log_std_max = config.get("policy_max_logstd", 2)
        self.tiny = 1.e-7
        self.clip = 1 - self.tiny
        self.apply(weight_init)

    def forward(self, x, sample=True):
        x = self.fc_net(x)
        mu, log_std = torch.chunk(x, 2, dim=-1)
        mu = 5 * torch.tanh(mu / 5)
        log_std = torch.sigmoid(log_std)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * log_std
        std = torch.exp(log_std)
        action_dist = SquashedNormal(mu, std)
        if sample:
            action_sample = action_dist.rsample()
            action_sample = action_sample.clamp(-self.clip, self.clip)
        else:
            action_sample = torch.tanh(mu)
        log_prob = action_dist.log_prob(action_sample).sum(dim=-1)
        return action_sample, log_prob, std


class DoubleCritic(nn.Module):
    def __init__(self, config, state_dims, action_dims):
        super().__init__()
        self.critic1 = FCNet(config, state_dims + action_dims)
        self.critic2 = FCNet(config, state_dims + action_dims)
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.apply(weight_init)

    def forward(self, states, actions):
        assert states.shape[-1] == self.state_dims
        assert actions.shape[-1] == self.action_dims
        inputs = torch.cat([states, actions], dim=-1)
        q1 = self.critic1(inputs).squeeze(-1)
        q2 = self.critic2(inputs).squeeze(-1)
        return q1, q2


class ContrastivePrediction(nn.Module):
    def __init__(self, config, obs_dims):
        super().__init__()
        init_inverse_temp = config['inverse_temperature_init']
        inverse_temp = torch.tensor([float(init_inverse_temp)])
        self.inverse_temp = nn.parameter.Parameter(inverse_temp)
        self.output_dims = obs_dims
        self.softmax_over = config.get('softmax_over', 'both')  # ['obs', 'symbol', 'both']
        mask_type = config.get('mask_type', None)
        if mask_type is None:
            self.mask = None
        else:
            if mask_type == 'exclude_same_sequence':
                mask = self.get_exclude_same_sequence_mask(31, 32)
            elif mask_type == 'exclude_other_sequences':
                mask = self.get_exclude_other_sequences_mask(31, 32)
            else:
                raise Exception('Unknown mask type')
            self.register_buffer('mask', mask)

    def get_exclude_same_sequence_mask(self, T, B):
        """
        Exclude other timesteps from the same sequence.
        mask[i, j] = True means replace that by -inf.
        """
        mask = torch.zeros(T, B, T, B)
        per_b_mask = 1 - torch.ones(T, T)
        for b in range(B):
            mask[:, b, :, b] = per_b_mask
        mask = mask.view(T*B, T*B)
        mask = mask == 1
        return mask

    def get_exclude_other_sequences_mask(self, T, B):
        """
        Exclude other sequences in the batch.
        mask[i, j] = True means replace that by -inf.
        """
        mask = torch.ones(T, B, T, B)
        for b in range(B):
            mask[:, b, :, b] = 0
        mask = mask.view(T*B, T*B)
        mask = mask == 1
        return mask

    def forward(self, x, y, train=False):
        """
        Inputs:
            x: (..., D) encoder features.
            y: (..., D) decoder features.
        Outputs:
            loss: contrastive loss.
        """
        x = x.view(-1, self.output_dims)  # (T * B, D)
        y = y.view(-1, self.output_dims)  # (T * B, D)
        inv_temp = F.softplus(self.inverse_temp)
        logits = inv_temp * torch.matmul(x, y.T)  # (B', B')
        if self.mask is not None and train:
            logits[self.mask] = float('-inf')
        log_probs1 = F.log_softmax(logits, dim=1)
        log_probs2 = F.log_softmax(logits, dim=0)
        loss1 = -(log_probs1.diagonal().mean())
        loss2 = -(log_probs2.diagonal().mean())
        if self.softmax_over == 'symbol':
            loss = loss1
        elif self.softmax_over == 'obs':
            loss = loss2
        else:
            loss = (loss1 + loss2) / 2
        return loss
