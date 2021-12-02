#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
""" Replay Buffer for storing sequential data"""
import random
import torch
from utils import torchify


class SequenceReplayBuffer(object):
    def __init__(self, size=None):
        self.data = []
        self.size = size
        self.index = 0

    def __len__(self):
        return len(self.data)

    def add(self, seq):
        """
        seq : list of dict of key, value, where value is numpy array or float/int.
        """
        seq = torchify(seq)  # dict of (T, ...)
        if self.size is None or len(self.data) < self.size:
            self.data.append(seq)
        else:
            self.data[self.index] = seq
            self.index += 1
            if self.index == self.size:
                self.index = 0

    def sample(self, num_seq, seq_len=0):
        """
        Sample a batch from the replay buffer.
        Args:
            num_seq: Batch size
            seq_len: Length of each sequence. Default=0 means pick the entire sequence (i.e. seq_len=T)
        Returns:
            res : dict of tensors of shape (num_seq, seq_len, *entity_shape)
        """
        # Pick seq_ids.
        seq_count = len(self.data)
        inds = list(range(seq_count))
        if num_seq < seq_count:
            inds = random.sample(inds, k=num_seq)
        elif num_seq > seq_count:
            inds = random.choices(inds, k=num_seq)
        batch = []
        for ind in inds:
            seq = self.data[ind]
            key = list(seq.keys())[0]
            T = len(seq[key])
            if seq_len <= 0 or T <= seq_len:
                seq_sample = seq
            else:
                start_pos = random.randint(0, T - seq_len)
                seq_sample = {k: v[start_pos:start_pos + seq_len] for k, v in seq.items()}
            batch.append(seq_sample)
        # pack the batch into a dict of tensors.
        keys = batch[0].keys()
        res = {}
        for key in keys:
            res[key] = torch.stack([sample[key] for sample in batch])  # (B, T, ...)
        return res

    def all(self):
        return self.sample(len(self.data))
