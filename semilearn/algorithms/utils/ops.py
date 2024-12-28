# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import torch
import numpy as np


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def mixup_one_target(x, y, alpha=1.0, is_bias=False):
    """Returns mixed inputs, mixed targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias:
        lam = max(lam, 1 - lam)

    index = torch.randperm(x.size(0)).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y, lam


@torch.no_grad()
def smooth_targets(logits, targets, smoothing=0.1):
    """
    label smoothing
    """
    true_dist = torch.zeros_like(logits)
    true_dist.fill_(smoothing / (logits.shape[-1] - 1))
    true_dist.scatter_(1, targets.data.unsqueeze(1), (1 - smoothing))
    return true_dist
