# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import torch


@torch.no_grad()
def smooth_targets(logits, targets, smoothing=0.1):
    """
    label smoothing
    """
    true_dist = torch.zeros_like(logits)
    true_dist.fill_(smoothing / (logits.shape[-1] - 1))
    true_dist.scatter_(1, targets.data.unsqueeze(1), (1 - smoothing))
    return true_dist
