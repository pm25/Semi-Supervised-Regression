# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import torch.nn as nn
from torch.nn import functional as F


def consistency_loss(logits, targets, name="mse", mask=None):
    """
    consistency regularization loss in semi-supervised learning (regression).

    Args:
        logits: logit to calculate the loss on and back-propagation, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use mean-absolute-error ('l1') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ["l1", "mse"]
    # logits_w = logits_w.detach()
    if name == "l1":
        loss = F.l1_loss(logits, targets, reduction="none")
    else:
        loss = F.mse_loss(logits, targets, reduction="none")

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()


class ConsistencyLoss(nn.Module):
    """
    Wrapper for consistency loss
    """

    def forward(self, logits, targets, name="mse", mask=None):
        return consistency_loss(logits, targets, name, mask)
