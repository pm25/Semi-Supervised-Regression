# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.
# Code in this file is adapted from xmed-lab/CLSS
# https://github.com/xmed-lab/CLSS/blob/main/age_estimation/models/OrdinalEntropy.py

import torch
import torch.nn.functional as F


def ordinal_entropy(features, targets):
    """
    Compute the ordinal entropy of features given targets.

    Args:
        features (torch.Tensor): Input features. Shape: (batch_size, feat_dim).
        targets (torch.Tensor): Target labels. Shape:  (batch_size,).

    Returns:
        torch.Tensor: Ordinal entropy.
    """
    if features.dim() != 2 or targets.dim() != 1 or features.size(0) != targets.size(0):
        raise ValueError("Input shapes are invalid.")

    batch_size, feat_dim = features.size()

    uni_values, uni_indices, uni_counts = torch.unique(targets, return_inverse=True, return_counts=True)

    center_feats = torch.zeros([len(uni_values), feat_dim], device=features.device)
    center_feats.index_add_(0, uni_indices, features)
    center_feats = center_feats / uni_counts.unsqueeze(1)

    norm_center_feats = F.normalize(center_feats, dim=1)
    distance = euclidean_dist(norm_center_feats, norm_center_feats)
    distance = flatten_upper_triangular(distance)

    _uni_values = uni_values.unsqueeze(1)
    weight = euclidean_dist(_uni_values, _uni_values)
    weight = flatten_upper_triangular(weight)
    weight = (weight - torch.min(weight)) / torch.max(weight) if len(weight) != 0 else 0

    distance = distance * weight
    entropy = torch.mean(distance)

    norm_feats = F.normalize(features, dim=1)
    norm_feats -= norm_center_feats[uni_indices, :]
    tightness = torch.sum(norm_feats.pow(2), dim=1)
    tightness = tightness[tightness > 0].mean()

    return tightness - entropy


def euclidean_dist(x, y):
    """
    Calculate Euclidean distance between two sets of vectors.

    Args:
        x (torch.Tensor): Set of vectors. Shape: (m, d).
        y (torch.Tensor): Set of vectors. Shape: (n, d).

    Returns:
        torch.Tensor: Pairwise Euclidean distance. Shape: (m, n).
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def flatten_upper_triangular(x):
    """
    Flatten the upper triangular elements of a square matrix.

    Args:
        x (torch.Tensor): Square matrix.

    Returns:
        torch.Tensor: Flattened upper triangular elements.
    """
    if len(x.shape) != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"Input tensor must be a square matrix, but got shape {x.shape}")
    n = x.shape[0]
    mask = torch.triu(torch.ones(n, n), diagonal=1).to(torch.bool)
    return x[mask]
