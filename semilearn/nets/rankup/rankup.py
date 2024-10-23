# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import torch
import torch.nn as nn

from semilearn.nets.utils import init_weights


class RankUp(nn.Module):
    """
    RankUp implementation with a auxiliary ranking classifier (ARC).

    Attributes:
        backbone (nn.Module): The underlying backbone model.
        num_features (int): Number of features from the model's hidden layer.
        arc_classifier (nn.Linear): Linear layer for Auxiliary Ranking Classifier (ARC) with two output classes.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.num_features = backbone.num_features

        # Auxiliary Ranking Classifier (ARC)
        self.arc_classifier = nn.Linear(self.num_features, 2)
        self.arc_classifier.apply(init_weights)

    def forward(self, x, use_arc=False, only_fc=False, targets=None, **kwargs):
        if not use_arc:
            return self.backbone(x, only_fc=only_fc, **kwargs)
        if only_fc:
            logits = self.arc_classifier(x)
            logits_mat, targets_mat = self.compute_rank_logits(logits, targets)
            return {"logits": logits_mat, "targets": targets_mat}
        feat = self.backbone(x, only_feat=True)
        logits = self.arc_classifier(feat)
        logits_mat, targets_mat = self.compute_rank_logits(logits, targets)
        return {"logits": logits_mat, "feat": feat, "targets": targets_mat}

    def compute_rank_logits(self, logits, targets=None):
        logits_mat = logits.unsqueeze(dim=0) - logits.unsqueeze(dim=1)
        logits_mat = logits_mat.flatten(0, 1)
        if targets is not None:
            targets_mat = (1 + torch.sign(targets.unsqueeze(dim=0) - targets.unsqueeze(dim=1))) / 2
            targets_mat = targets_mat.flatten(0, 1)
            # one-hot encode the targets_mat
            targets_onehot = torch.zeros((targets_mat.shape[0], 2)).to(targets_mat.device)
            targets_onehot[:, 0] = targets_mat
            targets_onehot[:, 1] = 1 - targets_mat
            return logits_mat, targets_onehot
        return logits_mat, None

    def group_matcher(self, coarse=False):
        if hasattr(self.backbone, "backbone"):
            # TODO: better way
            matcher = self.backbone.backbone.group_matcher(coarse, prefix="backbone.backbone")
        else:
            matcher = self.backbone.group_matcher(coarse, prefix="backbone.")
        return matcher


def rankup_wrapper(net_builder):
    def wrapper(*args, **kwargs):
        model = net_builder(*args, **kwargs)
        return RankUp(model)

    return wrapper
