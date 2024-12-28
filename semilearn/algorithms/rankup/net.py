# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import torch
import torch.nn as nn

from semilearn.nets.utils import init_weights


class RankUp_Net(nn.Module):
    """
    RankUp_Net implementation.

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

    def forward(self, x, use_arc=False, targets=None, **kwargs):
        if not use_arc:
            return self.backbone(x, **kwargs)
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        logits_arc = self.arc_classifier(feat)
        logits_mat, targets_mat = self.compute_rank_logits(logits_arc, targets)
        return {"logits": logits, "logits_arc": logits_mat, "feat": feat, "targets_arc": targets_mat}

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
        matcher = self.backbone.group_matcher(coarse, prefix="backbone.")
        return matcher
