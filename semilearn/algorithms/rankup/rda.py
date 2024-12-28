# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import torch
import numpy as np

from semilearn.core.hooks import Hook


class RDAHook(Hook):
    """
    RDA Hook
    """

    def __init__(self, train_ulb_len, lb_targets, num_refine_iter=1024):
        super(RDAHook, self).__init__()
        self.train_ulb_len = train_ulb_len
        self.sorted_lb_targets, _ = torch.sort(torch.tensor(lb_targets))
        self.num_refine_iter = num_refine_iter

        self.pseudo_raw = torch.ones(self.train_ulb_len, dtype=torch.float32)
        self.pseudo_refine = torch.ones(self.train_ulb_len, dtype=torch.float32)

    @torch.no_grad()
    def gen_ulb_targets(self, algorithm, logits):
        logits = logits.detach()
        pseudo_label = self.refine_pseudo_labels(algorithm.idx_ulb, logits, algorithm.it, algorithm.epoch)
        return pseudo_label.to(logits.device)

    @torch.no_grad()
    def refine_pseudo_labels(self, idx_ulb, logits_x_ulb, it, epoch):
        self.pseudo_raw[idx_ulb.to(self.pseudo_raw.device)] = logits_x_ulb.data.cpu().to(self.pseudo_raw.dtype)
        if it % self.num_refine_iter == 0:
            self.apply_dist_align()
        if epoch > 0:
            logits_x_ulb = self.pseudo_refine[idx_ulb.to(self.pseudo_raw.device)].detach()
        return logits_x_ulb

    @torch.no_grad()
    def apply_dist_align(self):
        """
        Apply distribution alignment to refine pseudo labels.
        """
        cdf_pseudo = np.linspace(0, 1, len(self.pseudo_raw))
        cdf_target = np.linspace(0, 1, len(self.sorted_lb_targets))
        pseudo_refine = np.interp(cdf_pseudo, cdf_target, self.sorted_lb_targets.cpu().numpy())
        idxes = torch.argsort(self.pseudo_raw)
        self.pseudo_refine[idxes] = torch.FloatTensor(pseudo_refine).to(self.pseudo_refine.device)
