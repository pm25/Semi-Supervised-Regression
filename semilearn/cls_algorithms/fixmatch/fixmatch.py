# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import torch

from semilearn.core import ClsAlgorithmBase
from semilearn.core.utils import CLS_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.cls_algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook


@CLS_ALGORITHMS.register("fixmatch")
class FixMatch(ClsAlgorithmBase):
    """
    FixMatch algorithm (https://arxiv.org/abs/2001.07685).

    Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
        - T (`float`):
            Temperature for pseudo-label sharpening
        - p_cutoff(`float`):
            Confidence threshold for generating pseudo-labels
        - hard_label (`bool`, *optional*, default to `False`):
            If True, targets have [Batch size] shape with int values. If False, the target is vector
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # fixmatch specified arguments
        self.cls_init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)

    def cls_init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, **kwargs):
        out_dict, log_dict = super().train_step(x_lb=x_lb, y_lb=y_lb, x_ulb_w=x_ulb_w, x_ulb_s=x_ulb_s, **kwargs)
        feats_dict = out_dict["feat"]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            outs_x_lb = self.model(feats_dict["x_lb"], use_arc=True, only_fc=True, targets=y_lb)
            logits_x_lb = outs_x_lb["logits"]
            y_lb = outs_x_lb["targets"]
            logits_x_ulb_s = self.model(feats_dict["x_ulb_s"], use_arc=True, only_fc=True)["logits"]
            with torch.no_grad():
                logits_x_ulb_w = self.model(feats_dict["x_ulb_w"], use_arc=True, only_fc=True)["logits"]

            sup_loss = self.cls_loss(logits_x_lb, y_lb, reduction="mean")

            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook(
                "gen_ulb_targets",
                "PseudoLabelingHook",
                logits=probs_x_ulb_w,
                use_hard_label=self.use_hard_label,
                T=self.T,
                softmax=False,
            )

            unsup_loss = self.cls_consistency_loss(logits_x_ulb_s, pseudo_label, "ce", mask=mask)

            total_cls_loss = sup_loss + self.cls_ulb_loss_ratio * unsup_loss
            total_loss = out_dict["loss"] + self.cls_loss_ratio * total_cls_loss

        out_dict["loss"] = total_loss
        log_dict["train_cls/sup_loss"] = sup_loss.item()
        log_dict["train_cls/unsup_loss"] = unsup_loss.item()
        log_dict["train_cls/total_loss"] = total_cls_loss.item()
        log_dict["train_cls/util_ratio"] = mask.float().mean().item()
        log_dict["train/total_loss"] = total_loss.item()
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument("--hard_label", str2bool, True),
            SSL_Argument("--T", float, 0.5),
            SSL_Argument("--p_cutoff", float, 0.95),
        ]
