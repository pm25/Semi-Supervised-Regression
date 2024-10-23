# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import numpy as np

import torch
import torch.nn.functional as F

from semilearn.core import ClsAlgorithmBase
from semilearn.core.utils import CLS_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument, str2bool, mixup_one_target


@CLS_ALGORITHMS.register("mixmatch")
class MixMatch(ClsAlgorithmBase):
    """
    MixMatch algorithm (https://arxiv.org/abs/1905.02249).

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
        - cls_unsup_warm_up (`float`, *optional*, defaults to 0.4):
            Ramp up for weights for unsupervised loss
        - cls_mixup_alpha (`float`, *optional*, defaults to 0.5):
            Hyper-parameter of mixup
        - cls_mixup_manifold (`bool`, *optional*, defaults to `False`):
            Whether or not to use manifold mixup
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # mixmatch specified arguments
        self.cls_init(
            T=args.T,
            cls_unsup_warm_up=args.cls_unsup_warm_up,
            cls_mixup_alpha=args.cls_mixup_alpha,
            cls_mixup_manifold=args.cls_mixup_manifold,
        )

    def cls_init(self, T, cls_unsup_warm_up=0.01525, cls_mixup_alpha=0.5, cls_mixup_manifold=False):
        self.T = T
        self.cls_unsup_warm_up = cls_unsup_warm_up
        self.cls_mixup_alpha = cls_mixup_alpha
        self.cls_mixup_manifold = cls_mixup_manifold
        self.num_classes = 2

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_w_2, **kwargs):
        out_dict, log_dict = super().train_step(x_lb=x_lb, y_lb=y_lb, x_ulb_w=x_ulb_w, x_ulb_w_2=x_ulb_w_2, **kwargs)
        feats_dict = out_dict["feat"]
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            with torch.no_grad():
                self.bn_controller.freeze_bn(self.model)
                logits_x_ulb_w1 = self.model(feats_dict["x_ulb_w"], use_arc=True, only_fc=True)["logits"]
                logits_x_ulb_w2 = self.model(feats_dict["x_ulb_w_2"], use_arc=True, only_fc=True)["logits"]
                self.bn_controller.unfreeze_bn(self.model)

                # avg
                # avg_prob_x_ulb = (torch.softmax(logits_x_ulb_w1, dim=1) + torch.softmax(logits_x_ulb_w2, dim=1)) / 2
                avg_prob_x_ulb = (self.compute_prob(logits_x_ulb_w1) + self.compute_prob(logits_x_ulb_w2)) / 2
                # avg_prob_x_ulb = (avg_prob_x_ulb / avg_prob_x_ulb.sum(dim=-1, keepdim=True))
                # sharpening
                sharpen_prob_x_ulb = avg_prob_x_ulb ** (1 / self.T)
                sharpen_prob_x_ulb = (sharpen_prob_x_ulb / sharpen_prob_x_ulb.sum(dim=-1, keepdim=True)).detach()

            # with torch.no_grad():
            # Pseudo Label
            input_labels = torch.cat([F.one_hot(y_lb, self.num_classes), sharpen_prob_x_ulb, sharpen_prob_x_ulb], dim=0)
            # Mix up
            if self.mixup_manifold:
                inputs = torch.cat((feats_dict["x_lb"], feats_dict["x_ulb_w"], feats_dict["x_ulb_w_2"]))
            else:
                inputs = torch.cat([x_lb, x_ulb_w, x_ulb_w_2])
            mixed_x, mixed_y, _ = mixup_one_target(inputs, input_labels, self.cls_mixup_alpha, is_bias=True)
            mixed_x = list(torch.split(mixed_x, num_lb))
            # mixed_x = interleave(mixed_x, num_lb)

            if self.mixup_manifold:
                logits = [self.model(mixed_x[0], use_arc=True, only_fc=self.cls_mixup_manifold)["logits"]]
                # calculate BN for only the first batch
                self.bn_controller.freeze_bn(self.model)
                for ipt in mixed_x[1:]:
                    logits.append(self.model(ipt, use_arc=True, only_fc=self.mixup_manifold)["logits"])
                self.bn_controller.unfreeze_bn(self.model)
            else:
                logits = [self.model(mixed_x[0], use_arc=True)["logits"]]
                # calculate BN for only the first batch
                self.bn_controller.freeze_bn(self.model)
                for ipt in mixed_x[1:]:
                    logits.append(self.model(ipt, use_arc=True)["logits"])
                self.bn_controller.unfreeze_bn(self.model)

            # put interleaved samples back
            # logits = interleave(logits, num_lb)

            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            sup_loss = self.cls_loss(logits_x, mixed_y[:num_lb], reduction="mean")
            unsup_loss = self.cls_consistency_loss(logits_u, mixed_y[num_lb:], name="mse")

            # set ramp_up for lambda_u
            unsup_warmup = float(np.clip(self.it / (self.unsup_warm_up * self.num_train_iter), 0.0, 1.0))
            total_cls_loss = sup_loss + self.cls_loss_ratio * unsup_loss * unsup_warmup

            total_loss = out_dict["loss"] + self.cls_loss_ratio * total_cls_loss

        out_dict["loss"] = total_loss
        log_dict["train_cls/sup_loss"] = sup_loss.item()
        log_dict["train_cls/unsup_loss"] = unsup_loss.item()
        log_dict["train_cls/total_loss"] = total_cls_loss.item()
        log_dict["train/total_loss"] = total_loss.item()
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument("--T", float, 0.5, "parameter for Temperature Sharpening"),
            SSL_Argument("--cls_unsup_warm_up", float, 1 / 64, "ramp up ratio for unsupervised loss"),
            SSL_Argument("--cls_mixup_alpha", float, 0.5, "parameter for Beta distribution of Mix Up"),
            SSL_Argument("--cls_mixup_manifold", str2bool, False, "use manifold mixup (for nlp)"),
        ]
