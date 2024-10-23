# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import numpy as np

from semilearn.core import ClsAlgorithmBase
from semilearn.core.utils import CLS_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument


@CLS_ALGORITHMS.register("pimodel")
class PiModel(ClsAlgorithmBase):
    """
    Pi-Model algorithm (https://arxiv.org/abs/1610.02242).

    Args:
    - args (`argparse`):
        algorithm arguments
    - net_builder (`callable`):
        network loading function
    - tb_log (`TBLog`):
        tensorboard logger
    - logger (`logging.Logger`):
        logger to use
    - cls_unsup_warm_up (`float`, *optional*, defaults to 0.4):
        Ramp up for weights for unsupervised loss
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        self.cls_init(cls_unsup_warm_up=args.cls_unsup_warm_up)

    def cls_init(self, cls_unsup_warm_up=0.4):
        self.cls_unsup_warm_up = cls_unsup_warm_up

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_w_2, **kwargs):
        out_dict, log_dict = super().train_step(x_lb=x_lb, y_lb=y_lb, x_ulb_w=x_ulb_w, x_ulb_w_2=x_ulb_w_2, **kwargs)
        feats_dict = out_dict["feat"]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            outs_x_lb = self.model(feats_dict["x_lb"], use_arc=True, only_fc=True, targets=y_lb)
            logits_x_lb = outs_x_lb["logits"]
            y_lb = outs_x_lb["targets"]

            # calculate BN only for the first batch
            self.bn_controller.freeze_bn(self.model)
            logits_x_ulb_w = self.model(feats_dict["x_ulb_w"], use_arc=True, only_fc=True)["logits"]
            logits_x_ulb_w_2 = self.model(feats_dict["x_ulb_w_2"], use_arc=True, only_fc=True)["logits"]
            self.bn_controller.unfreeze_bn(self.model)

            sup_loss = self.cls_loss(logits_x_lb, y_lb, reduction="mean")
            unsup_loss = self.cls_consistency_loss(
                logits_x_ulb_w_2,
                # torch.softmax(logits_x_ulb_w.detach(), dim=-1),
                self.compute_prob(logits_x_ulb_w.detach()),
                "mse",
            )

            unsup_warmup = np.clip(self.it / (self.cls_unsup_warm_up * self.num_train_iter), a_min=0.0, a_max=1.0)
            total_cls_loss = sup_loss + self.cls_ulb_loss_ratio * unsup_loss * unsup_warmup

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
            SSL_Argument("--cls_unsup_warm_up", float, 0.4, "warm up ratio for unsupervised loss"),
        ]
