# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import numpy as np
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument


@ALGORITHMS.register("pimodel")
class PiModel(AlgorithmBase):
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
    - unsup_warm_up (`float`, *optional*, defaults to 0.4):
        Ramp up for weights for unsupervised loss
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        self.init(unsup_warm_up=args.unsup_warm_up)

    def init(self, unsup_warm_up=0.4):
        self.unsup_warm_up = unsup_warm_up

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_w_2, **kwargs):
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb["logits"]
            feats_x_lb = outs_x_lb["feat"]

            # calculate BN only for the first batch
            self.bn_controller.freeze_bn(self.model)
            outs_x_ulb_w = self.model(x_ulb_w)
            logits_x_ulb_w = outs_x_ulb_w["logits"]
            feats_x_ulb_w = outs_x_ulb_w["feat"]
            outs_x_ulb_w_2 = self.model(x_ulb_w_2)
            logits_x_ulb_w_2 = outs_x_ulb_w_2["logits"]
            feats_x_ulb_w_2 = outs_x_ulb_w_2["feat"]
            self.bn_controller.unfreeze_bn(self.model)

            # extract features for further use in the classification algorithm.
            feat_dict = {"x_lb": feats_x_lb, "x_ulb_w": feats_x_ulb_w, "x_ulb_w_2": feats_x_ulb_w_2}
            for k in kwargs:
                feat_dict[k] = self.model(kwargs[k], only_feat=True)

            sup_loss = self.reg_loss(logits_x_lb, y_lb, reduction="mean")
            unsup_loss = self.consistency_loss(logits_x_ulb_w_2, logits_x_ulb_w.detach(), "mse")

            unsup_warmup = np.clip(self.it / (self.unsup_warm_up * self.num_train_iter), a_min=0.0, a_max=1.0)
            total_loss = sup_loss + self.ulb_loss_ratio * unsup_loss * unsup_warmup

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), unsup_loss=unsup_loss.item(), total_loss=total_loss.item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument("--unsup_warm_up", float, 0.4, "warm up ratio for regression unsupervised loss"),
        ]
