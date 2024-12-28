# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import numpy as np

from .utils import RDAHook

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument


@ALGORITHMS.register("rda")
class RDA(AlgorithmBase):
    """
    RDA algorithm (https://arxiv.org/abs/2410.22124).

    Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
        - reg_unsup_warm_up (`float`, *optional*, defaults to 0.4):
            Ramp up for weights for unsupervised loss
        - rda_num_refine_iter (`int`):
            Number of iterations to apply RDA.
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        self.reg_init(
            reg_unsup_warm_up=args.reg_unsup_warm_up,
            rda_num_refine_iter=args.rda_num_refine_iter,
        )
        super().__init__(args, net_builder, tb_log, logger)

    def reg_init(self, reg_unsup_warm_up, rda_num_refine_iter):
        self.reg_unsup_warm_up = reg_unsup_warm_up
        self.rda_num_refine_iter = rda_num_refine_iter

    def set_hooks(self):
        super().set_hooks()
        # reset PseudoLabelingHook hook
        self.register_hook(
            RDAHook(
                train_ulb_len=len(self.dataset_dict["train_ulb"]),
                lb_targets=np.copy(self.dataset_dict["train_lb"].targets),
                num_refine_iter=self.rda_num_refine_iter,
            ),
            "RDAHook",
        )

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, **kwargs):
        self.idx_ulb = idx_ulb

        # inference and calculate sup losses
        with self.amp_cm():
            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb["logits"]
            feats_x_lb = outs_x_lb["feat"]
            sup_loss = self.reg_loss(logits_x_lb, y_lb, reduction="mean")

            self.bn_controller.freeze_bn(self.model)
            outs_x_ulb_w = self.model(x_ulb_w)
            logits_x_ulb_w = outs_x_ulb_w["logits"]
            feats_x_ulb_w = outs_x_ulb_w["feat"]
            self.bn_controller.unfreeze_bn(self.model)

            # extract features for further use in the classification algorithm.
            feat_dict = {"x_lb": feats_x_lb, "x_ulb_w": feats_x_ulb_w}
            for k in kwargs:
                feat_dict[k] = self.model(kwargs[k], only_feat=True)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook(
                "gen_ulb_targets",
                "RDAHook",
                logits=logits_x_ulb_w,
            )

            unsup_loss = self.reg_consistency_loss(logits_x_ulb_w, pseudo_label.detach(), "mse")

            unsup_warmup = np.clip(self.it / (self.reg_unsup_warm_up * self.num_train_iter), a_min=0.0, a_max=1.0)
            total_loss = sup_loss + self.reg_ulb_loss_ratio * unsup_loss * unsup_warmup

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(total_loss=total_loss.item())
        log_dict["train_reg/sup_loss"] = sup_loss.item()
        log_dict["train_reg/unsup_loss"] = unsup_loss.item()
        log_dict["train_reg/total_loss"] = total_loss.item()
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument("--reg_unsup_warm_up", float, 0.4),
            SSL_Argument("--rda_num_refine_iter", int, 1024),
        ]
