# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

from inspect import signature
from semilearn.core import ClsAlgorithmBase
from semilearn.core.utils import CLS_ALGORITHMS


@CLS_ALGORITHMS.register("fullysupervised")
class FullySupervised(ClsAlgorithmBase):
    """
    Train a fully supervised model using labeled data only. This serves as a baseline for comparison.

    Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)

    def process_batch(self, **kwargs):
        # get core algorithm parameters
        input_args = signature(super().train_step).parameters
        input_args = list(input_args.keys())
        return super().process_batch(input_args=input_args, **kwargs)

    def train_step(self, x_lb, y_lb, **kwargs):
        out_dict, log_dict = super().train_step(x_lb=x_lb, y_lb=y_lb, **kwargs)

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            feat = out_dict["feat"]["x_lb"]
            outs_x_lb = self.model(feat, use_arc=True, only_fc=True, targets=y_lb)
            logits_x_lb = outs_x_lb["logits"]
            y_lb = outs_x_lb["targets"]
            sup_loss = self.cls_loss(logits_x_lb, y_lb, reduction="mean")
            total_loss = out_dict["loss"] + self.cls_loss_ratio * sup_loss

        out_dict["loss"] = total_loss
        log_dict["train_cls/sup_loss"] = sup_loss.item()
        log_dict["train/total_loss"] = total_loss.item()
        return out_dict, log_dict


CLS_ALGORITHMS["supervised"] = FullySupervised
