# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.


from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS


@ALGORITHMS.register("fullysupervised")
class FullySupervised(AlgorithmBase):
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

    def train_step(self, x_lb, y_lb, **kwargs):
        # inference and calculate sup losses
        with self.amp_cm():
            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb["logits"]
            feats_x_lb = outs_x_lb["feat"]
            sup_loss = self.reg_loss(logits_x_lb, y_lb, reduction="mean")

            # extract features for further use in the classification algorithm.
            feat_dict = {"x_lb": feats_x_lb}
            for k in kwargs:
                feat_dict[k] = self.model(kwargs[k], only_feat=True)

        out_dict = self.process_out_dict(loss=sup_loss, feat=feat_dict)
        log_dict = self.process_log_dict(total_loss=sup_loss.item())
        log_dict["train_reg/sup_loss"] = sup_loss.item()
        return out_dict, log_dict

    def train(self):
        # lb: labeled, ulb: unlabeled
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

            # prevent the training iterations exceed args.num_train_iter
            if self.it > self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb in self.loader_dict["train_lb"]:
                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")
        self.call_hook("after_run")


ALGORITHMS["supervised"] = FullySupervised
