# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.
# Code in this file is adapted from xmed-lab/UCVME
# https://github.com/xmed-lab/UCVME

import torch
import torch.nn as nn

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument


class UCVME_Net(nn.Module):
    def __init__(self, base_1, base_2, drop_rate=0.05):
        super(UCVME_Net, self).__init__()
        self.backbone_1 = base_1
        self.backbone_2 = base_2
        self.dropout = nn.Dropout(p=drop_rate, inplace=False)
        self.fc_v_1 = nn.Linear(base_1.num_features, 1)
        self.fc_v_2 = nn.Linear(base_2.num_features, 1)

    def enable_dropout(self):
        for m in self.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def single_forward(self, x, num_ensemble=5, backbone_id=1, only_fc=False):
        if self.training == False:  # using dropout in train & eval mode
            self.enable_dropout()
        backbone = self.backbone_1 if backbone_id == 1 else self.backbone_2
        fc = self.fc_v_1 if backbone_id == 1 else self.fc_v_2
        logits_m_list = []
        logits_v_list = []
        feat_list = []
        for _ in range(num_ensemble):
            feat = x if only_fc else backbone(x, only_feat=True)
            logits_m = backbone(self.dropout(feat), only_fc=True)
            logits_v = fc(self.dropout(feat)).flatten()
            logits_m_list.append(logits_m)
            logits_v_list.append(logits_v)
            feat_list.append(feat)
        mean_logits_m = torch.stack(logits_m_list, dim=0).mean(dim=0)
        mean_logits_v = torch.stack(logits_v_list, dim=0).mean(dim=0)
        mean_feat = torch.stack(feat_list, dim=0).mean(dim=0)
        return {"logits": mean_logits_m, "logits_v": mean_logits_v, "feat": mean_feat}

    def forward(self, x, num_ensemble=5, only_fc=False):
        outs_1 = self.single_forward(x, num_ensemble, 1, only_fc)
        outs_2 = self.single_forward(x, num_ensemble, 2, only_fc)
        return {
            "logits": (outs_1["logits"] + outs_2["logits"]) / 2,
            "logits_v": (outs_1["logits_v"] + outs_2["logits_v"]) / 2,
            "logits_1": outs_1["logits"],
            "logits_2": outs_2["logits"],
            "logits_v_1": outs_1["logits_v"],
            "logits_v_2": outs_2["logits_v"],
            "feat": (outs_1["feat"] + outs_2["feat"]) / 2,
            "feat_1": outs_1["feat"],
            "feat_2": outs_2["feat"],
        }

    def init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()

    # def group_matcher(self, coarse=False):
    #     matcher = self.backbone_1.group_matcher(coarse, prefix="backbone.")
    #     return matcher


@ALGORITHMS.register("ucvme")
class UCVME(AlgorithmBase):
    """
    UCVME algorithm (https://arxiv.org/abs/2302.07579).

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

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        self.init(dropout_rate=args.dropout_rate, num_ensemble=args.num_ensemble)
        super().__init__(args, net_builder, tb_log, logger, **kwargs)

    def init(self, dropout_rate=0.05, num_ensemble=5):
        self.dropout_rate = dropout_rate
        self.num_ensemble = num_ensemble

    def set_model(self):
        model_1 = super().set_model(drop_rate=self.dropout_rate)
        model_2 = super().set_model(drop_rate=self.dropout_rate)
        model = UCVME_Net(model_1, model_2, drop_rate=self.dropout_rate)
        return model

    def set_ema_model(self):
        """
        initialize ema model from model
        """
        ema_model_1 = super().set_model(drop_rate=self.dropout_rate)
        ema_model_2 = super().set_model(drop_rate=self.dropout_rate)
        ema_model = UCVME_Net(ema_model_1, ema_model_2, drop_rate=self.dropout_rate)
        ema_model.load_state_dict(self.check_prefix_state_dict(self.model.state_dict()))
        return ema_model

    def train_step(self, x_lb, y_lb, x_ulb_w, **kwargs):
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            outs_x_lb = self.model(x_lb, num_ensemble=1)
            avg_feats_lb = outs_x_lb["feat"]
            avg_logits_v_lb = outs_x_lb["logits_v"]
            logits_m_lb_1 = outs_x_lb["logits_1"]
            logits_v_lb_1 = outs_x_lb["logits_v_1"]
            logits_m_lb_2 = outs_x_lb["logits_2"]
            logits_v_lb_2 = outs_x_lb["logits_v_2"]

            outs_x_ulb_w = self.model(x_ulb_w, num_ensemble=1)
            avg_feats_ulb_w = outs_x_ulb_w["feat"]
            logits_m_ulb_w_1 = outs_x_ulb_w["logits_1"]
            logits_v_ulb_w_1 = outs_x_ulb_w["logits_v_1"]
            logits_m_ulb_w_2 = outs_x_ulb_w["logits_2"]
            logits_v_ulb_w_2 = outs_x_ulb_w["logits_v_2"]

            with torch.no_grad():
                self.bn_controller.freeze_bn(self.model)
                outs_x_ulb_w = self.model(x_ulb_w, num_ensemble=self.num_ensemble)
                avg_logits_m_ulb = outs_x_ulb_w["logits"]
                avg_logits_v_ulb = outs_x_ulb_w["logits_v"]
                self.bn_controller.unfreeze_bn(self.model)

            # extract features for further use in the classification algorithm.
            feat_dict = {"x_lb": avg_feats_lb, "x_ulb_w": avg_feats_ulb_w}
            for k in kwargs:
                feat_dict[k] = self.model(kwargs[k], only_feat=True)

            # supervised loss

            loss_mse_1 = (logits_m_lb_1 - y_lb) ** 2
            loss_mse_2 = (logits_m_lb_2 - y_lb) ** 2

            sup_reg_loss_1 = 0.5 * (torch.mul(torch.exp(-avg_logits_v_lb), loss_mse_1) + avg_logits_v_lb)
            sup_reg_loss_2 = 0.5 * (torch.mul(torch.exp(-avg_logits_v_lb), loss_mse_2) + avg_logits_v_lb)

            sup_reg_loss = sup_reg_loss_1.mean() + sup_reg_loss_2.mean()
            sup_unc_loss = ((logits_v_lb_2 - logits_v_lb_1) ** 2).mean()
            sup_loss = sup_reg_loss + sup_unc_loss

            # unsupervised loss

            loss_mse_cps_0 = (logits_m_ulb_w_1 - avg_logits_m_ulb) ** 2
            loss_mse_cps_1 = (logits_m_ulb_w_2 - avg_logits_m_ulb) ** 2

            loss_cmb_cps_0 = 0.5 * (torch.mul(torch.exp(-avg_logits_v_ulb), loss_mse_cps_0) + avg_logits_v_ulb)
            loss_cmb_cps_1 = 0.5 * (torch.mul(torch.exp(-avg_logits_v_ulb), loss_mse_cps_1) + avg_logits_v_ulb)

            unsup_reg_loss = loss_cmb_cps_0.mean() + loss_cmb_cps_1.mean()
            unsup_unc_loss_1 = ((logits_v_ulb_w_1 - avg_logits_v_ulb) ** 2).mean()
            unsup_unc_loss_2 = ((logits_v_ulb_w_2 - avg_logits_v_ulb) ** 2).mean()
            unsup_loss = unsup_reg_loss + unsup_unc_loss_1 + unsup_unc_loss_2

            total_loss = sup_loss + self.ulb_loss_ratio * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(total_loss=total_loss.item())
        log_dict["train_reg/sup_loss"] = sup_loss.item()
        log_dict["train_reg/unsup_loss"] = unsup_loss.item()
        log_dict["train_reg/total_loss"] = total_loss.item()
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument("--dropout_rate", float, 0.05),
            SSL_Argument("--num_ensemble", int, 5),
        ]
