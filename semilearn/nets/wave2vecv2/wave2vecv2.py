# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import torch
import torch.nn as nn

from transformers import Wav2Vec2Model, Wav2Vec2Config

from semilearn.nets.utils import init_weights


class RegressionWav2Vec2(nn.Module):
    """
    A regression model based on the Wav2Vec2 architecture (https://arxiv.org/abs/2006.11477).

    Args:
        use_pretrained (bool): Whether to use a pretrained weights. If `pretrained_path` is set and
                                `use_pretrained` is `False`, the model will load architecture
                               without pretrained weights. Defaults to True.
        pretrained_path (str or None): The Hugging Face path to the pretrained model.
                                        If None, a model with default configuration will be created.
        drop_rate (float): The dropout rate applied before the regression layer. Defaults to 0.1.
        freeze_encoder (bool): If True, the encoder will be frozen during training,
                               and only the regressor head will be trained.
                               Do not freeze the encoder when using with RankUp or ARC.

    Attributes:
        model (Wav2Vec2Model): The underlying Wav2Vec2 model.
        config (Wav2Vec2Config): Configuration of the Wav2Vec2 model.
        dropout (nn.Dropout): Dropout layer for regularization.
        num_features (int): Number of features from the model's hidden layer.
        regressor (nn.Sequential): The regressor head consisting of linear layers and activation.
    """

    def __init__(self, use_pretrained=True, pretrained_path=None, drop_rate=0.1, freeze_encoder=True):
        super(RegressionWav2Vec2, self).__init__()
        self.model, self.config = self.load_model(use_pretrained, pretrained_path)
        if freeze_encoder:
            self.model.freeze_feature_encoder()
        self.dropout = torch.nn.Dropout(p=drop_rate, inplace=False)
        self.num_features = self.config.hidden_size
        self.regressor = nn.Sequential(
            *[nn.Linear(self.config.hidden_size, self.config.hidden_size), nn.GELU(), nn.Linear(self.config.hidden_size, 1)]
        )

        self.regressor.apply(init_weights)

    def load_model(self, use_pretrained=True, pretrained_path=None):
        if use_pretrained and pretrained_path:
            config = Wav2Vec2Config.from_pretrained(pretrained_path)
            model = Wav2Vec2Model.from_pretrained(pretrained_path)
            return model, config

        config = Wav2Vec2Config() if not pretrained_path else Wav2Vec2Config.from_pretrained(pretrained_path)
        model = Wav2Vec2Model(config)
        return model, config

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """
        if only_fc:
            logits = self.regressor(x).flatten()
            return logits

        pooled_output = self.extract(x)

        if only_feat:
            return pooled_output

        logits = self.regressor(pooled_output).flatten()
        result_dict = {"logits": logits, "feat": pooled_output}
        return result_dict

    def extract(self, x):
        out_dict = self.model(x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict["last_hidden_state"]
        embed = out_dict["hidden_states"][0]
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        return pooled_output

    def group_matcher(self, coarse=False, prefix=""):
        matcher = dict(
            stem=r"^{}model.feature_projection|^{}model.feature_extractor".format(prefix, prefix),
            blocks=r"^{}model.encoder.layers.(\d+)".format(prefix),
        )
        return matcher

    def no_weight_decay(self):
        return []


def wave2vecv2_base(pretrained=True, pretrained_path="facebook/wav2vec2-base-960h", **kwargs):
    model = RegressionWav2Vec2(use_pretrained=pretrained, pretrained_path=pretrained_path, **kwargs)
    return model


if __name__ == "__main__":
    model = wave2vecv2_base(True, "facebook/wav2vec2-base-960h")
    print(model)
