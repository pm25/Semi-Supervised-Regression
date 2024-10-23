# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from transformers import AutoFeatureExtractor
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data import default_data_collator


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    sample_rate: Optional[int] = 16000
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        wav_keys = ["x_lb", "x_lb_s", "x_ulb_w", "x_ulb_w_2", "x_ulb_s", "x_ulb_s2"]
        wav_features = {k: [] for k in wav_keys}
        other_features = []
        for f in features:
            exist_ks = [k for k in wav_keys if k in f]
            for k in exist_ks:
                feat = f.pop(k)
                wav_features[k].append(feat)
            other_features.append(f)

        batch = default_data_collator(other_features, return_tensors="pt")

        for key, feats in wav_features.items():
            if len(feats) > 0:
                out = self.tokenizer(
                    feats,
                    padding=True if key == "x_lb" else "max_length",
                    max_length=int(self.max_length * self.sample_rate),
                    sampling_rate=self.sample_rate,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                    truncation=True,
                )
                if "input_values" in out:
                    input_values = out["input_values"]
                elif "input_features" in out:
                    input_values = out["input_features"]
                batch[key] = input_values

        return batch


def get_wave2vecv2_base_collactor(pretrain_path="facebook/wav2vec2-base-960h", max_length=4, sample_rate=16000):
    pretrain_path = "facebook/wav2vec2-base-960h" if pretrain_path == "" else pretrain_path
    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrain_path)
    collator = DataCollatorWithPadding(feature_extractor, max_length=max_length, sample_rate=sample_rate)
    return collator


def get_hubert_base_collactor(pretrain_path="facebook/hubert-base-ls960", max_length=4, sample_rate=16000):
    pretrain_path = "facebook/hubert-base-ls960" if pretrain_path == "" else pretrain_path
    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrain_path)
    collator = DataCollatorWithPadding(feature_extractor, max_length=max_length, sample_rate=sample_rate)
    return collator


def get_whisper_base_collactor(pretrain_path="openai/whisper-base", max_length=30, sample_rate=16000):
    pretrain_path = "openai/whisper-base" if pretrain_path == "" else pretrain_path
    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrain_path)
    collator = DataCollatorWithPadding(feature_extractor, max_length=max_length, sample_rate=sample_rate)
    return collator
