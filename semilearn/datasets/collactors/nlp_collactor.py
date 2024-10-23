# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Optional, Union

from transformers import BertTokenizerFast
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
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        text_keys = ["x_lb", "x_lb_s", "x_ulb_w", "x_ulb_w_2", "x_ulb_s", "x_ulb_s2"]
        text_features = {k: [] for k in text_keys}
        other_features = []
        for f in features:
            exist_ks = [k for k in text_keys if k in f]
            for k in exist_ks:
                text = f.pop(k)
                input_ids = self.tokenizer(text, max_length=self.max_length, truncation=True, padding=False)["input_ids"]
                text_features[k].append({"input_ids": input_ids})
            other_features.append(f)

        batch = default_data_collator(other_features, return_tensors="pt")

        for key, feats in text_features.items():
            if len(feats) > 0:
                out = self.tokenizer.pad(
                    feats,
                    padding=True,
                    max_length=None,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                    return_tensors=self.return_tensors,
                )
                batch[key] = {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

        return batch


def get_bert_base_collactor(pretrain_path="bert-base-uncased", max_length=512):
    tokenizer = BertTokenizerFast.from_pretrained(pretrain_path)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True  # turn off
    collact_fn = DataCollatorWithPadding(tokenizer, max_length=max_length)
    return collact_fn
