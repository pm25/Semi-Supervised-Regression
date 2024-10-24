# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.
# Code in this file is adapted from pytorch/pytorch
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

import json
import numpy as np
from pathlib import Path
from typing import Any, Tuple

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg


class YELP_REVIEW(Dataset):
    """`Yelp Review Dataset <http://www.yelp.com/dataset_challenge> <https://arxiv.org/abs/2208.07204>`

    The Yelp Review dataset is a sentiment ordinal regression dataset, where the goal
    is to predict the rating of a customer based on their comment. The labels are divided
    into 5 classes (scores ranging from 0 to 4). Originally, each class contains 130,000
    training samples and 10,000 test samples.

    This version uses a processed Yelp Review dataset provided by USB
    (https://github.com/microsoft/semi-supervised-learning). It contains:
        - 50,000 samples per class for the training split (250,000 samples total)
        - 5,000 samples per class for the validation split (25,000 samples total)
        - The original test dataset remains unchanged (50,000 samples total)

    Additionally, the dataset includes preprocessed augmented text data (aug_0 and aug_1)
    generated using back-translation, along with the original text (ori).

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "https://huggingface.co/datasets/py97/Yelp-Review/resolve/main/YelpReview.tar.gz"
    _MD5 = "4c3e3736f3dc2c175f5ff9b0f69e6043"

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = False,
    ) -> None:
        super().__init__()
        self._split = verify_str_arg(split, "split", ("train", "dev", "test"))
        self._base_folder = Path(root) / "yelp_review"
        self._text_folder = self._base_folder / "YelpReview"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        with open(self._text_folder / f"{split}.json", "r") as f:
            data = json.load(f)

        texts, labels = [], []
        for idx in data:
            aug_0 = data[idx].get("aug_0", None)
            aug_1 = data[idx].get("aug_1", None)
            texts.append((data[idx]["ori"], aug_0, aug_1))
            labels.append(float(data[idx]["label"]))

        self._texts = np.array(texts, dtype="object")
        self._labels = np.array(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        text, label = self._texts[idx], self._labels[idx]
        return text, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return self._text_folder.exists() and self._text_folder.is_dir()

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self._base_folder, md5=self._MD5)
