# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.
# Code in this file is adapted from pytorch/pytorch
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Tuple

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg


class VCC2018(Dataset):
    """`The VCC2018 Data Set <https://datashare.ed.ac.uk/handle/10283/3061> <https://datashare.ed.ac.uk/handle/10283/3257>`

    The Voice Conversion Challenge 2018 (VCC2018) dataset is an audio quality assessment dataset,
    where the objective is to predict the quality of an audio sample. The labels, ranging from 1
    to 5, are obtained by averaging the scores provided by multiple listeners. The dataset
    comprises over 20,000 audio files, which we split into 16,464 training samples and 4,116 test samples.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL_MD5 = {
        "data": (
            "https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_submitted_systems_converted_speech.tar.gz",
            "75b0f937240f6850a56ec2cbad34b4ad",
        ),
        "meta": ("https://github.com/pm25/regression-datasets/raw/refs/heads/main/data/vcc2018/meta.zip", "66ea41b35ffbc1ad6565e538320f011d"),
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = False,
    ) -> None:
        super().__init__()
        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = Path(root) / "vcc2018"
        self._meta_folder = self._base_folder / "meta"
        self._audio_folder = self._base_folder / "mnt/sysope/test_files/testVCC2"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        metadata = pd.read_csv(self._meta_folder / f"{split}.csv")
        self._file_paths = metadata["file_name"].apply(lambda x: self._audio_folder / x).to_numpy(dtype="object")
        self._labels = metadata["label"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self._file_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        audio_file, label = self._file_paths[idx], self._labels[idx]
        waveform, sample_rate = librosa.load(audio_file, sr=None, mono=True)
        return waveform, sample_rate, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._audio_folder))

    def _download(self) -> None:
        if self._check_exists():
            return
        for url, md5 in self._URL_MD5.values():
            download_and_extract_archive(url, download_root=self._base_folder, md5=md5)
