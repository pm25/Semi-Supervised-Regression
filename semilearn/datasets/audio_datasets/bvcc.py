# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.
# Code in this file is adapted from pytorch/pytorch
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

import os
import shutil
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from hashlib import md5
from pathlib import Path
from typing import Any, Tuple

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg


class BVCC(Dataset):
    """`The BVCC Data Set <https://zenodo.org/records/6572573>`

    The VoiceMOS2022 (BVCC) dataset is an audio quality assessment dataset, where the objective
    is to predict the quality of an audio sample. The labels, ranging from 1 to 5, are obtained
    by averaging the scores provided by multiple listeners. The dataset is split into training
    (4,974 samples), evaluation (1,066 samples), and testing (1,066 samples) sets.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _BVCC_URL = ("https://zenodo.org/records/6572573/files/main.tar.gz", "fc880c2a208c3285a47bd9a64f34eb11")
    _BC_URL = [
        ("https://data.cstr.ed.ac.uk/blizzard/wavs_and_scores/blizzard_wavs_and_scores_2008_release_version_1.tar.bz2", "5360686aac07ffe22420ffd00c90ea74"),
        ("https://data.cstr.ed.ac.uk/blizzard/wavs_and_scores/blizzard_wavs_and_scores_2009_release_version_1.tar.bz2", "1ffdf2c0ddb5f2e0c97908a70d0302b2"),
        ("https://data.cstr.ed.ac.uk/blizzard/wavs_and_scores/blizzard_wavs_and_scores_2010_release_version_1.tar.bz2", "e60d504c4e3a95d7792e6fd056e74aec"),
        ("https://data.cstr.ed.ac.uk/blizzard/wavs_and_scores/blizzard_wavs_and_scores_2011_release_version_1.tar.bz2", "8e59a48f88568f86d1962644fdd568c5"),
        ("https://data.cstr.ed.ac.uk/blizzard/wavs_and_scores/blizzard_wavs_and_scores_2013_release_version_2.tar.bz2", "837e20399409393332322fdd59d114de"),
        ("https://data.cstr.ed.ac.uk/blizzard/wavs_and_scores/blizzard_wavs_and_scores_2016_release_version_1.tar.bz2", "fff9d42a97161835f2545e02e5392e06"),
    ]
    _LABEL_URL = ("https://github.com/pm25/regression-datasets/raw/refs/heads/main/data/bvcc/meta.zip", "a2054d0e14b36e5e8692600007244528")

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = False,
    ) -> None:
        super().__init__()
        self._split = verify_str_arg(split, "split", ("train", "dev", "test"))
        self._base_folder = Path(root) / "bvcc"
        self._meta_folder = self._base_folder / "meta"
        self._audio_folder = self._base_folder / "audios"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        metadata = pd.read_csv(self._meta_folder / f"{split}.csv")
        self._file_paths = metadata["file_name"].apply(lambda x: self._audio_folder / x).to_numpy()
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
        download_and_extract_archive(self._BVCC_URL[0], download_root=self._base_folder, md5=self._BVCC_URL[1])
        download_and_extract_archive(self._LABEL_URL[0], download_root=self._base_folder, md5=self._LABEL_URL[1])
        for url, md5 in self._BC_URL:
            download_and_extract_archive(url, download_root=self._base_folder, md5=md5)
        self._move_to_folder()
        self._gather()

    def _move_to_folder(self) -> None:
        out_folder = self._base_folder / "blizzard"
        out_folder.mkdir(exist_ok=True, parents=True)
        for folder in self._base_folder.glob("blizzard_wavs_and_scores*"):
            if not folder.is_dir():
                continue
            folder.rename(out_folder / folder.name)
        for path in out_folder.glob("**/*"):
            os.chmod(path, 0o755)

    def _gather(self) -> None:
        self._audio_folder.mkdir(exist_ok=True, parents=True)
        gather_file_path = self._base_folder / "main" / "gather.scp"
        lines = [x.strip() for x in open(gather_file_path, "r").readlines()]
        for year in tqdm(["2008", "2009", "2010", "2011", "2013", "2016"], desc="Gathering files"):
            keep_files = [l for l in lines if l.split("-")[0] == "BC" + year]
            ver = 1 if year != "2013" else 2
            base_dir = self._base_folder / f"blizzard/blizzard_wavs_and_scores_{year}_release_version_{ver}"

            for f in keep_files:
                t = f.split("-")[1]
                g = f.split("-")[2].split("_")[0]
                uid = f.split("-")[-1]
                if year == "2008":
                    wav_dir = base_dir / f"{t}/submission_directory/english/full/{year}/{g}"
                elif year in ["2009", "2010"]:
                    wav_dir = base_dir / f"{t}/submission_directory/english/EH1/{year}/{g}/wavs"
                elif year in "2011":
                    wav_dir = base_dir / f"{t}/submission_directory/{year}/{g}/wav"
                elif year in "2016":
                    wav_dir = base_dir / f"{t}/submission_directory/{year}/audiobook/wav"
                elif year == "2013":
                    tk = "EH2-English" if t == "B" else "EH1-English"
                    gd = "audiobook_sentences" if g == "booksent" else g
                    wav_dir = base_dir / f"{t}/submission_directory/{year}/{tk}/{gd}/wav"

                sid = f"BC{year}-{t}"
                sh = md5(sid[::-1].encode()).hexdigest()[0:5][::-1]
                fwn = f"BC{year}-{t}-{uid}".split(".")[0]
                uh = md5(fwn[::-1].encode()).hexdigest()[4:11][::-1]
                new_file_name = f"sys{sh}-utt{uh}.wav"
                shutil.copy(wav_dir / uid, self._audio_folder / new_file_name)
        # copy vcc data to out_folder
        for src_wav in (self._base_folder / "main/DATA/wav").glob("*.wav"):
            shutil.copy(src_wav, self._audio_folder / src_wav.name)
        shutil.copy(self._base_folder / "main" / "silence.wav", self._audio_folder / "sys4bafa-uttc2e86f6.wav")
        shutil.rmtree(self._base_folder / "blizzard")
