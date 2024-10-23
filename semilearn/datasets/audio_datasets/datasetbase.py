# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import librosa

from torch.utils.data import Dataset


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of audio and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation,
    and return both weakly and strongly augmented images.
    """

    def __init__(self, reg_alg, cls_alg, data, targets=None, transform=None, is_ulb=False, strong_transform=None, sample_rate=16000, *args, **kwargs):
        """
        Args:
            reg_alg (str): Algorithm for regression output.
            cls_alg (str): Algorithm for classification (ARC) output.
            data (list): List of audio data.
            targets (list or None): Target labels corresponding to the images.
            is_ulb (bool): Indicates if the dataset is unlabeled.
            transform (callable or None): Basic transformation function applied to the image.
            is_ulb (bool): Indicates if the dataset is unlabeled.
            strong_transform (callable or None): Strong transformation function applied to the image.
        """
        super(BasicDataset, self).__init__()
        self.reg_alg = reg_alg
        self.cls_alg = cls_alg
        self.data = data
        self.targets = targets
        self.transform = transform
        self.strong_transform = strong_transform
        self.is_ulb = is_ulb
        self.sample_rate = sample_rate

        self._check_transform()

    def __sample__(self, idx):
        """Retrieve the audio and corresponding target at a specific index."""
        audio = self.data[idx]
        target = None if self.targets is None else self.targets[idx]
        return audio, target

    def __getitem__(self, idx):
        """
        Returns weakly and/or strongly augmented images based on the algorithm and dataset type.
        """
        wav, target = self.__sample__(idx)

        if self.transform is None:
            return {"x_lb": wav, "y_lb": target}

        data_dict = {
            "idx_lb": lambda: idx,
            "x_lb": lambda: self.transform(wav, sample_rate=self.sample_rate),
            "x_lb_s": lambda: self.strong_transform(wav, sample_rate=self.sample_rate),
            "y_lb": lambda: target,
            "idx_ulb": lambda: idx,
            "x_ulb_w": lambda: self.transform(wav, sample_rate=self.sample_rate),
            "x_ulb_w_2": lambda: self.transform(wav, sample_rate=self.sample_rate),
            "x_ulb_s": lambda: self.strong_transform(wav, sample_rate=self.sample_rate),
            "x_ulb_s_2": lambda: self.strong_transform(wav, sample_rate=self.sample_rate),
        }

        data_keys = self._determine_data_keys()
        return {k: data_dict[k]() for k in data_keys}

    def __len__(self):
        return len(self.data)

    def _check_transform(self):
        """Ensure strong augmentation is used if required by the algorithm."""
        if self.strong_transform is None and self.is_ulb:
            assert self.cls_alg not in ["fixmatch"], f"cls_alg {self.cls_alg} requires strong augmentation"

    def _determine_data_keys(self):
        """Determine the required output data based on the algorithm type."""
        data_keys = set()

        if not self.is_ulb:
            data_keys.update({"idx_lb", "x_lb", "y_lb"})
            return data_keys

        # for regression algorithms
        if self.reg_alg == "fullysupervised" or self.reg_alg == "supervised":
            data_keys.update({"idx_ulb"})
        elif self.reg_alg == "pimodel" or self.reg_alg == "meanteacher" or self.reg_alg == "mixmatch":
            data_keys.update({"idx_ulb", "x_ulb_w", "x_ulb_w_2"})
        else:
            data_keys.update({"idx_ulb", "x_ulb_w"})

        # for classification algorithms
        if self.cls_alg == "fullysupervised" or self.cls_alg == "supervised":
            data_keys.update({"idx_ulb"})
        elif self.cls_alg == "pseudolabel":
            data_keys.update({"idx_ulb", "x_ulb_w"})
        elif self.cls_alg == "pimodel" or self.cls_alg == "meanteacher" or self.cls_alg == "mixmatch":
            data_keys.update({"idx_ulb", "x_ulb_w", "x_ulb_w_2"})
        else:
            data_keys.update({"idx_ulb", "x_ulb_w", "x_ulb_s"})

        return data_keys


class AudioPathDataset(BasicDataset):
    """Dataset subclass that directly opens audio from file paths."""

    def __sample__(self, idx):
        path, target = super().__sample__(idx)
        waveform, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        return waveform, target
