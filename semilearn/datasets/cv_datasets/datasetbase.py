# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import numpy as np
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation,
    and return both weakly and strongly augmented images.
    """

    def __init__(self, alg, data, targets=None, transform=None, is_ulb=False, strong_transform=None, *args, **kwargs):
        """
        Args:
            alg (str): Algorithm.
            data (list): List of PIL images or numpy arrays.
            targets (list or None): Target labels corresponding to the images.
            transform (callable or None): Basic transformation function applied to the image.
            is_ulb (bool): Indicates if the dataset is unlabeled.
            strong_transform (callable or None): Strong transformation function applied to the image.
        """
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets
        self.transform = transform
        self.strong_transform = strong_transform
        self.is_ulb = is_ulb

        self._check_transform()

    def __sample__(self, idx):
        """Retrieve the image and corresponding target at a specific index."""
        img = self.data[idx]
        target = None if self.targets is None else self.targets[idx]
        return img, target

    def __getitem__(self, idx):
        """
        Returns weakly and/or strongly augmented images based on the algorithm and dataset type.
        """
        img, target = self.__sample__(idx)

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        if self.transform is None:
            return {"x_lb": transforms.ToTensor()(img), "y_lb": target}

        data_dict = {
            "idx_lb": lambda: idx,
            "x_lb": lambda: self.transform(img),
            "x_lb_s": lambda: self.strong_transform(img),
            "y_lb": lambda: target,
            "idx_ulb": lambda: idx,
            "x_ulb_w": lambda: self.transform(img),
            "x_ulb_w_2": lambda: self.transform(img),
            "x_ulb_s": lambda: self.strong_transform(img),
            "x_ulb_s_2": lambda: self.strong_transform(img),
        }

        data_keys = self._determine_data_keys()
        return {k: data_dict[k]() for k in data_keys}

    def __len__(self):
        return len(self.data)

    def _check_transform(self):
        """Ensure strong augmentation is used if required by the algorithm."""
        if self.strong_transform is None and self.is_ulb:
            assert self.alg not in ["rankup"], f"alg {self.alg} requires strong augmentation"

    def _determine_data_keys(self):
        """Determine the required output data based on the algorithm type."""
        data_keys = set()

        if not self.is_ulb:
            data_keys.update({"idx_lb", "x_lb", "y_lb"})
            return data_keys

        # for regression algorithms
        if self.alg == "fullysupervised" or self.alg == "supervised":
            data_keys.update({"idx_ulb"})
        elif self.alg == "rankup" or self.alg == "rankuprda":
            data_keys.update({"idx_ulb", "x_ulb_w", "x_ulb_s"})
        elif self.alg == "pimodel" or self.alg == "meanteacher" or self.alg == "mixmatch":
            data_keys.update({"idx_ulb", "x_ulb_w", "x_ulb_w_2"})
        else:
            data_keys.update({"idx_ulb", "x_ulb_w"})

        return data_keys


class ImagePathDataset(BasicDataset):
    """Dataset subclass that directly opens images from file paths."""

    def __sample__(self, idx):
        path, target = super().__sample__(idx)
        img = Image.open(path).convert("RGB")
        return img, target
