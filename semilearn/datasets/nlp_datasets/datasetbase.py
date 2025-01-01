# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import random

from torch.utils.data import Dataset


class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of text and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    Returns both original and augmented text. Augmented texts can be None.
    """

    def __init__(self, alg, data, targets=None, is_ulb=False, *args, **kwargs):
        """
        Args:
            alg (str): Algorithm.
            data (list): List of text data along with two augmented texts (e.g., [text, aug_text1 (or None), aug_text2 (or None)]).
            targets (list or None): Target labels corresponding to the images.
            is_ulb (bool): Indicates if the dataset is unlabeled.
        """
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets
        self.is_ulb = is_ulb
        self.transform = None

    def random_choose_sen(self):
        """Randomly choose one of the augmented sentences."""
        return random.randint(1, 2)

    def __sample__(self, idx):
        """Retrieve the text and corresponding target at a specific index."""
        sen = self.data[idx]
        target = None if self.targets is None else self.targets[idx]
        return sen, target

    def __getitem__(self, idx):
        """
        Returns raw and/or augmented text based on the algorithm and dataset type.
        """
        sen, target = self.__sample__(idx)

        data_dict = {
            "idx_lb": lambda: idx,
            "x_lb": lambda: sen[0],
            "x_lb_s": lambda: sen[self.random_choose_sen()],
            "y_lb": lambda: target,
            "idx_ulb": lambda: idx,
            "x_ulb_w": lambda: sen[0],
            "x_ulb_w_2": lambda: sen[0],
            "x_ulb_s": lambda: sen[self.random_choose_sen()],
            "x_ulb_s_2": lambda: sen[self.random_choose_sen()],
        }

        data_keys = self._determine_data_keys()
        return {k: data_dict[k]() for k in data_keys}

    def __len__(self):
        return len(self.data)

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
