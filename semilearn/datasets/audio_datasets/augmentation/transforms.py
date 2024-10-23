# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import random
import warnings

from audiomentations import *


class AudioTransforms:
    """
    Strong transformation function for audio data.

    Args:
        max_length_seconds (float): Maximum output length of the audio in seconds.
        dataset (str): Name of the dataset.
    """

    def __init__(self, max_length_seconds, dataset_name=""):
        self.max_length_seconds = max_length_seconds
        self.effects_list, self.n = self.get_effects_list(dataset_name)
        self.adjust_duration = AdjustDuration(duration_seconds=max_length_seconds, p=1.0)

    def get_effects_list(self, dataset_name):
        if dataset_name.lower() in ["bvcc", "vcc2018"]:
            effects_list = [TimeMask(p=1.0), Trim(p=1.0), Padding(p=1.0)]
            num_effects = 1
        else:
            effects_list = [Gain(p=1.0), PitchShift(p=1.0), TimeStretch(p=1.0), RoomSimulator(p=1.0)]
            num_effects = 2
        return effects_list, num_effects

    def __call__(self, audio, sample_rate):
        transform = Compose(random.choices(self.effects_list, k=self.n))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Possible clipped samples in output.")
            aug_wav = transform(samples=audio, sample_rate=sample_rate)
        aug_wav = self.adjust_duration(aug_wav, sample_rate=sample_rate)
        return aug_wav
