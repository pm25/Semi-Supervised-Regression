# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import random
from audiomentations import AdjustDuration


class Subsample:
    """Sample the first `max_length` seconds from the input audio.

    Args:
        max_length_seconds (float): Maximum output length of the audio in seconds.
                                     Output will be padded or truncated to this length.
    """

    def __init__(self, max_length_seconds):
        self.max_length_seconds = max_length_seconds
        self.adjust_duration = AdjustDuration(duration_seconds=max_length_seconds, p=1.0)

    def __call__(self, audio, sample_rate):
        max_sample_length = int(round(sample_rate * self.max_length_seconds))
        if len(audio) > max_sample_length:
            audio = audio[:max_sample_length]
        audio = self.adjust_duration(audio, sample_rate)  # padding to the `max_length_seconds`
        return audio


class RandomSubsample:
    """Randomly samples a chunk of audio of length between [`min_length`, `max_length`] seconds and pads it to `max_length` seconds.

    Args:
        max_length_seconds (float): Maximum output length of the audio in seconds.
                                     Output will be padded or truncated to this length.
        min_ratio (float): Minimum ratio of the maximum length for subsampling,
                           should be between 0.0 and 1.0.
    """

    def __init__(self, max_length_seconds, min_ratio=1.0):
        if not (0.0 <= min_ratio <= 1.0):
            raise ValueError("min_ratio should be between 0 and 1")

        self.max_length_seconds = max_length_seconds
        self.min_ratio = min_ratio
        self.adjust_duration = AdjustDuration(duration_seconds=max_length_seconds, p=1.0)
        self.min_length_seconds = max_length_seconds * self.min_ratio

    def __call__(self, audio, sample_rate):
        subsample_seconds = random.uniform(self.min_length_seconds, self.max_length_seconds)
        subsample_length = int(round(sample_rate * subsample_seconds))
        if len(audio) > subsample_length:
            max_offset = len(audio) - subsample_length
            random_offset = random.randint(0, max_offset)
            audio = audio[random_offset : random_offset + subsample_length]
        audio = self.adjust_duration(audio, sample_rate)  # padding to the `max_length_seconds`
        return audio
