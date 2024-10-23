# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import os
import librosa
import numpy as np
from PIL import Image
from tqdm import tqdm
from io import BytesIO


# Base directory for data
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def split_ssl_data(
    args,
    data,
    targets,
    lb_num_labels,
    ulb_num_labels=None,
    lb_index=None,
    ulb_index=None,
    include_lb_to_ulb=True,
):
    """
    Splits data and targets into labeled and unlabeled sets.

    Args:
        data (list or np.array): The data to be split into labeled and unlabeled sets.
        targets (list or np.array): The targets corresponding to the data.
        lb_num_labels (int): The total number of labeled samples.
        ulb_num_labels (int or None): Similar to lb_num_labels but for unlabeled data.
                        default to None, denoting use all remaining data except for labeled data as unlabeled set
        lb_index (np.array or None): If np.array of index is given, select the data[index], target[index] as labeled samples.
        ulb_index (np.array or None): If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb (bool): If True, labeled data is also included in unlabeled data

    Returns:
        Labeled data, labeled targets, unlabeled data, unlabeled targets.
    """
    data, targets = np.array(data), np.array(targets)
    lb_idx, ulb_idx = sample_labeled_unlabeled_data(args, targets, lb_num_labels, ulb_num_labels, load_exist=True)

    # manually set lb_idx and ulb_idx, do not use except for debug
    if lb_index is not None:
        lb_idx = lb_index
    if ulb_index is not None:
        ulb_idx = ulb_index

    if include_lb_to_ulb:
        ulb_idx = np.concatenate([lb_idx, ulb_idx], axis=0)

    return data[lb_idx], targets[lb_idx], data[ulb_idx], targets[ulb_idx]


def sample_labeled_unlabeled_data(args, target, lb_num_labels, ulb_num_labels=None, load_exist=True):
    """
    Samples indices for labeled and unlabeled data.

    Args:
        targets (list or np.array): The targets corresponding to the data.
        lb_num_labels (int): The total number of labeled samples.
        ulb_num_labels (int): Optional, number of unlabeled samples.
        load_exist (bool): If True, loads existing sampled indices.

    Returns:
        Labeled indices and unlabeled indices.
    """
    dump_dir = os.path.join(base_dir, "data", args.dataset, "labeled_idx")
    os.makedirs(dump_dir, exist_ok=True)
    lb_dump_path = os.path.join(dump_dir, f"lb_labels{args.num_labels}_seed{args.seed}_idx.npy")
    ulb_dump_path = os.path.join(dump_dir, f"ulb_labels{args.num_labels}_seed{args.seed}_idx.npy")

    if os.path.exists(lb_dump_path) and os.path.exists(ulb_dump_path) and load_exist:
        lb_idx = np.load(lb_dump_path)
        ulb_idx = np.load(ulb_dump_path)
        return lb_idx, ulb_idx

    idx = np.arange(len(target))
    np.random.shuffle(idx)

    lb_idx = idx[:lb_num_labels]
    if ulb_num_labels is None:
        ulb_idx = idx[lb_num_labels:]
    else:
        ulb_idx = idx[lb_num_labels : lb_num_labels + ulb_num_labels]

    if isinstance(lb_idx, list):
        lb_idx = np.asarray(lb_idx)
    if isinstance(ulb_idx, list):
        ulb_idx = np.asarray(ulb_idx)

    np.save(lb_dump_path, lb_idx)
    np.save(ulb_dump_path, ulb_idx)

    return lb_idx, ulb_idx


def get_collactor(args, net):
    """
    Returns the appropriate collactor function based on the specified network.
    """

    if net == "bert_base":
        from semilearn.datasets.collactors import get_bert_base_collactor

        collact_fn = get_bert_base_collactor(args.pretrain_path, args.max_length)
    elif net == "wave2vecv2_base":
        from semilearn.datasets.collactors import get_wave2vecv2_base_collactor

        collact_fn = get_wave2vecv2_base_collactor(args.pretrain_path, args.max_length_seconds, args.sample_rate)
    elif net == "hubert_base":
        from semilearn.datasets.collactors import get_hubert_base_collactor

        collact_fn = get_hubert_base_collactor(args.pretrain_path, args.max_length_seconds, args.sample_rate)
    elif net == "whisper_base":
        from semilearn.datasets.collactors import get_whisper_base_collactor

        collact_fn = get_whisper_base_collactor(args.pretrain_path, args.max_length_seconds, args.sample_rate)
    else:
        collact_fn = None

    return collact_fn


def bytes_to_array(b: bytes) -> np.ndarray:
    """
    Converts bytes data to a Numpy array.
    """
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


def load_audio_files(paths, sample_rate=16000):
    """
    Loads audio files from the given paths.

    Returns:
        Array of loaded audio.
    """
    waveforms = []
    for path in tqdm(paths, desc="Loading audio data"):
        try:
            waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
            waveforms.append(waveform)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return np.array(waveforms, dtype="object")


def load_image_files(paths):
    """
    Loads image files from the given paths.

    Returns:
        Array of loaded images.
    """
    images = np.empty((len(paths),), dtype="object")
    for idx, path in tqdm(enumerate(paths), desc="Loading image data", total=len(paths)):
        try:
            img = Image.open(path).convert("RGB")
            images[idx] = img
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return images
