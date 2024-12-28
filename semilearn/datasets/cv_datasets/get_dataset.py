# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

from semilearn.datasets import cv_datasets
from semilearn.datasets.utils import split_ssl_data, load_image_files

from .datasetbase import BasicDataset, ImagePathDataset
from .augmentation import get_val_transforms, get_weak_transforms, get_strong_transforms


def get_cv_dataset(args, reg_alg, dataset_name, num_labels, data_dir="./data", include_lb_to_ulb=True):
    """
    Get the computer vision dataset and split the training samples into labeled and unlabeled sets.

    Args:
        reg_alg (str): Algorithm for regression output.
        dataset_name (str): The name of the dataset to load.
        num_labels (int): The number of labeled samples for the training set.
        data_dir (str): The directory from which to load the dataset.
        include_lb_to_ulb (bool): Indicates whether to include labeled data in the unlabeled set.

    Returns:
        Tuple[Dataset, Dataset, Dataset, Dataset]:
            A tuple containing:
                - train labeled dataset
                - train unlabeled dataset
                - evaluation dataset
                - test dataset
    """

    dataset = getattr(cv_datasets, dataset_name.upper())

    train_dataset = dataset(data_dir, split="train", download=True)
    train_paths, train_targets = train_dataset._file_paths, train_dataset._labels

    test_dataset = dataset(data_dir, split="test", download=True)
    test_paths, test_targets = test_dataset._file_paths, test_dataset._labels

    if args.preload:
        train_data = load_image_files(train_paths)
        test_data = load_image_files(test_paths)
        ImageDataset = BasicDataset
    else:
        train_data = train_paths
        test_data = test_paths
        ImageDataset = ImagePathDataset

    transform_weak = get_weak_transforms(crop_size=args.img_size, crop_ratio=args.crop_ratio, dataset_name=dataset_name)
    transform_strong = get_strong_transforms(crop_size=args.img_size, crop_ratio=args.crop_ratio, dataset_name=dataset_name)
    transform_val = get_val_transforms(crop_size=args.img_size, dataset_name=dataset_name)

    eval_dset = ImageDataset(reg_alg, test_data, test_targets, transform_val, False, None)
    test_dset = None

    if reg_alg == "fullysupervised":
        lb_dset = ImageDataset(reg_alg, train_data, train_targets, transform_weak, False, transform_strong)
        return lb_dset, None, eval_dset, test_dset

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(
        args,
        train_data,
        train_targets,
        lb_num_labels=num_labels,
        ulb_num_labels=args.ulb_num_labels,
        include_lb_to_ulb=include_lb_to_ulb,
    )

    lb_dset = ImageDataset(reg_alg, lb_data, lb_targets, transform_weak, False, transform_strong)
    ulb_dset = ImageDataset(reg_alg, ulb_data, ulb_targets, transform_weak, True, transform_strong)

    if reg_alg == "supervised":
        ulb_dset = None

    return lb_dset, ulb_dset, eval_dset, test_dset
