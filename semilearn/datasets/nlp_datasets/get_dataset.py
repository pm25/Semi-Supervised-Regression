# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

from semilearn.datasets import nlp_datasets
from semilearn.datasets.utils import split_ssl_data

from .datasetbase import BasicDataset


def get_nlp_dataset(args, reg_alg, dataset_name, num_labels=40, data_dir="./data", include_lb_to_ulb=True):
    """
    Get the NLP dataset and split the training samples into labeled and unlabeled sets.

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
    dataset = getattr(nlp_datasets, dataset_name.upper())

    train_dataset = dataset(data_dir, split="train", download=True)
    train_texts, train_targets = train_dataset._texts, train_dataset._labels

    test_dataset = dataset(data_dir, split="test", download=True)
    test_texts, test_targets = test_dataset._texts, test_dataset._labels

    eval_dset = BasicDataset(reg_alg, test_texts, test_targets, False)
    test_dset = None

    if dataset_name.lower() in ["yelp_review", "amazon_review"]:
        dev_dataset = dataset(data_dir, split="dev", download=True)
        dev_texts, dev_targets = dev_dataset._texts, dev_dataset._labels
        eval_dset = BasicDataset(reg_alg, dev_texts, dev_targets, False)
        test_dset = BasicDataset(reg_alg, test_texts, test_targets, False)

    if reg_alg == "fullysupervised":
        lb_dset = BasicDataset(reg_alg, train_texts, train_targets, False)
        return lb_dset, None, eval_dset, test_dset

    lb_texts, lb_targets, ulb_texts, ulb_targets = split_ssl_data(
        args,
        train_texts,
        train_targets,
        lb_num_labels=num_labels,
        ulb_num_labels=args.ulb_num_labels,
        include_lb_to_ulb=include_lb_to_ulb,
    )

    lb_dset = BasicDataset(reg_alg, lb_texts, lb_targets, False)
    ulb_dset = BasicDataset(reg_alg, ulb_texts, ulb_targets, True)

    if reg_alg == "supervised":
        ulb_dset = None

    return lb_dset, ulb_dset, eval_dset, test_dset
