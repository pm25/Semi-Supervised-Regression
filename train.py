# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import os
import random
import logging
import warnings
import argparse
import numpy as np

import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from semilearn.algorithms import get_algorithm, name2alg
from semilearn.algorithms.utils import str2bool
from semilearn.core.utils import (
    TBLog,
    count_parameters,
    get_logger,
    get_net_builder,
    get_port,
    over_write_args_from_file,
    send_model_cuda,
)


# custom function to prettify warning messages
def custom_warning_format(message, category, filename, lineno, line=None):
    return f"\033[93m[{category.__name__}] \033[0m({filename}:{lineno}): {message}\n"


warnings.formatwarning = custom_warning_format


def get_config():
    parser = argparse.ArgumentParser(description="Semi-Supervised Regression (SSR)")

    """
    Saving & Loading of the Model.
    """
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("-sn", "--save_name", type=str, default="rankup")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load_path", type=str)
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true", help="Use tensorboard to plot and save curves")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb to plot and save curves")
    parser.add_argument("--use_aim", action="store_true", help="Use aim to plot and save curves")

    """
    Training Configuration
    """
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--num_train_iter", type=int, default=20, help="Total number of training iterations")
    parser.add_argument("--num_warmup_iter", type=int, default=0, help="Cosine linear warmup iterations")
    parser.add_argument("--num_eval_iter", type=int, default=10, help="Evaluation frequency")
    parser.add_argument("--num_log_iter", type=int, default=5, help="Logging frequency")
    parser.add_argument("-nl", "--num_labels", type=int, default=250)
    parser.add_argument("-unl", "--ulb_num_labels", type=int, default=None, help="Number of labels for unlabeled data")
    parser.add_argument("-bsz", "--batch_size", type=int, default=8)
    parser.add_argument("--uratio", type=float, default=1.0, help="Ratio of unlabeled data to labeled data in each mini-batch")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--ema_m", type=float, default=0.999, help="EMA momentum for eval_model")
    parser.add_argument("--reg_criterion", type=str, default="l1", choices=["l1", "mse"])
    parser.add_argument("--reg_ulb_loss_ratio", type=float, default=1.0)

    """
    Optimizer Configurations
    """
    parser.add_argument("--optim", type=str, default="SGD")
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--layer_decay", type=float, default=1.0, help="Layer-wise learning rate decay")

    """
    Backbone Net Configurations
    """
    parser.add_argument("--net", type=str, default="wrn_28_2")
    parser.add_argument("--net_from_name", type=str2bool, default=False)
    parser.add_argument("--use_pretrain", type=str2bool, default=False)
    parser.add_argument("--pretrain_path", type=str, default=None)

    """
    Algorithms Configurations
    """
    ## core algorithm setting
    parser.add_argument("-alg", "--algorithm", type=str, default="supervised", help="Semi-supervised regression algorithm")
    parser.add_argument("--use_cat", type=str2bool, default=True, help="Use cat operation in algorithms")
    parser.add_argument("--amp", type=str2bool, default=False, help="Use mixed precision training")
    parser.add_argument("--clip_grad", type=float, default=0)

    """
    Data Configurations
    """
    ## standard setting configurations
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("-ds", "--dataset", type=str, default="utkface")
    parser.add_argument("--preload", type=str2bool, default=False)
    parser.add_argument("--train_sampler", type=str, default="RandomSampler")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--include_lb_to_ulb", type=str2bool, default="True", help="Include labeled data into unlabeled data")
    ## cv dataset arguments
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--crop_ratio", type=float, default=0.875)
    ## nlp dataset arguments
    parser.add_argument("--max_length", type=int, default=512)
    ## speech dataset algorithms
    parser.add_argument("--max_length_seconds", type=float, default=4.0)
    parser.add_argument("--sample_rate", type=int, default=16000)

    """
    Multi-GPUs & Distributed Training
    """
    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)  # noqa: E501
    parser.add_argument("--world-size", type=int, default=1, help="Number of nodes for distributed training")
    parser.add_argument("--rank", type=int, default=0, help="Node rank for distributed training")
    parser.add_argument("-du", "--dist-url", type=str, default="tcp://127.0.0.1:11111", help="URL for distributed training")
    parser.add_argument("--dist-backend", type=str, default="nccl", help="Distributed backend")
    parser.add_argument("--seed", type=int, default=1, help="Seed for initializing training. ")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use.")
    parser.add_argument(
        "--multiprocessing-distributed",
        type=str2bool,
        default=False,
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )

    # config file
    parser.add_argument("--c", type=str, default="")

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    # add regression algorithm specific parameters
    for argument in name2alg[args.algorithm].get_argument():
        parser.add_argument(
            argument.name,
            type=argument.type,
            default=argument.default,
            help=argument.help,
        )

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    return args


def main(args):
    """
    For (Distributed)DataParallelism,
    main(args) spawns each process (main_worker) to each GPU.
    """

    assert (
        args.num_train_iter % args.epoch == 0
    ), f"# total training iterations {args.num_train_iter} must be divisible by number of epochs {args.epoch}"

    save_path = os.path.join(args.save_dir, args.save_name)

    # handle model saving and overwriting logic
    if os.path.exists(save_path):
        if args.overwrite and not args.resume:
            import shutil

            shutil.rmtree(save_path)
        elif not args.overwrite:
            raise FileExistsError(f"Model already exists at {save_path}. Use --overwrite to replace it.")

    # handle resuming logic
    if args.resume:
        if not args.load_path:
            raise ValueError("Resuming training requires --load_path.")
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise ValueError("Saving and loading paths are the same. Use --overwrite to continue.")

    # seed warning
    if args.seed is not None:
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, but may slow down performance. "
            "Unexpected behavior could occur when restarting from checkpoints."
        )

    # GPU configuration
    if args.gpu == "None":
        args.gpu = None

    if args.gpu is not None:
        warnings.warn("You have selected a specific GPU. Data parallelism will be disabled.")

    # determine distributed setup
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node

    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size

        # args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    """
    main_worker is conducted on each GPU for distributed training.
    """

    # assign the current GPU
    args.gpu = gpu

    # random seed has to be set for the synchronization of labeled data sampling in each process.
    assert args.seed is not None, "Seed must be provided for synchronization."
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # setup for distributed training
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu  # compute global rank

        # initialize the process group for distributed training
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # set save path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None

    # set up tensorboard logging only for the primary GPU (rank 0)
    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, "tensorboard", use_tensorboard=args.use_tensorboard)
        logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.info(f"Using GPU: {args.gpu} for training")

    # build the network and set up the algorithm (RankUp, etc.)
    _net_builder = get_net_builder(args.net, args.net_from_name)
    model = get_algorithm(args, _net_builder, tb_log, logger)

    logger.info(f"Number of Trainable Parameters: {count_parameters(model.model)}")

    # send the model and EMA model (if applicable) to the GPU
    model.model = send_model_cuda(args, model.model)
    model.ema_model = send_model_cuda(args, model.ema_model, clip_batch=False)
    logger.info(f"Training Arguments: {model.args}")

    # if args.resume, load checkpoints from args.load_path
    if args.resume and os.path.exists(args.load_path):
        try:
            model.load_model(args.load_path)
            logger.info(f"Resumed training from {args.load_path}")
        except Exception as e:
            logger.error(f"Failed to resume load path {args.load_path}. Error: {e}")
            args.resume = False
    else:
        logger.info(f"Checkpoint path {args.load_path} does not exist. Starting fresh.")

    # warmup stage if the model supports it
    if hasattr(model, "warmup"):
        logger.info("Starting warmup stage")
        model.warmup()

    # start the main training loop
    logger.info("Starting model training")
    model.train()

    # print validation (and test results)
    for key, item in model.results_dict.items():
        logger.info(f"Model result - {key} : {item}")

    # finetuning stage if the model has it
    if hasattr(model, "finetune"):
        logger.info("Starting finetuning stage")
        model.finetune()

    # training completion notification
    logging.warning(f"Training on GPU {args.rank} is finished")


if __name__ == "__main__":
    args = get_config()
    port = get_port()
    args.dist_url = "tcp://127.0.0.1:" + str(port)
    main(args)
