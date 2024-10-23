# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import os
import contextlib
import numpy as np
from inspect import signature
from collections import OrderedDict

import torch
from torch.cuda.amp import GradScaler, autocast
from scipy.stats import pearsonr, spearmanr, kendalltau, gmean
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from semilearn.nets import rankup_wrapper
from semilearn.core.criterions import CELoss, ClsConsistencyLoss, RegConsistencyLoss
from semilearn.core.hooks import (
    AimHook,
    CheckpointHook,
    DistSamplerSeedHook,
    EMAHook,
    EvaluationHook,
    Hook,
    LoggingHook,
    ParamUpdateHook,
    TimerHook,
    WANDBHook,
    get_priority,
)
from semilearn.core.utils import (
    Bn_Controller,
    TorchMinMaxScaler,
    get_cosine_schedule_with_warmup,
    get_data_loader,
    get_dataset,
    get_optimizer,
    get_criterion,
)


class AlgorithmBase:
    """
    Base class for algorithms
    init algorithm specific parameters and common parameters

    Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        # common arguments
        self.args = args
        self.ema_m = args.ema_m
        self.epochs = args.epoch
        self.num_train_iter = args.num_train_iter
        self.num_eval_iter = args.num_eval_iter
        self.num_log_iter = args.num_log_iter
        self.num_iter_per_epoch = int(self.num_train_iter // self.epochs)
        self.reg_criterion = args.reg_criterion
        self.reg_ulb_loss_ratio = args.reg_ulb_loss_ratio
        self.use_cat = args.use_cat
        self.use_amp = args.amp
        self.clip_grad = args.clip_grad
        self.save_name = args.save_name
        self.save_dir = args.save_dir
        self.resume = args.resume
        self.algorithm = args.algorithm
        self.cls_algorithm = args.cls_algorithm

        # common utils arguments
        self.tb_log = tb_log
        self.print_fn = print if logger is None else logger.info
        self.ngpus_per_node = torch.cuda.device_count()
        self.loss_scaler = GradScaler()
        self.amp_cm = autocast if self.use_amp else contextlib.nullcontext
        self.gpu = args.gpu
        self.rank = args.rank
        self.distributed = args.distributed
        self.world_size = args.world_size

        # common model related parameters
        self.it = 0
        self.start_epoch = 0
        self.best_eval_mae, self.best_it = float("inf"), 0
        self.bn_controller = Bn_Controller()
        self.net_builder = net_builder
        self.ema = None

        # build dataset
        self.dataset_dict = self.set_dataset()
        lb_min = self.dataset_dict["train_lb"].targets.min()
        lb_max = self.dataset_dict["train_lb"].targets.max()
        self.input_range = (lb_min, lb_max)  # should only calculate the input_range on train_lb data
        self.output_range = (0, 1)

        # set scaler
        self.scaler = self.set_scaler(self.input_range, self.output_range)

        # build data loader
        self.loader_dict = self.set_data_loader()

        # cv, nlp, speech builder different arguments
        self.model = self.set_model()
        self.ema_model = self.set_ema_model()

        # build optimizer and scheduler
        self.optimizer, self.scheduler = self.set_optimizer()

        # build supervised loss and unsupervised loss
        self.reg_loss, self.reg_consistency_loss = self.set_reg_criterions()

        # other arguments specific to the algorithm
        # self.init(**kwargs)

        # set common hooks during training
        self._hooks = []  # record underlying hooks
        self.hooks_dict = OrderedDict()  # actual object to be used to call hooks
        self.set_hooks()

    def init(self, **kwargs):
        """
        algorithm specific init function, to add parameters into class
        """
        raise NotImplementedError

    def set_dataset(self):
        """
        set dataset_dict
        """
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        dataset_dict = get_dataset(
            self.args,
            self.algorithm,
            self.cls_algorithm,
            self.args.dataset,
            self.args.num_labels,
            self.args.data_dir,
            self.args.include_lb_to_ulb,
        )
        if dataset_dict is None:
            return dataset_dict

        self.args.ulb_dest_len = len(dataset_dict["train_ulb"]) if dataset_dict["train_ulb"] is not None else 0
        self.args.lb_dest_len = len(dataset_dict["train_lb"])
        self.print_fn("unlabeled data number: {}, labeled data number {}".format(self.args.ulb_dest_len, self.args.lb_dest_len))
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict

    def set_data_loader(self):
        """
        set loader_dict
        """
        if self.dataset_dict is None:
            return

        self.print_fn("Create train and test data loaders")
        loader_dict = {}
        loader_dict["train_lb"] = get_data_loader(
            self.args,
            self.dataset_dict["train_lb"],
            self.args.batch_size,
            data_sampler=self.args.train_sampler,
            num_iters=self.num_train_iter,
            num_epochs=self.epochs,
            num_workers=self.args.num_workers,
            distributed=self.distributed,
        )

        loader_dict["train_ulb"] = get_data_loader(
            self.args,
            self.dataset_dict["train_ulb"],
            int(self.args.batch_size * self.args.uratio),
            data_sampler=self.args.train_sampler,
            num_iters=self.num_train_iter,
            num_epochs=self.epochs,
            num_workers=2 * self.args.num_workers,
            distributed=self.distributed,
        )

        loader_dict["eval"] = get_data_loader(
            self.args,
            self.dataset_dict["eval"],
            self.args.eval_batch_size,
            # make sure data_sampler is None for evaluation
            data_sampler=None,
            num_workers=self.args.num_workers,
            drop_last=False,
        )

        if self.dataset_dict["test"] is not None:
            loader_dict["test"] = get_data_loader(
                self.args,
                self.dataset_dict["test"],
                self.args.eval_batch_size,
                # make sure data_sampler is None for evaluation
                data_sampler=None,
                num_workers=self.args.num_workers,
                drop_last=False,
            )
        self.print_fn(f"[!] data loader keys: {loader_dict.keys()}")
        return loader_dict

    def set_optimizer(self):
        """
        set optimizer for algorithm
        """
        self.print_fn("Create optimizer and scheduler")
        optimizer = get_optimizer(
            self.model,
            self.args.optim,
            self.args.lr,
            self.args.momentum,
            self.args.weight_decay,
            self.args.layer_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, self.num_train_iter, num_warmup_steps=self.args.num_warmup_iter)
        return optimizer, scheduler

    def set_reg_criterions(self):
        reg_loss = get_criterion(self.reg_criterion)
        reg_consistency_loss = RegConsistencyLoss()
        return reg_loss, reg_consistency_loss

    def set_model(self, **kwargs):
        """
        initialize model
        """
        model = self.net_builder(
            pretrained=self.args.use_pretrain,
            pretrained_path=self.args.pretrain_path,
            **kwargs
        )
        return model

    def set_ema_model(self, **kwargs):
        """
        initialize ema model from model
        """
        ema_model = self.net_builder(
            pretrained=self.args.use_pretrain,
            pretrained_path=self.args.pretrain_path,
            **kwargs
        )
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def set_hooks(self):
        """
        register necessary training hooks
        """
        # parameter update hook is called inside each train_step
        self.register_hook(ParamUpdateHook(), None, "HIGHEST")
        self.register_hook(EMAHook(), None, "HIGH")
        self.register_hook(EvaluationHook(), None, "HIGH")
        self.register_hook(CheckpointHook(), None, "HIGH")
        self.register_hook(DistSamplerSeedHook(), None, "NORMAL")
        self.register_hook(TimerHook(), None, "LOW")
        self.register_hook(LoggingHook(), None, "LOWEST")
        if self.args.use_wandb:
            self.register_hook(WANDBHook(), None, "LOWEST")
        if self.args.use_aim:
            self.register_hook(AimHook(), None, "LOWEST")

    def set_scaler(self, input_range=(0, 1), output_range=(0, 1)):
        scaler = TorchMinMaxScaler(output_range=output_range)
        scaler.fit(torch.tensor(input_range))
        for dataset in self.dataset_dict.values():
            if dataset is None:
                continue
            dataset.targets = scaler.transform(torch.from_numpy(dataset.targets)).numpy()
        return scaler

    def process_batch(self, input_args=None, **kwargs):
        """
        process batch data, send data to cuda
        NOTE: **kwargs should have the same arguments to train_step function as keys to
        work properly.
        """
        if input_args is None:
            input_args = signature(self.train_step).parameters
            input_args = list(input_args.keys())

        input_dict = {}

        for arg, var in kwargs.items():
            if arg not in input_args:
                continue

            if var is None:
                continue

            # send var to cuda
            if isinstance(var, dict):
                var = {k: v.cuda(self.gpu) for k, v in var.items()}
            else:
                var = var.cuda(self.gpu)
            input_dict[arg] = var
        return input_dict

    def process_out_dict(self, out_dict=None, **kwargs):
        """
        process the out_dict as return of train_step
        """
        if out_dict is None:
            out_dict = {}

        for arg, var in kwargs.items():
            out_dict[arg] = var

        # process res_dict, add output from res_dict to out_dict if necessary
        return out_dict

    def process_log_dict(self, log_dict=None, prefix="train", **kwargs):
        """
        process the tb_dict as return of train_step
        """
        if log_dict is None:
            log_dict = {}

        for arg, var in kwargs.items():
            log_dict[f"{prefix}/" + arg] = var
        return log_dict

    def compute_prob(self, logits):
        return torch.softmax(logits, dim=-1)

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        """
        train_step specific to each algorithm
        """
        # implement train step for each algorithm
        # compute loss
        # update model
        # record log_dict
        # return log_dict
        raise NotImplementedError

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(self.loader_dict["train_lb"], self.loader_dict["train_ulb"]):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def evaluate(self, eval_dest="eval", out_key="logits", return_logits=False):
        """
        evaluation function
        """
        self.model.eval()
        self.ema.apply_shadow()

        eval_loader = self.loader_dict[eval_dest]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data in eval_loader:
                x = data["x_lb"]
                y = data["y_lb"]

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)

                y = y.cuda(self.gpu)
                inv_y = self.scaler.inverse_transform(torch.clone(y.cpu()))

                num_batch = y.shape[0]
                total_num += num_batch

                logits = self.model(x)[out_key]
                inv_logits = self.scaler.inverse_transform(torch.clone(logits.cpu()))

                loss = self.reg_loss(logits, y, reduction="mean")
                y_true.extend(inv_y.cpu().tolist())
                y_pred.extend(inv_logits.cpu().tolist())
                total_loss += loss.item() * num_batch
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        lcc, _ = pearsonr(y_true, y_pred)
        srcc, _ = spearmanr(y_true, y_pred)
        ktau, _ = kendalltau(y_true, y_pred)
        raw_mae = np.abs(y_true - y_pred)
        gmae = gmean(np.where(raw_mae == 0, 1e-5, raw_mae))

        self.ema.restore()
        self.model.train()

        eval_dict = {
            eval_dest + "/loss": total_loss / total_num,
            eval_dest + "/mae": mae,
            eval_dest + "/mse": mse,
            eval_dest + "/r2": r2,
            eval_dest + "/lcc": lcc,
            eval_dest + "/srcc": srcc,
            eval_dest + "/ktau": ktau,
            eval_dest + "/gmae": gmae,
        }
        if return_logits:
            eval_dict[eval_dest + "/logits"] = y_pred
        return eval_dict

    def get_save_dict(self):
        """
        Create a dictionary of additional arguments to save for when saving the model.
        """
        # Base arguments for all models
        save_dict = {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss_scaler": self.loss_scaler.state_dict(),
            "it": self.it + 1,
            "epoch": self.epoch + 1,
            "best_it": self.best_it,
            "best_eval_mae": self.best_eval_mae,
            "input_range": self.input_range,
            "output_range": self.output_range,
        }

        if self.scheduler is not None:
            # Using a scheduler, so save its state dict
            save_dict["scheduler"] = self.scheduler.state_dict()

        if hasattr(self, "aim_run_hash"):
            # Using Aim, so save the run hash so that tracking can be resumed
            save_dict["aim_run_hash"] = self.aim_run_hash

        return save_dict

    def save_model(self, save_name, save_path):
        """
        save model and specified parameters for resume
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        save_filename = os.path.join(save_path, save_name)
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_filename)
        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        """
        Load a model and the necessary parameters for resuming training.
        """
        checkpoint = torch.load(load_path, map_location="cpu")

        self.model.load_state_dict(checkpoint["model"])
        self.ema_model.load_state_dict(checkpoint["ema_model"])
        self.loss_scaler.load_state_dict(checkpoint["loss_scaler"])
        self.it = checkpoint["it"]
        self.start_epoch = checkpoint["epoch"]
        self.epoch = self.start_epoch
        self.best_it = checkpoint["best_it"]
        self.best_eval_mae = checkpoint["best_eval_mae"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.scheduler is not None and "scheduler" in checkpoint:
            # Using a scheduler, so load its state dict
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if "aim_run_hash" in checkpoint:
            # Using Aim, so load the run hash so that tracking can be resumed
            self.aim_run_hash = checkpoint["aim_run_hash"]

        self.print_fn("Model loaded")

        return checkpoint

    def check_prefix_state_dict(self, state_dict):
        """
        remove prefix state dict in ema model
        """
        new_state_dict = dict()
        for key, item in state_dict.items():
            if key.startswith("module"):
                new_key = ".".join(key.split(".")[1:])
            else:
                new_key = key
            new_state_dict[new_key] = item
        return new_state_dict

    def register_hook(self, hook, name=None, priority="NORMAL"):
        """
        Ref: https://github.com/open-mmlab/mmcv/blob/a08517790d26f8761910cac47ce8098faac7b627/mmcv/runner/base_runner.py#L263  # noqa: E501
        Register a hook into the hook list.
        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            hook_name (:str, default to None): Name of the hook to be registered.
                Default is the hook class name.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, "priority"):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  # type: ignore
        hook.name = name if name is not None else type(hook).__name__

        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  # type: ignore
                self._hooks.insert(i + 1, hook)
                inserted = True
                break

        if not inserted:
            self._hooks.insert(0, hook)

        # call set hooks
        self.hooks_dict = OrderedDict()
        for hook in self._hooks:
            self.hooks_dict[hook.name] = hook

    def call_hook(self, fn_name, hook_name=None, *args, **kwargs):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            hook_name (str): The specific hook name to be called, such as
                "param_update" or "dist_align", used to call single hook in train_step.
        """

        if hook_name is not None:
            return getattr(self.hooks_dict[hook_name], fn_name)(self, *args, **kwargs)

        for hook in self.hooks_dict.values():
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args, **kwargs)

    def registered_hook(self, hook_name):
        """
        Check if a hook is registered
        """
        return hook_name in self.hooks_dict

    @staticmethod
    def get_argument():
        """
        Get specified arguments into argparse for each algorithm
        """
        return {}


class ClsAlgorithmBase(AlgorithmBase):
    """
    Parent class wrapping the AlgorithmBase class for classification algorithms.
    Initializes classification-specific parameters and overrides necessary functions.

    Attributes:
        cls_ulb_loss_ratio (float): The ratio for the unsupervised loss in classification tasks.
        cls_loss_ratio (float): The ratio for the classification loss in the overall loss with (regression + classification).
        cls_loss: The criterion for calculating supervised classification loss.
        cls_consistency_loss: The criterion for calculating semi-supervised consistency loss in classification.
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        net_builder = rankup_wrapper(net_builder)
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        self.cls_ulb_loss_ratio = args.cls_ulb_loss_ratio
        self.cls_loss_ratio = args.cls_loss_ratio
        self.cls_loss, self.cls_consistency_loss = self.set_cls_criterions()

    def cls_init(self, *args, **kwargs):
        """
        initialize classification algorithm parameters
        """
        pass

    def set_cls_criterions(self):
        cls_loss = CELoss()
        cls_consistency_loss = ClsConsistencyLoss()
        return cls_loss, cls_consistency_loss

    def process_batch(self, input_args=None, **kwargs):
        """
        process batch data, send data to cuda
        NOTE: **kwargs should have the same arguments to train_step function as keys to
        work properly.
        """
        if input_args is None:
            reg_input_args = signature(super().train_step).parameters
            reg_input_args = set(reg_input_args.keys())
            cls_input_args = signature(self.train_step).parameters
            cls_input_args = set(cls_input_args.keys())
            input_args = reg_input_args.union(cls_input_args)

        input_dict = {}

        for arg, var in kwargs.items():
            if arg not in input_args:
                continue

            if var is None:
                continue

            # send var to cuda
            if isinstance(var, dict):
                var = {k: v.cuda(self.gpu) for k, v in var.items()}
            else:
                var = var.cuda(self.gpu)
            input_dict[arg] = var

        return input_dict
