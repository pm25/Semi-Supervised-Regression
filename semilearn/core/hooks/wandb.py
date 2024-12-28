# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import os
import wandb

from .hook import Hook


class WANDBHook(Hook):
    """
    Wandb Hook
    """

    def __init__(self):
        super().__init__()
        self.log_key_list = [
            "train/sup_loss",
            "train/unsup_loss",
            "train/total_loss",
            "train/util_ratio",
            "train/run_time",
            "train/prefetch_time",
            "lr",
            "eval/mae",
            "eval/mse",
            "eval/r2",
            "eval/lcc",
            "eval/srcc",
            "eval/ktau",
            "eval/gmae",
        ]

    def before_run(self, algorithm):
        name = algorithm.save_name
        project = "ssr_" + algorithm.save_dir.split("/")[-1]
        group = "_".join(algorithm.args.save_name.split("_")[:-1])

        # tags
        benchmark = f"benchmark: {project}"
        dataset = f"dataset: {algorithm.args.dataset}"
        data_setting = f"setting: {algorithm.args.dataset}_lb{algorithm.args.num_labels}_ulb{algorithm.args.ulb_num_labels}"
        alg = f"alg: {algorithm.args.algorithm}"
        tags = [benchmark, dataset, data_setting, alg]
        if algorithm.args.resume:
            resume = "auto"
        else:
            resume = "never"
        # resume = 'never'

        save_dir = os.path.join(algorithm.args.save_dir, "wandb", algorithm.args.save_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.run = wandb.init(name=name, tags=tags, config=algorithm.args.__dict__, project=project, group=group, resume=resume, dir=save_dir)

    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_log_iter):
            log_dict = {}
            for key, item in algorithm.log_dict.items():
                if key in self.log_key_list:
                    log_dict[key] = item
            self.run.log(log_dict, step=algorithm.it)

        if self.every_n_iters(algorithm, algorithm.num_eval_iter):
            self.run.log({"eval/best-mae": algorithm.best_eval_mae}, step=algorithm.it)

    def after_run(self, algorithm):
        self.run.finish()
