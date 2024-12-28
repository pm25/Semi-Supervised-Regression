# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import aim

from .hook import Hook


class AimHook(Hook):
    """
    A hook for tracking training progress with Aim.
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
        """Setup the Aim tracking.

        Args:
            algorithm (AlgorithmBase): The training algorithm.
        """
        # initialize aim run
        name = algorithm.save_name
        project = algorithm.save_dir.split("/")[-1]
        repo = algorithm.args.save_dir.split("/")[-2]
        self.run = aim.Run(experiment=name, repo=repo, log_system_params=True)

        # set configuration
        self.run["hparams"] = algorithm.args.__dict__

        # set tags
        benchmark = f"benchmark: {project}"
        dataset = f"dataset: {algorithm.args.dataset}"
        data_setting = f"setting: {algorithm.args.dataset}_lb{algorithm.args.num_labels}_ulb{algorithm.args.ulb_num_labels}"
        alg = f"alg: {algorithm.args.algorithm}"
        self.run.add_tag(benchmark)
        self.run.add_tag(dataset)
        self.run.add_tag(data_setting)
        self.run.add_tag(alg)

    def after_train_step(self, algorithm):
        """Log the metric values in the log dictionary to Aim.

        Args:
            algorithm (AlgorithmBase): The training algorithm.
        """
        if self.every_n_iters(algorithm, algorithm.num_log_iter):
            for key, item in algorithm.log_dict.items():
                if key in self.log_key_list:
                    self.run.track(item, name=key, step=algorithm.it)

        if self.every_n_iters(algorithm, algorithm.num_eval_iter):
            self.run.track(algorithm.best_eval_mae, name="eval/best-mae", step=algorithm.it)
