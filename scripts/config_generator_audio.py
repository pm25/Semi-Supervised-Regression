# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

"""
Create the .yaml configuration file for each experiment.
"""
import os


def create_configuration(cfg, cfg_file):
    cfg["save_name"] = "{alg}_{cls_alg}_{dataset}_lb{num_lb}_s{seed}".format(
        alg=cfg["algorithm"],
        cls_alg=str(cfg["cls_algorithm"]).lower(),
        dataset=cfg["dataset"],
        num_lb=cfg["num_labels"],
        seed=cfg["seed"],
    )

    if cfg["algorithm"] == "fullysupervised":
        cfg["save_name"] = cfg["save_name"].replace(f"_lb{cfg['num_labels']}", "")
        cfg.pop("num_labels")

    # resume
    cfg["resume"] = True
    cfg["load_path"] = "{}/{}/latest_model.pth".format(cfg["save_dir"], cfg["save_name"])

    if cfg["cls_algorithm"] is None:
        alg_file = cfg_file + cfg["algorithm"] + "/"
    else:
        alg_file = cfg_file + "rankup/"

    if not os.path.exists(alg_file):
        os.mkdir(alg_file)

    print(alg_file + cfg["save_name"] + ".yaml")
    with open(alg_file + cfg["save_name"] + ".yaml", "w", encoding="utf-8") as w:
        lines = []
        for k, v in cfg.items():
            line = str(k) + ": " + str(v)
            lines.append(line)
        for line in lines:
            w.writelines(line)
            w.write("\n")


def get_algorithm_specific_config(reg_algorithm, cls_algorithm):
    alg_cfg = {}

    # regression algorithms
    if reg_algorithm == "pimodel":
        alg_cfg["uratio"] = 1
        alg_cfg["reg_ulb_loss_ratio"] = 0.1
        alg_cfg["reg_unsup_warm_up"] = 0.4
    elif reg_algorithm == "meanteacher":
        alg_cfg["uratio"] = 1
        alg_cfg["reg_ulb_loss_ratio"] = 0.1
        alg_cfg["reg_unsup_warm_up"] = 0.4
    elif reg_algorithm == "mixmatch":
        alg_cfg["uratio"] = 1
        alg_cfg["reg_ulb_loss_ratio"] = 0.1
        alg_cfg["reg_unsup_warm_up"] = 0.4
        alg_cfg["reg_mixup_alpha"] = 0.5
    elif reg_algorithm == "ucvme":
        alg_cfg["uratio"] = 1
        alg_cfg["reg_ulb_loss_ratio"] = 0.05
        alg_cfg["dropout_rate"] = 0.05
        alg_cfg["num_ensemble"] = 5
    elif reg_algorithm == "clss":
        alg_cfg["uratio"] = 1
        alg_cfg["reg_lb_ctr_loss_ratio"] = 1.0
        alg_cfg["reg_ulb_ctr_loss_ratio"] = 0.05
        alg_cfg["reg_ulb_rank_loss_ratio"] = 0.01
        alg_cfg["reg_lambda_val"] = 2.0
    elif reg_algorithm == "rda":
        alg_cfg["uratio"] = 1
        alg_cfg["reg_ulb_loss_ratio"] = 1.0
        alg_cfg["reg_unsup_warm_up"] = 0.4
        alg_cfg["rda_num_refine_iter"] = 1024

    # classification algorithms
    if cls_algorithm == "supervised":
        alg_cfg["cls_loss_ratio"] = 0.2
    elif cls_algorithm == "pseudolabel":
        alg_cfg["uratio"] = max(1, alg_cfg.get("uratio", 0))
        alg_cfg["cls_loss_ratio"] = 0.2
        alg_cfg["cls_ulb_loss_ratio"] = 0.1
        alg_cfg["cls_unsup_warm_up"] = 0.4
        alg_cfg["hard_label"] = True
        alg_cfg["p_cutoff"] = 0.95
    elif cls_algorithm == "pimodel":
        alg_cfg["uratio"] = max(1, alg_cfg.get("uratio", 0))
        alg_cfg["cls_loss_ratio"] = 0.2
        alg_cfg["cls_ulb_loss_ratio"] = 0.1
        alg_cfg["cls_unsup_warm_up"] = 0.4
    elif cls_algorithm == "meanteacher":
        alg_cfg["uratio"] = max(1, alg_cfg.get("uratio", 0))
        alg_cfg["cls_loss_ratio"] = 0.2
        alg_cfg["cls_ulb_loss_ratio"] = 0.1
        alg_cfg["cls_unsup_warm_up"] = 0.4
    elif cls_algorithm == "fixmatch":
        alg_cfg["uratio"] = max(1, alg_cfg.get("uratio", 0))
        alg_cfg["cls_loss_ratio"] = 0.2
        alg_cfg["cls_ulb_loss_ratio"] = 1.0
        alg_cfg["hard_label"] = True
        alg_cfg["T"] = 0.5
        alg_cfg["p_cutoff"] = 0.95

    return alg_cfg


def create_usb_audio_config(alg, seed, dataset, net, num_labels, port, lr, weight_decay, layer_decay, max_length_seconds, sample_rate):
    cfg = {}

    cfg["algorithm"] = alg[0]
    cfg["cls_algorithm"] = alg[1]

    # save config
    cfg["save_dir"] = "./saved_models/audio"
    cfg["save_name"] = None
    cfg["resume"] = True
    cfg["load_path"] = None
    cfg["overwrite"] = True
    cfg["use_tensorboard"] = True
    cfg["use_wandb"] = False
    cfg["use_aim"] = False

    # training config
    cfg["epoch"] = 100
    cfg["num_train_iter"] = 102400
    cfg["num_eval_iter"] = 1024
    cfg["num_log_iter"] = 256
    cfg["num_warmup_iter"] = int(1024 * 5)
    cfg["num_labels"] = num_labels
    cfg["batch_size"] = 8
    cfg["eval_batch_size"] = 16
    cfg["ema_m"] = 0.0

    alg_cfg = get_algorithm_specific_config(alg[0], alg[1])
    cfg.update(alg_cfg)

    # optimization config
    cfg["optim"] = "AdamW"
    cfg["lr"] = lr
    cfg["momentum"] = 0.9
    cfg["weight_decay"] = weight_decay
    cfg["layer_decay"] = layer_decay
    cfg["amp"] = False
    cfg["clip_grad"] = 0.0
    cfg["use_cat"] = False
    cfg["reg_criterion"] = "l1"

    # net config
    cfg["net"] = net
    cfg["net_from_name"] = False
    cfg["use_pretrain"] = True
    cfg["pretrain_path"] = "openai/whisper-base"

    # data config
    cfg["data_dir"] = "./data"
    cfg["dataset"] = dataset
    cfg["train_sampler"] = "RandomSampler"
    cfg["num_workers"] = 8
    cfg["max_length_seconds"] = max_length_seconds
    cfg["sample_rate"] = sample_rate
    cfg["preload"] = True

    # seed & distributed config
    cfg["seed"] = seed
    cfg["world_size"] = 1
    cfg["rank"] = 0
    cfg["multiprocessing_distributed"] = False
    cfg["dist_url"] = "tcp://127.0.0.1:" + str(port)
    cfg["dist_backend"] = "nccl"

    return cfg


def exp_usb_speech(dataset="bvcc", label_amount=250, seed=0, port=10001):
    configs_dir = r"./config/audio/"
    saved_dir = r"./saved_models/audio"

    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(saved_dir, exist_ok=True)

    algs = [
        ("supervised", None),
        ("fullysupervised", None),
        ("pimodel", None),
        ("meanteacher", None),
        ("ucvme", None),
        ("clss", None),
        ("mixmatch", None),
        ("rda", None),
        ("supervised", "supervised"),
        ("supervised", "pimodel"),
        ("supervised", "meanteacher"),
        ("supervised", "fixmatch"),
        ("pimodel", "fixmatch"),
        ("meanteacher", "fixmatch"),
        ("mixmatch", "fixmatch"),
        ("rda", "fixmatch"),
    ]

    weight_decay = 2e-5
    sampling_rate = 16000

    for alg in algs:
        # change the configuration of each dataset
        if dataset == "bvcc":
            max_length_seconds = 6.0
            net = "whisper_base"
            lr = 2e-6
            layer_decay = 0.75

        # prepare the configuration file
        cfg = create_usb_audio_config(
            alg,
            seed,
            dataset,
            net,
            label_amount,
            port,
            lr,
            weight_decay,
            layer_decay,
            max_length_seconds,
            sampling_rate,
        )
        create_configuration(cfg, configs_dir)


if __name__ == "__main__":
    label_amount = {
        "bvcc": [250],
    }
    seeds = [0, 1, 2]

    for dataset, label_amount_list in label_amount.items():
        dist_port = list(range(10001, 10001 + len(label_amount_list) * len(seeds)))
        for label_amount in label_amount_list:
            for seed in seeds:
                port = dist_port.pop(0)
                exp_usb_speech(dataset=dataset, label_amount=label_amount, seed=seed, port=port)
