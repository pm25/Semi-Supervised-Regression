# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import random
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau, gmean
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.backends.cudnn as cudnn

from semilearn.core.utils import get_net_builder, get_dataset, TorchMinMaxScaler, over_write_args_from_file, get_data_loader


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True


def get_scaler(input_range, output_range):
    scaler = TorchMinMaxScaler(output_range=output_range)
    scaler.fit(torch.tensor(input_range))
    return scaler


def save_features(features, labels, save_path):
    out_feats = {"feat": features, "label": labels}
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(save_path, out_feats)
    print(f"Successfully saved features to: {save_path}.npy")


def load_model_state_dict(load_model):
    """Extract and clean model state dict."""
    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith("module") or key.startswith("backbone"):
            new_key = ".".join(key.split(".")[1:])
            load_state_dict[new_key] = item
        elif key.startswith("arc_classifier"):  # Skip arc weights
            continue
        else:
            load_state_dict[key] = item
    return load_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_path", type=str, required=False)

    # Backbone Net Configurations
    parser.add_argument("--net", type=str, default="wrn_28_2")
    parser.add_argument("--net_from_name", type=bool, default=False)
    parser.add_argument("--pretrain_path", type=str, default="")

    # Data Configurations
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="utkface")
    parser.add_argument("--img_size", type=int, default=40)
    parser.add_argument("--crop_ratio", type=float, default=0.875)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_length_seconds", type=float, default=4.0)
    parser.add_argument("--sample_rate", type=int, default=16000)

    # Seed & GPU Configurations
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")

    parser.add_argument("--save_features", action="store_true")
    parser.add_argument("--c", type=str, default="")  # config file

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    # Ensure the model path is correct when loading from config
    if args.c:
        args.load_path = str(Path(args.save_dir) / args.save_name / "model_best.pth")

    set_seed(args.seed)
    args.preload = False

    print(f"Loading the Model: {args.load_path}")
    try:
        checkpoint = torch.load(args.load_path)
    except FileNotFoundError:
        print(f"Error: Checkpoint file {args.load_path} not found.")
        exit(1)

    load_model = checkpoint["ema_model"]
    load_state_dict = load_model_state_dict(load_model)

    # Initialize network and load the model weights
    net_builder = get_net_builder(args.net, args.net_from_name)
    net = net_builder()
    net.load_state_dict(load_state_dict)
    print("Model Loaded")

    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu")
    net.to(device)
    net.eval()

    # Load dataset and evaluation loader
    dataset_dict = get_dataset(args, "fullysupervised", None, args.dataset, 0, args.data_dir, False)
    eval_dset = dataset_dict["eval"]
    eval_loader = get_data_loader(args, eval_dset, args.eval_batch_size, data_sampler=None, num_epochs=0, num_iters=0, num_workers=4, drop_last=False)

    # Load input/output ranges from checkpoint if available
    lb_min = dataset_dict["train_lb"].targets.min()
    lb_max = dataset_dict["train_lb"].targets.max()
    input_range = checkpoint.get("input_range", (lb_min, lb_max))
    output_range = checkpoint.get("output_range", (0, 1))

    # Initialize scaler
    scaler = get_scaler(input_range, output_range)

    y_true, y_pred, x_feats = [], [], []
    with torch.no_grad():
        for data in tqdm(eval_loader, total=len(eval_loader)):
            x, y = data["x_lb"].to(device), data["y_lb"].to(device)

            # Forward pass
            feat = net(x, only_feat=True)
            logits = net(feat, only_fc=True)

            inv_logits = scaler.inverse_transform(logits.cpu())

            # Collect outputs
            y_true.append(y.cpu())
            y_pred.append(inv_logits)
            x_feats.append(feat.cpu().numpy())

    # Concatenate results
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    x_feats = np.concatenate(x_feats)

    # Save features for visualization (e.g., t-SNE/UMAP)
    if args.save_features:
        save_name = Path(args.load_path).parent.name
        save_path = f"./visualization/features/{save_name}"
        save_features(x_feats, y_true, save_path=save_path)

    # Evaluation metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    lcc, _ = pearsonr(y_true, y_pred)
    srcc, _ = spearmanr(y_true, y_pred)
    ktau, _ = kendalltau(y_true, y_pred)
    gmae = gmean(np.abs(y_true - y_pred))

    # Print results
    print(f"MAE : {mae:.5f}")
    print(f"MSE : {mse:.5f}")
    print(f"R2  : {r2:.5f}")
    print(f"LCC : {lcc:.5f}")
    print(f"SRCC: {srcc:.5f}")
    print(f"KTAU: {ktau:.5f}")
    print(f"GMAE: {gmae:.5f}")
