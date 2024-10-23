# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

import re
import csv
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

parser = argparse.ArgumentParser(description="Calculate and average metrics from log files.")
parser.add_argument("--log_dir", type=str, default="./saved_models", help="Directory to load log files from")
parser.add_argument("--out_dir", type=str, default="./results", help="Directory to save output CSV files")
parser.add_argument("--type", type=str, default="txt", choices=["txt", "tensorboard"], help="Log file type to process")
parser.add_argument("--detailed", action="store_true", help="Output detailed logs for each run")
parser.add_argument("--data_types", type=str, nargs="+", default=["classic_cv", "cv", "audio", "nlp"], help="Data types to process")
parser.add_argument("--precision", type=int, default=3, help="Number of decimal places for metric calculations")
args = parser.parse_args()

log_dir = Path(args.log_dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(exist_ok=True, parents=True)


def get_txt_static(file_path):
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found at: {file_path}")

    re_bestMAE = r"BEST_EVAL_MAE: (([0-9]|\.)*)"  # .group(1)
    re_bestIt = r"at ([0-9]*)"  # .group(1)
    re_MAE = r"eval\/mae: (([0-9]|\.)*)"
    re_MSE = r"eval\/mse: (([0-9]|\.)*)"
    re_R2 = r"eval\/r2: (([0-9]|\.|-)*)"
    re_LCC = r"eval\/lcc: (([0-9]|\.|-)*)"
    re_SRCC = r"eval\/srcc: (([0-9]|\.|-)*)"
    re_KTAU = r"eval\/ktau: (([0-9]|\.|-)*)"
    re_GMAE = r"eval\/gmae: (([0-9]|\.|-)*)"

    stat = {
        "bestMAE": None,
        "bestIt": None,
        "MAE": [],
        "MSE": [],
        "R2": [],
        "LCC": [],
        "SRCC": [],
        "KTAU": [],
        "GMAE": [],
        "Finish": False,
    }

    with open(file_path, "r", encoding="utf-8") as f:
        lines = list(f.readlines())

    for line in lines:
        if line.endswith("iters\n"):
            stat["bestMAE"] = float(re.search(re_bestMAE, line).group(1))
            stat["bestIt"] = int(re.search(re_bestIt, line).group(1))
            stat["MAE"].append(float(re.search(re_MAE, line).group(1)))
            stat["MSE"].append(float(re.search(re_MSE, line).group(1)))
            stat["R2"].append(float(re.search(re_R2, line).group(1)))
            stat["LCC"].append(float(re.search(re_LCC, line).group(1)))
            stat["SRCC"].append(float(re.search(re_SRCC, line).group(1)))
            stat["KTAU"].append(float(re.search(re_KTAU, line).group(1)))
            stat["GMAE"].append(float(re.search(re_GMAE, line).group(1)))
        elif "Model result" in line:
            stat["Finish"] = True

    is_empty = False
    for s in stat.values():
        if (isinstance(s, list) and len(s) == 0) or (s is None):
            is_empty = True
            break

    return stat, is_empty


def save_stats(statics, save_name="log.csv"):
    csv_file = open(out_dir / save_name, "w", newline="")
    fieldnames = ["exp_name", "seed", "finish", "min_MAE", "min_MSE", "max_R2", "max_LCC", "max_SRCC", "max_KTAU", "min_GMAE"]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    for name, seed_stats in statics.items():
        for _seed, _stats in seed_stats.items():
            row = {"exp_name": name, "seed": _seed, "finish": _stats["Finish"]}
            for key in fieldnames[3:]:
                s = np.array(_stats[key])
                row[key] = f"{s.mean():.{args.precision}f}±{s.std():.{args.precision}f}"
            csv_writer.writerow(row)
    csv_file.close()
    print(f"Successfully saved statistics to: {out_dir / save_name}")


def save_average_stats(statics, save_name="average_log.csv"):
    csv_file = open(out_dir / save_name, "w", newline="")
    fieldnames = ["exp_name", "num_exp", "min_MAE", "min_MSE", "max_R2", "max_LCC", "max_SRCC", "max_KTAU", "min_GMAE"]
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    for name, seed_stats in statics.items():
        stats = {}
        for _seed, _stats in seed_stats.items():
            if _stats["Finish"] == False:
                continue
            for metric, value in _stats.items():
                stats[metric] = stats.get(metric, []) + [value]
        if len(stats) == 0:
            continue
        row = {"exp_name": name, "num_exp": sum(stats["Finish"])}
        for key in fieldnames[2:]:
            s = np.array(stats[key])
            row[key] = f"{s.mean():.{args.precision}f}±{s.std():.{args.precision}f}"
        csv_writer.writerow(row)
    csv_file.close()
    print(f"Successfully saved average statistics to: {out_dir / save_name}")


def calc_average_log(log_paths):
    exp_statics = {}
    for log_path in log_paths:
        stats, is_empty = get_txt_static(log_path)
        if not is_empty:
            exp_statics[log_path.parent.name] = stats

    out_statics = {}
    for name, statics in tqdm(sorted(exp_statics.items())):
        _seed = re.search(r"_s([0-9]*)$", name)
        if _seed is None:
            print(f"can't find the SEED number, skip {name}.")
            continue
        seed = _seed.group(1)
        name = re.sub(r"_s([0-9]*)$", "", name)

        if name not in out_statics.keys():
            out_statics[name] = {}
        out_statics[name][seed] = {}

        for metric, values in statics.items():
            if isinstance(values, list):
                values = np.array(values)
                out_statics[name][seed][f"max_{metric}"] = values.max()
                out_statics[name][seed][f"min_{metric}"] = values.min()
                out_statics[name][seed][f"avg_{metric}"] = values.mean()
                out_statics[name][seed][f"std_{metric}"] = values.std()
                out_statics[name][seed][f"last_{metric}"] = values[-1]
                out_statics[name][seed][f"last_10_{metric}"] = values[-10:].mean()
                out_statics[name][seed][f"last_20_{metric}"] = values[-20:].mean()
            else:
                out_statics[name][seed][metric] = values

    return out_statics


if __name__ == "__main__":
    # data_types = ["classic_cv", "cv", "audio", "nlp"]
    for data_type in args.data_types:
        log_paths = log_dir.glob(f"./{data_type}/**/log.txt")
        statics = calc_average_log(log_paths)
        if statics != {}:
            if args.detailed:
                save_stats(statics, save_name=f"{data_type}_log.csv")
            save_average_stats(statics, save_name=f"{data_type}_average_log.csv")
