import re
from pathlib import Path
from typing import Dict, Union

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def read_flower_mnist_log_file(path: Path):
    lines = path.read_text().splitlines()
    progress_lines = [l for l in lines if "fit progress: (" in l]
    entries = []
    round_start_lines = [l for l in lines if "fit_round: strategy sampled" in l]
    first_round_start_time_secs = 0.0
    if len(round_start_lines) > 0:
        first_round_start_time_secs = pd.to_datetime(
            " ".join(round_start_lines[0].split(" ")[2:4])
        ).timestamp()
    for l in progress_lines:
        timestamp = pd.to_datetime(" ".join(l.split(" ")[2:4])).timestamp()
        x = l.split("fit progress: ")[-1]
        round, loss, metrics, time = eval(x)
        # print(first_round_start_time_secs, time - first_round_start_time_secs, time)
        entries.append(
            {
                "round": round,
                "loss": loss,
                "metrics": metrics,
                "accuracy": metrics.get("accuracy"),
                "time_since_start": time,
                "time": (timestamp - first_round_start_time_secs)
                if len(entries) == 0
                else (time - entries[-1]["time_since_start"]),
            }
        )
    times_agg_eval = []
    for idx, line in enumerate(lines):
        if idx + 1 == len(lines):
            break
        next_line = lines[idx + 1]
        if "fit_round received" not in line:
            continue
        if "fit progress: (" in next_line:
            t_start = pd.to_datetime(" ".join(line.split(" ")[2:4]))
            t_end = pd.to_datetime(" ".join(next_line.split(" ")[2:4]))
            times_agg_eval.append((t_end - t_start).total_seconds())
    # "time_agg_eval": time_agg_eval,
    df = pd.DataFrame.from_records(entries)
    df["time_agg_eval"] = times_agg_eval
    new_dtypes = {
        "time": float,
        "loss": float,
        "accuracy": float,
        "time_since_start": float,
        "round": int,
        "time_agg_eval": float,
    }
    if not df.empty:
        df = df.astype(new_dtypes)
    return df


def read_flower_leaf_log_file(f_err: Path, f_out: Path):
    out_lines = f_out.read_text().splitlines()

    # Metrics
    eval_entries = []
    for idx, l in enumerate(out_lines):
        matches = re.findall(r"EvaluateRes([^\)]+)", l)
        client_accs = []
        client_cardinalities = []
        client_losses = []
        for m in matches:
            eval_dict = eval(f"dict{m})")
            client_acc = eval_dict.get("metrics").get("accuracy")
            client_cardinality = eval_dict.get("num_examples")
            client_loss = eval_dict.get("loss")
            client_accs.append(client_acc)
            client_cardinalities.append(client_cardinality)
            client_losses.append(client_loss)
        if len(matches) == 0:
            continue
        acc = np.average(client_accs, weights=client_cardinalities)
        loss = np.average(client_losses, weights=client_cardinalities)
        eval_entries.append({"round": idx + 1, "accuracy": acc, "loss": loss})
    eval_entries = eval_entries[:-1]
    df = pd.DataFrame.from_records(eval_entries)
    if not df.empty:
        df = df.astype({"round": int, "accuracy": float, "loss": float})

    # Timing info
    err_lines = f_err.read_text().splitlines()
    timing_entries = []
    t_start_training = None
    for idx, line in enumerate(err_lines):
        if "fit_round: strategy sampled" not in line:
            continue
        try:
            received_line = err_lines[idx + 1]
            eval_start_line = err_lines[idx + 2]
            eval_end_line = err_lines[idx + 3]
            assert "evaluate_round received" in eval_end_line
            assert "evaluate_round: strategy sampled" in eval_start_line
            assert "fit_round received" in received_line
        except (IndexError, AssertionError) as e:
            continue
        t_fit_start = pd.to_datetime(" ".join(line.split(" ")[2:4]))
        t_fit_end = pd.to_datetime(" ".join(received_line.split(" ")[2:4]))
        t_eval_start = pd.to_datetime(" ".join(eval_start_line.split(" ")[2:4]))
        t_eval_end = pd.to_datetime(" ".join(eval_end_line.split(" ")[2:4]))
        if not t_start_training:
            t_start_training = t_fit_start
        total_time = (t_eval_end - t_fit_start).total_seconds()
        timing_entries.append(
            {
                "time": total_time,
                "time_eval": (t_eval_end - t_eval_start).total_seconds(),
                "time_agg_eval": (t_eval_end - t_fit_end).total_seconds(),
                "time_clients_fit": (t_fit_end - t_fit_start).total_seconds(),
                "time_since_start": (t_eval_end - t_start_training).total_seconds(),
            }
        )
        # total_seconds
    #    assert len(timing_entries) == len(df)
    timing_df = pd.DataFrame.from_records(timing_entries[: len(df)])

    return df.join(timing_df)


def process_flower_logs(root: Union[str, Path]):
    root = Path(root)
    files = []
    dfs = []

    for f in root.glob("fedless_*.err"):
        if (len(f.name.split("_"))) == 6:  # Local Client Log
            (
                _,
                dataset,
                clients_in_round,
                clients_total,
                local_epochs,
                seed,
            ) = f.name.split("_")
            batch_size = 5
        elif (len(f.name.split("_"))) == 7:  # Local Client Log
            (
                _,
                dataset,
                clients_in_round,
                clients_total,
                local_epochs,
                batch_size,
                seed,
            ) = f.name.split("_")
        else:
            continue
        seed = seed.split(".")[0]
        if dataset == "mnist":  # All required data lies in .err file
            logs_df = read_flower_mnist_log_file(f)
        elif dataset in ["femnist", "shakespeare"]:
            logs_df = read_flower_leaf_log_file(f_err=f, f_out=f.with_suffix(".out"))

        if logs_df.empty:
            continue

        index = pd.MultiIndex.from_tuples(
            [(dataset, clients_in_round, clients_total, local_epochs, batch_size, seed)]
            * len(logs_df),
            names=[
                "dataset",
                "clients_in_round",
                "clients_total",
                "local_epochs",
                "batch_size",
                "seed",
            ],
        )
        df = pd.DataFrame(
            logs_df.values, index=index, columns=logs_df.columns
        )  # .reset_index()
        df = df.astype(logs_df.dtypes)

        integer_index_levels = [1, 2, 3]
        for i in integer_index_levels:
            df.index = df.index.set_levels(df.index.levels[i].astype(int), level=i)
        dfs.append(df)
    return pd.concat(dfs).sort_index()
