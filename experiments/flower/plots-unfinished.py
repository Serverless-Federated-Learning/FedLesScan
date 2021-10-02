from typing import Optional, List

import re
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
import pandas as pd
import dateutil.parser as dp
import os
from pathlib import Path


@click.command()
@click.option("files", "-f", "--file", type=Path, required=True, multiple=True)
@click.option("-o", "--out", type=Path, required=False)
def run(files: List[Path], out: Optional[Path]):
    process_lines = []
    eval_lines = []
    for file in files:
        print(f"Processing file {file.resolve()}")
        log_text = file.read_text()
        lines = log_text.splitlines()
        if ".err" in file.name:
            process_lines = lines
        else:
            eval_lines = lines
        if not out:
            out = file.parent

    values = []
    for line in eval_lines:
        round, *rest = line.split(" ")
        rest = " ".join(rest)
        # m = re.findall(r"EvaluateRes\(.+\)", rest)
        m = re.findall(r"'accuracy': ([-+]?\d*\.\d+|\d+)", rest)
        accuracies = [float(v) for v in m]

        m = re.findall(r"loss=([-+]?\d*\.\d+|\d+)", rest)
        losses = [float(v) for v in m]

        m = re.findall(r"num_examples=(\d+)", rest)
        num_examples = [int(v) for v in m]
        for loss, acc, card in zip(losses, accuracies, num_examples):
            values.append(
                {
                    "round": round,
                    "accuracy": acc,
                    "loss": loss,
                    "num_examples": card,
                    "weighted_average": np.average(accuracies, weights=num_examples),
                }
            )

    timing_infos = []
    if process_lines:
        new_round_times = list(
            dp.parse(" ".join(line.split(" ")[2:4])).timestamp()
            for line in process_lines
            if "fit_round: strategy sampled" in line
        )
        # last_time_entry = list(
        #    dp.parse(line.split(" ")[3]).timestamp()
        #    for line in process_lines
        #    if "FL finished in" in line
        # )[0]
        print(new_round_times)
        start_time = new_round_times[0]
        new_round_times = [*new_round_times[1:]]
        round_durations = [(t - start_time) for t in new_round_times]
        print(start_time)
        # print(round_durations)
        print(len(round_durations), len(eval_lines))
        for i, dur in enumerate(round_durations):
            timing_infos.append({"round": i, "time": dur})

        timing_df = pandas.DataFrame.from_records(timing_infos)
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        timing_df["round_duration"] = [
            timing_df["time"].iloc[0],
            *timing_df["time"].diff()[1:],
        ]
        timing_df["round_duration_min"] = timing_df["round_duration"] / 60
        sns.barplot(x="round", y="round_duration_min", ax=ax, data=timing_df)
        fig.savefig(out / "tim.pdf")

    df = pd.DataFrame.from_records(values)
    # PLOT TIME
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x="round", y="accuracy", ax=ax, data=df)
    sns.lineplot(x="round", y="weighted_average", ax=ax, data=df)
    fig.savefig(out / "acc.pdf")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x="round", y="loss", ax=ax, data=df)
    fig.savefig(out / "loss.pdf")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x="round", y="num_examples", ax=ax, data=df)
    fig.savefig(out / "num_examples.pdf")

    # fig, ax = plt.subplots(figsize=(10, 5))
    # sns.lineplot(x="round", y="loss", ax=ax, data=df)
    # fig.savefig(out / "loss.pdf")


# TODO
# fig, ax = plt.subplots(figsize=(10, 5))
# sns.barplot(x="round", y="round_duration", ax=ax, data=df)
# fig.savefig(out / "timings.pdf")
# centralized evaluation (mnist)
# records = []
# loss_tuples = list(
#    eval(line.split("losses_distributed")[-1])
#    for line in lines
#    if "losses_distributed" in line
# )[0]

## Create dataframe
# df = pd.DataFrame.from_records(
#    records, columns=["round", "loss", "accuracy", "time"]
# )
# df["round_duration"] = [df["time"].iloc[0], *df["time"].diff()[1:]]


#


if __name__ == "__main__":
    run()
