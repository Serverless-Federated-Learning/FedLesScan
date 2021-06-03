from typing import Optional

import click
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import dateutil.parser as dp
import os
from pathlib import Path


@click.command()
@click.option("-f", "--file", type=Path, required=True)
@click.option("-o", "--out", type=Path, required=False)
def run(file: Path, out: Optional[Path]):
    print(f"Processing file {file.resolve()}")
    if not out:
        out = file.parent
    log_text = file.read_text()
    lines = log_text.splitlines()

    # centralized evaluation (mnist)
    records = []
    if not "losses_distributed" in log_text:
        fit_progress_lines = (line for line in lines if "fit progress: (" in line)
        fit_progress_tuples = list(
            eval(line.split("fit progress: ")[-1]) for line in fit_progress_lines
        )

        for entry in fit_progress_tuples:
            v = []
            for e in entry:
                if isinstance(e, dict):
                    v.extend(e.values())
                else:
                    v.append(e)
            records.append(tuple(v))
    else:
        loss_tuples = list(
            eval(line.split("losses_distributed")[-1])
            for line in lines
            if "losses_distributed" in line
        )[0]
        accuracy_tuples = list(
            eval(line.split("metrics_distributed")[-1])["accuracy"]
            for line in lines
            if "metrics_distributed" in line
        )[0]
        new_round_times = list(
            dp.parse(line.split(" ")[3]).timestamp()
            for line in lines
            if "fit_round: strategy sampled" in line
        )
        last_time_entry = list(
            dp.parse(line.split(" ")[3]).timestamp()
            for line in lines
            if "FL finished in" in line
        )[0]
        start_time = new_round_times[0]
        new_round_times = [*new_round_times[1:], last_time_entry]
        assert len(accuracy_tuples) == len(loss_tuples) == len(new_round_times)
        for loss_tuple, accuracy_tuple, round_time in zip(
            loss_tuples, accuracy_tuples, new_round_times
        ):
            assert loss_tuple[0] == accuracy_tuple[0]  # Same round
            records.append(
                (
                    loss_tuple[0],
                    loss_tuple[1],
                    accuracy_tuple[1],
                    round_time - start_time,
                )
            )
    # Create dataframe
    df = pd.DataFrame.from_records(
        records, columns=["round", "loss", "accuracy", "time"]
    )
    df["round_duration"] = [df["time"].iloc[0], *df["time"].diff()[1:]]

    # PLOT TIME
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x="round", y="accuracy", ax=ax, data=df)
    fig.savefig(out / "accuracy.pdf")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x="round", y="round_duration", ax=ax, data=df)
    fig.savefig(out / "timings.pdf")


if __name__ == "__main__":
    run()
