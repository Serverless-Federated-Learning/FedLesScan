from typing import Optional

import click
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path


@click.command()
@click.option("-f", "--file", type=Path, required=True)
@click.option("-o", "--out", type=Path, required=False)
def run(file: Path, out: Optional[Path]):
    print(f"Processing file {file.resolve()}")
    if not out:
        out = file.parent
    log_text = file.read_text().splitlines()
    fit_progress_lines = (line for line in log_text if "fit progress: (" in line)
    fit_progress_tuples = list(
        eval(line.split("fit progress: ")[-1]) for line in fit_progress_lines
    )
    records = []
    for entry in fit_progress_tuples:
        v = []
        for e in entry:
            if isinstance(e, dict):
                v.extend(e.values())
            else:
                v.append(e)

        records.append(tuple(v))
    # Create dataframe
    df = pd.DataFrame.from_records(
        records, columns=["round", "loss", "accuracy", "time"]
    )
    df["round_duration"] = df["time"].diff()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x="round", y="accuracy", ax=ax, data=df)
    fig.savefig(out / "accuracy.pdf")


if __name__ == "__main__":
    run()
