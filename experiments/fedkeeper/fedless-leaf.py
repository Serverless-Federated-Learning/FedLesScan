#!/usr/bin/env python3
import asyncio
from itertools import zip_longest
from pathlib import Path

import click

from fedless.benchmark.common import parse_yaml_file
from fedless.benchmark.fedkeeper_indep import FedlessStrategy, ClusterConfig
from fedless.benchmark.leaf import create_shakespeare_lstm, create_femnist_cnn
from fedless.models import DatasetLoaderConfig, LEAFConfig


@click.command()
@click.option("--dataset", type=str, required=True)
@click.option(
    "--config",
    help="cluster config file",
    type=click.Path(),
    required=True,
)
@click.option("--n-clients", type=int, default=100)
@click.option("--clients-per-round", type=int, default=10)
@click.option("--allowed-stragglers", type=int, default=0)
@click.option("--accuracy-threshold", type=float, default=0.99)
@click.option("--log-dir", type=click.Path(), default=None)
@click.option("--aggregate-online/--no-aggregate-online", type=bool, default=False)
def run(
    dataset: str,
    config: str,
    n_clients: int,
    clients_per_round: int,
    allowed_stragglers: int,
    accuracy_threshold: float,
    log_dir: str,
    aggregate_online: bool,
):
    config_path = Path(config).parent
    config: ClusterConfig = parse_yaml_file(config, model=ClusterConfig)
    dataset = dataset.lower()
    if dataset == "femnist":
        model = create_femnist_cnn()
    elif dataset == "shakespeare":
        model = create_shakespeare_lstm()
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")
    client_train_data_configs = [
        DatasetLoaderConfig(
            type="leaf",
            params=LEAFConfig(
                dataset=dataset,
                location=f"http://138.246.235.163:31715/data/leaf/data/{dataset}/data/train/user_{i}_train_9.json",
            ),
        )
        for i in range(n_clients)
    ]
    client_test_data_configs = [
        DatasetLoaderConfig(
            type="leaf",
            params=LEAFConfig(
                dataset=dataset,
                location=f"http://138.246.235.163:31715/data/leaf/data/{dataset}/data/test/user_{i}_test_9.json",
            ),
        )
        for i in range(n_clients)
    ]

    client_data_configs = list(
        zip_longest(client_train_data_configs, client_test_data_configs)
    )

    # Create log directory
    log_dir = (
        Path(log_dir)
        if log_dir
        else config_path
        / f"logs_fedless_{dataset}_{n_clients}_{clients_per_round}_{allowed_stragglers}_{accuracy_threshold}"
    )
    log_dir.mkdir(exist_ok=True)

    fedkeeper = FedlessStrategy(
        config=config,
        model=model,
        client_data_configs=client_data_configs,
        aggregate_online=aggregate_online,
    )

    asyncio.run(
        fedkeeper.run(
            clients_per_round=clients_per_round,
            allowed_stragglers=allowed_stragglers,
            accuracy_threshold=accuracy_threshold,
            out_dir=log_dir,
        )
    )


if __name__ == "__main__":
    run()
