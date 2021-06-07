#!/usr/bin/env python3
import asyncio
from pathlib import Path

import click

from fedless.benchmark.common import parse_yaml_file
from fedless.models import DatasetLoaderConfig, MNISTConfig

from fedless.benchmark.fedkeeper import (
    FedkeeperStrategy,
    create_mnist_cnn,
    create_mnist_train_data_loader_configs,
)
from benchmark.models import ClusterConfig


@click.command()
@click.option(
    "--config",
    help="cluster config file",
    type=click.Path(),
    required=True,
)
@click.option("--n-clients", type=int, default=100)
@click.option("--clients-per-round", type=int, default=25)
@click.option("--allowed-stragglers", type=int, default=5)
@click.option("--accuracy-threshold", type=float, default=0.99)
@click.option("--log-dir", type=click.Path(), default=None)
def run(
    config: str,
    n_clients: int,
    clients_per_round: int,
    allowed_stragglers: int,
    accuracy_threshold: float,
    log_dir: str,
):
    config_path = Path(config).parent
    config: ClusterConfig = parse_yaml_file(config, model=ClusterConfig)

    model = create_mnist_cnn()
    client_data_configs = list(
        create_mnist_train_data_loader_configs(
            n_devices=n_clients, n_shards=n_clients * 2
        )
    )

    # Create log directory
    log_dir = Path(log_dir) if log_dir else config_path / "logs"
    log_dir.mkdir(exist_ok=True)

    fedkeeper = FedkeeperStrategy(
        config=config,
        model=model,
        client_data_configs=client_data_configs,
        test_data=DatasetLoaderConfig(type="mnist", params=MNISTConfig(split="test")),
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
