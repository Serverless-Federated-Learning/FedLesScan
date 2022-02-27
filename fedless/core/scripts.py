import asyncio
import logging
import random
import uuid
from itertools import cycle
from pathlib import Path
from typing import List, Union, Tuple, Dict

import click
import numpy as np
import tensorflow as tf

from fedless.core.common import parse_yaml_file
from fedless.core.models import (
    CognitoConfig,
    ExperimentConfig,
    FedkeeperClientsConfig,
)
from fedless.datasets.benchmark_configurator import (
    create_model,
    init_store_model,
    create_mnist_test_config,
    create_data_configs,
)
from fedless.strategies.Intelligent_selection import (
    DBScanClientSelection,
    RandomClientSelection,
)
from fedless.strategies.strategy_selector import select_strategy
from fedless.models import (
    ClientConfig,
    ClientPersistentHistory,
    MongodbConnectionConfig,
    DatasetLoaderConfig,
)
from fedless.persistence.client_daos import (
    ClientConfigDao,
    ClientHistoryDao,
)
from fedless.providers import OpenwhiskCluster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-d",
    "--dataset",
    type=click.Choice(
        ["mnist", "femnist", "shakespeare", "speech"], case_sensitive=False
    ),
    required=True,
    # help='Evaluation dataset. One of ("mnist", "femnist", "shakespeare")',
)
@click.option(
    "-c",
    "--config",
    help="Config file with faas platform and client function information",
    type=click.Path(),
    required=True,
)
@click.option(
    "-s",
    "--strategy",
    type=click.Choice(["fedless", "fedless_enhanced"], case_sensitive=False),
    required=True,
)
@click.option(
    "--clients",
    type=int,
    help="number of clients",
    required=True,
)
@click.option(
    "--clients-in-round",
    type=int,
    help="number of clients sampled per round",
    required=True,
)
@click.option(
    "--stragglers",
    type=int,
    help="number of allowed stragglers per round",
    default=0,
)
@click.option(
    "--timeout",
    type=float,
    help="maximum wait time for functions to finish",
    default=300,
)
@click.option(
    "--rounds",
    type=int,
    help="maximum wait time for functions to finish",
    default=100,
)
@click.option(
    "--separate-invokers/--no-separate-invokers",
    help="use separate invoker function for each client (only applies when fedkeeper strategy is used)",
    default=True,
)
@click.option(
    "--max-accuracy",
    help="stop training if this test accuracy is reached",
    type=float,
    default=0.99,
)
@click.option(
    "-o",
    "--out",
    help="directory where logs will be stored",
    type=click.Path(),
    required=True,
)
@click.option(
    "--tum-proxy/--no-tum-proxy",
    help="use in.tum.de proxy",
    default=False,
)
@click.option(
    "--proxy-in-evaluator/--no-proxy-in-evaluator",
    help="use proxy also in evaluation function",
    default=False,
)
@click.option(
    "--aggregate-online/--aggregate-offline",
    help="use in.tum.de proxy",
    default=False,
)
@click.option(
    "--test-batch-size",
    type=int,
    default=10,
)
# @click.option(
#     "--invocation-delay",
#     type=float,
#     default=None,
# )
@click.option(
    "--mock/--no-mock",
    help="use mocks",
    default=False,
)
@click.option(
    "--simulate-stragglers",
    help="define a percentage of the clients to straggle",
    type=float,
    default=0.0,
)
def run(
    dataset: str,
    config: str,
    strategy: str,
    clients: int,
    clients_in_round: int,
    stragglers: int,
    timeout: float,
    rounds: int,
    separate_invokers: bool,
    max_accuracy: float,
    out: str,
    tum_proxy: bool,
    proxy_in_evaluator: bool,
    aggregate_online: bool,
    test_batch_size: int,
    # invocation_delay: float,
    mock: bool,
    simulate_stragglers: float,
):
    session = str(uuid.uuid4())
    log_dir = Path(out) if out else Path(config).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    config: ExperimentConfig = parse_yaml_file(config, model=ExperimentConfig)
    with (log_dir / f"config_{session}.json").open("w+") as f:
        f.write(config.json())

    # Configure proxy if specified
    proxies = (
        {
            "https": "http://proxy.in.tum.de:8080/",
            "http": "http://proxy.in.tum.de:8080/",
            "https://138.246.233.81": "",
            "http://138.246.233.81": "",
            "http://138.246.233.217": "",
            "https://138.246.233.217": "",
            "https://127.0.0.1": "",
            "http://127.0.0.1": "",
            "https://localhost": "",
            "http://localhost": "",
        }
        if tum_proxy
        else None
    )

    model = create_model(dataset)
    data_configs = create_data_configs(dataset, clients)  # , proxies=proxies)

    clients = store_client_configs(
        session=session,
        clients=config.clients,
        num_clients=clients,
        data_configs=data_configs,
        database=config.database,
        stragglers_precentage=simulate_stragglers,
    )
    init_store_model(
        session=session,
        model=model,
        database=config.database,
        store_json_serializable=(strategy == "fedkeeper"),
    )

    cluster = OpenwhiskCluster(
        apihost=config.cluster.apihost,
        auth=config.cluster.auth,
        insecure=config.cluster.insecure,
        namespace=config.cluster.namespace,
        package=config.cluster.package,
    )

    inv_params = {
        "session": session,
        "cognito": config.cognito,
        "provider": cluster,
        "clients": clients,
        "evaluator_config": config.server.evaluator,
        "aggregator_config": config.server.aggregator,
        "mongodb_config": config.database,
        "allowed_stragglers": stragglers,
        "client_timeout": timeout,
        "save_dir": log_dir,
        "aggregator_params": {
            "online": aggregate_online,
            "test_batch_size": test_batch_size,
        },
        "global_test_data": (
            create_mnist_test_config(proxies=(proxies if proxy_in_evaluator else None))
            if dataset.lower() == "mnist"
            else None
        ),
        "proxies": proxies,
        "mock": mock,
    }

    strategy = select_strategy(strategy, inv_params)

    asyncio.run(strategy.deploy_all_functions())
    asyncio.run(
        strategy.fit(
            n_clients_in_round=clients_in_round,
            max_rounds=rounds,
            max_accuracy=max_accuracy,
        )
    )


def store_client_configs(
    session: str,
    clients: FedkeeperClientsConfig,
    num_clients: int,
    data_configs: List[
        Union[DatasetLoaderConfig, Tuple[DatasetLoaderConfig, DatasetLoaderConfig]]
    ],
    database: MongodbConnectionConfig,
    stragglers_precentage: float,
) -> List[ClientConfig]:
    client_config_dao = ClientConfigDao(database)
    client_history_dao = ClientHistoryDao(database)
    n_clients = sum(function.replicas for function in clients.functions)
    clients_unrolled = list(f for f in clients.functions for _ in range(f.replicas))
    logger.info(
        f"{len(data_configs)} data configurations given with the "
        f"instruction to setup {num_clients} clients from {n_clients} potential endpoints."
    )
    # todo add delay param for all clients
    stragglers_delay_list = [-1, 80]

    num_stragglers = int(stragglers_precentage * num_clients)
    logger.info(f"simulate stragglers {num_stragglers} clients for {num_clients}.")
    data_shards = iter(data_configs)
    function_iter = cycle(clients_unrolled)
    default_hyperparms = clients.hyperparams
    final_configs = []
    stragglers_idx_list = random.sample(list(np.arange(num_clients)), num_stragglers)
    for idx, shard in enumerate(data_shards):
        client = next(function_iter)
        hp = client.hyperparams or default_hyperparms
        client_id = str(uuid.uuid4())
        train_config, test_config = shard if isinstance(shard, tuple) else (shard, None)
        client_config = ClientConfig(
            session_id=session,
            client_id=client_id,
            function=client.function,
            data=train_config,
            test_data=test_config,
            hyperparams=hp,
        )
        # add straggler
        if idx in stragglers_idx_list:
            client_config.function.invocation_delay = random.sample(
                stragglers_delay_list, 1
            )[0]

        client_history = ClientPersistentHistory(
            client_id=client_id,
            session_id=session,
        )

        logger.info(
            f"Initializing client {client_id} of type " f"{client.function.type}"
        )
        client_config_dao.save(client_config)
        logger.info(
            f"Initializing client_history for {client_id} of type "
            f"{client.function.type}"
        )
        client_history_dao.save(client_history)
        final_configs.append(client_config)
    logger.info(
        f"Configured and stored all {len(data_configs)} clients configurations..."
    )
    return final_configs


if __name__ == "__main__":
    run()
