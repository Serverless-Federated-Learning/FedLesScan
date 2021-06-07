import asyncio
import logging
import sys
import uuid
from itertools import cycle
from pathlib import Path
from typing import List, Union, Tuple

import click

import tensorflow as tf

from fedless.providers import OpenwhiskCluster
from fedless.benchmark.common import parse_yaml_file
from fedless.benchmark.fedkeeper import (
    create_mnist_cnn,
    create_mnist_train_data_loader_configs,
)
from fedless.benchmark.models import (
    ClusterConfig,
    ExperimentConfig,
    FedkeeperClientConfig,
    FedkeeperClientsConfig,
)
from fedless.data import LEAF
from fedless.benchmark.leaf import create_femnist_cnn, create_shakespeare_lstm
from fedless.models import (
    LeafDataset,
    ClientConfig,
    MongodbConnectionConfig,
    DatasetLoaderConfig,
    BinaryStringFormat,
    SerializedParameters,
    WeightsSerializerConfig,
    NpzWeightsSerializerConfig,
    LEAFConfig,
    MNISTConfig,
)
from fedless.benchmark.strategy import FedkeeperStrategy
from fedless.persistence import ClientConfigDao, ParameterDao, ModelDao
from fedless.serialization import (
    serialize_model,
    NpzWeightsSerializer,
    Base64StringConverter,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILE_SERVER = "http://138.246.235.163:31715"


def create_model(dataset) -> tf.keras.Sequential:
    if dataset.lower() == "femnist":
        return create_femnist_cnn()
    elif dataset.lower() == "shakespeare":
        return create_shakespeare_lstm()
    elif dataset.lower() == "mnist":
        return create_mnist_cnn()
    else:
        raise NotImplementedError()


def create_mnist_test_config() -> DatasetLoaderConfig:
    return DatasetLoaderConfig(type="mnist", params=MNISTConfig(split="test"))


# noinspection PydanticTypeChecker,PyTypeChecker
def create_data_configs(
    dataset: str, clients: int
) -> List[Union[DatasetLoaderConfig, Tuple[DatasetLoaderConfig, DatasetLoaderConfig]]]:
    dataset = dataset.lower()
    if dataset == "mnist":
        return list(
            create_mnist_train_data_loader_configs(n_devices=clients, n_shards=600)
        )
    elif dataset in ["femnist", "shakespeare"]:
        configs = []
        for client_idx in range(clients):
            train = DatasetLoaderConfig(
                type="leaf",
                params=LEAFConfig(
                    dataset=dataset,
                    location=f"{FILE_SERVER}/data/leaf/data/{dataset}/data/"
                    f"train/user_{client_idx}_train_9.json",
                ),
            )
            test = DatasetLoaderConfig(
                type="leaf",
                params=LEAFConfig(
                    dataset=dataset,
                    location=f"{FILE_SERVER}/data/leaf/data/{dataset}/data/"
                    f"test/user_{client_idx}_test_9.json",
                ),
            )
            configs.append((train, test))
        return configs
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")


@click.command()
@click.option(
    "-d",
    "--dataset",
    type=click.Choice(["mnist", "femnist", "shakespeare"], case_sensitive=False),
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
    type=click.Choice(["fedkeeper", "fedless"], case_sensitive=False),
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
):
    session = str(uuid.uuid4())
    config_path = Path(config).parent
    config: ExperimentConfig = parse_yaml_file(config, model=ExperimentConfig)

    model = create_model(dataset)
    data_configs = create_data_configs(dataset, clients)

    clients = store_client_configs(
        session=session,
        clients=config.clients,
        num_clients=clients,
        data_configs=data_configs,
        database=config.database,
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
    strategy = FedkeeperStrategy(
        session=session,
        provider=cluster,
        clients=clients,
        invoker_config=config.server.invoker,
        evaluator_config=config.server.evaluator,
        aggregator_config=config.server.aggregator,
        mongodb_config=config.database,
        allowed_stragglers=stragglers,
        client_timeout=timeout,
        global_test_data=(
            create_mnist_test_config() if dataset.lower() == "mnist" else None
        ),
        use_separate_invokers=separate_invokers,
    )

    asyncio.run(strategy.deploy_all_functions())
    asyncio.run(
        strategy.fit(
            n_clients_in_round=clients_in_round,
            max_rounds=rounds,
            max_accuracy=max_accuracy,
        )
    )


def init_store_model(
    session: str,
    model: tf.keras.Sequential,
    database: MongodbConnectionConfig,
    store_json_serializable: bool = False,
):
    parameters_dao = ParameterDao(db=database)
    models_dao = ModelDao(db=database)

    serialized_model = serialize_model(model)
    weights = model.get_weights()
    weights_serialized = NpzWeightsSerializer(compressed=False).serialize(weights)
    weights_format = BinaryStringFormat.NONE
    if store_json_serializable:
        weights_serialized = Base64StringConverter.to_str(weights_serialized)
        weights_format = BinaryStringFormat.BASE64
    params = SerializedParameters(
        blob=weights_serialized,
        serializer=WeightsSerializerConfig(
            type="npz", params=NpzWeightsSerializerConfig(compressed=False)
        ),
        string_format=weights_format,
    )
    logger.debug(
        f"Model loaded and successfully serialized. Total size is {sys.getsizeof(weights_serialized) // 10 ** 6}MB. "
        f"Saving initial parameters to database"
    )
    parameters_dao.save(session_id=session, round_id=0, params=params)
    models_dao.save(session_id=session, model=serialized_model)


def store_client_configs(
    session: str,
    clients: FedkeeperClientsConfig,
    num_clients: int,
    data_configs: List[
        Union[DatasetLoaderConfig, Tuple[DatasetLoaderConfig, DatasetLoaderConfig]]
    ],
    database: MongodbConnectionConfig,
) -> List[ClientConfig]:
    client_config_dao = ClientConfigDao(database)

    n_clients = sum(function.replicas for function in clients.functions)

    logger.info(
        f"{len(data_configs)} data configurations given with the "
        f"instruction to setup {num_clients} clients from {n_clients} potential endpoints."
    )

    data_shards = iter(data_configs)
    function_iter = cycle(clients.functions)
    default_hyperparms = clients.hyperparams
    final_configs = []
    for shard in data_shards:
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

        logger.info(
            f"Initializing client {client_id} of type " f"{client.function.type}"
        )
        client_config_dao.save(client_config)
        final_configs.append(client_config)
    logger.info(
        f"Configured and stored all {len(data_configs)} clients configurations..."
    )
    return final_configs


if __name__ == "__main__":
    run()
