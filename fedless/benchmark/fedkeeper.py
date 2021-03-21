import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disable tensorflow logs

import time
import uuid
import random
import asyncio
from pathlib import Path
from sys import getsizeof
from typing import Iterator, List, Optional, Dict

import click
import numpy as np
import pydantic
import pymongo
import urllib3
import pandas as pd
from pydantic import ValidationError
from tensorflow import keras

from fedless.benchmark.common import parse_yaml_file, run_in_executor
from fedless.models import (
    FunctionInvocationConfig,
    Hyperparams,
    MongodbConnectionConfig,
    FaaSProviderConfig,
    FunctionDeploymentConfig,
    ClientConfig,
    MNISTConfig,
    DatasetLoaderConfig,
    EvaluatorParams,
    EvaluatorResult,
    AggregatorFunctionResult,
    AggregatorFunctionParams,
    InvokerParams,
    SerializedParameters,
    WeightsSerializerConfig,
    NpzWeightsSerializerConfig,
)

# Model Definitions for Config files
from fedless.invocation import invoke_sync, retry_session
from fedless.persistence import (
    ClientConfigDao,
    ClientResultDao,
    ParameterDao,
    PersistenceError,
    ModelDao,
)
from fedless.providers import OpenwhiskCluster, FaaSProvider
from fedless.serialization import (
    NpzWeightsSerializer,
    Base64StringConverter,
    serialize_model,
)


class FedkeeperFunctions(pydantic.BaseModel):
    provider: str
    invoker: FunctionDeploymentConfig
    evaluator: FunctionDeploymentConfig
    aggregator: FunctionDeploymentConfig


class FedkeeperClientConfig(pydantic.BaseModel):
    function: FunctionInvocationConfig
    hyperparams: Optional[Hyperparams]
    replicas: int = 1


class FedkeeperClientsConfig(pydantic.BaseModel):
    functions: List[FedkeeperClientConfig]
    hyperparams: Optional[Hyperparams]


class ClusterConfig(pydantic.BaseModel):
    database: MongodbConnectionConfig
    clients: FedkeeperClientsConfig
    providers: Dict[str, FaaSProviderConfig]
    fedkeeper: FedkeeperFunctions


# Helper functions to create dataset shards / model


def create_mnist_train_data_loader_configs(
    n_devices: int, n_shards: int
) -> Iterator[List[int]]:
    if n_shards % n_devices != 0:
        raise ValueError(
            f"Can not equally distribute {n_shards} dataset shards among {n_devices} devices..."
        )

    (_, y_train), (_, _) = keras.datasets.mnist.load_data()
    num_train_examples, *_ = y_train.shape

    sorted_labels_idx = np.argsort(y_train)
    sorted_labels_idx_shards = np.split(sorted_labels_idx, n_shards)
    shards_per_device = len(sorted_labels_idx_shards) // n_devices
    np.random.shuffle(sorted_labels_idx_shards)

    for client_idx in range(n_devices):
        client_shards = sorted_labels_idx_shards[
            client_idx * shards_per_device : (client_idx + 1) * shards_per_device
        ]
        indices = np.concatenate(client_shards)
        yield indices.tolist()


def create_mnist_cnn(num_classes=10):
    model = keras.models.Sequential(
        [
            keras.layers.Input((28 * 28,)),
            keras.layers.Reshape((28, 28, 1)),
            keras.layers.Conv2D(
                32,
                kernel_size=(5, 5),
                activation="relu",
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(
                64,
                kernel_size=(5, 5),
                activation="relu",
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


async def get_deployment_manager(cluster_provider) -> FaaSProvider:
    if cluster_provider.type == "openwhisk":
        return OpenwhiskCluster(
            apihost=cluster_provider.params.apihost, auth=cluster_provider.params.auth
        )
    else:
        raise NotImplementedError(
            f"Deployment manager for {cluster_provider.type} not implemented!"
        )


class FedkeeperStrategy:
    def __init__(self, config: ClusterConfig, separate_invokers: bool = False):
        self.config = config
        self.separate_invokers = separate_invokers

        self.mongo_client = pymongo.MongoClient(
            host=config.database.host,
            port=config.database.port,
            username=config.database.username,
            password=config.database.password,
        )

        self.client_timing_infos = []

    def _init_clients(
        self,
        session_id: str,
    ):
        client_config_dao = ClientConfigDao(db=self.mongo_client)

        clients = self.config.clients
        default_hyperparams = clients.hyperparams
        n_clients = sum(function.replicas for function in clients.functions)
        client_idx_lists = create_mnist_train_data_loader_configs(
            n_devices=n_clients, n_shards=200
        )
        print(
            f"{n_clients} found in total. Generating dataset shards and client configurations..."
        )
        for client in clients.functions:
            for client_replica_idx in range(client.replicas):
                data_config = DatasetLoaderConfig(
                    type="mnist", params=MNISTConfig(indices=next(client_idx_lists))
                )
                client_hyperparams = client.hyperparams or default_hyperparams
                client_id = str(uuid.uuid4())
                client_config = ClientConfig(
                    session_id=session_id,
                    client_id=client_id,
                    function=client.function,
                    data=data_config,
                    hyperparams=client_hyperparams,
                )

                print(
                    f"Initializing client configurations with new id {client_id} of type "
                    f"{client.function.type} and {client.replicas} replicas"
                )
                client_config_dao.save(client_config)

    def _init_model(self, session_id: str):
        model = create_mnist_cnn()

        parameters_dao = ParameterDao(db=self.mongo_client)
        models_dao = ModelDao(db=self.mongo_client)

        weight_bytes = NpzWeightsSerializer().serialize(model.get_weights())
        weight_string = Base64StringConverter.to_str(weight_bytes)

        serialized_model = serialize_model(model)
        params = SerializedParameters(
            blob=weight_string,
            serializer=WeightsSerializerConfig(
                type="npz", params=NpzWeightsSerializerConfig()
            ),
        )
        print(
            f"Model loaded and successfully serialized. Total size is {getsizeof(weight_string) // 10 ** 6}MB. "
            f"Saving initial parameters to database"
        )

        parameters_dao.save(session_id=session_id, round_id=0, params=params)
        models_dao.save(session_id=session_id, model=serialized_model)

        print(f"Model successfully stored to database")

    async def deploy(self):
        providers = self.config.providers
        invoker = self.config.fedkeeper.invoker
        evaluator = self.config.fedkeeper.evaluator
        aggregator = self.config.fedkeeper.aggregator

        # Check if referenced provider actually exist
        if self.config.fedkeeper.provider not in providers.keys():
            raise KeyError(f"Provider {self.config.fedkeeper.provider} not specified")

        # Get deployment-manager
        cluster_provider: FaaSProviderConfig = providers[self.config.fedkeeper.provider]
        deployment_manager = await get_deployment_manager(cluster_provider)

        # Deploy or update evaluator and aggregator
        await asyncio.gather(
            deployment_manager.deploy(evaluator.params),
            deployment_manager.deploy(aggregator.params),
        )

        print("Successfully deployed evaluator and aggregator")

        # Deploy invoker(s)
        if not self.separate_invokers:
            await deployment_manager.deploy(invoker.params),
        else:
            n_clients = sum(
                function.replicas for function in self.config.clients.functions
            )
            print(
                f"{n_clients} client functions found in total, creating one invoker for each"
            )
            invoker_configs = []
            for i in range(n_clients):
                invoker_config = invoker.params.copy()
                invoker_config.name = f"{invoker_config.name}_{i}"
                invoker_configs.append(invoker_config)
            await asyncio.gather(
                *(deployment_manager.deploy(inv_conf) for inv_conf in invoker_configs)
            )
        print("Successfully deployed invoker function(s)")

    @run_in_executor
    def _call_invoker(self, params: InvokerParams, function: FunctionInvocationConfig):
        start_time = time.time()
        print(f"Client {params.client_id} invoked for round {params.round_id}")
        result = invoke_sync(
            function_config=function,
            data=params.dict(),
            session=retry_session(backoff_factor=1.0, retries=5),
        )
        print(f"Invoker received result from client {params.client_id}: {result}")
        self.client_timing_infos.append(
            {
                "client_id": params.client_id,
                "session_id": params.session_id,
                "round_id": params.round_id,
                "cardinality": result.get("cardinality", None),
                "seconds": time.time() - start_time,
            }
        )
        return result

    async def _invoke_clients(self, clients_in_round, round_id, session_id, invoker):
        print(f"Running round {round_id} with {len(clients_in_round)} clients")
        client_tasks = []
        for client in clients_in_round:
            invoker_params = InvokerParams(
                session_id=session_id,
                round_id=round_id,
                client_id=client.client_id,
                database=self.config.database,
            )

            async def g(params, invoker):
                return await self._call_invoker(params, invoker)

            task = asyncio.create_task(g(invoker_params, invoker))

            client_tasks.append(task)
        await asyncio.wait(client_tasks)
        return client_tasks

    async def run(
        self,
        clients_per_round: int,
        allowed_stragglers: int,
        out_dir: Path,
        session_id: Optional[str] = None,
        accuracy_threshold: float = 0.99,
    ):
        urllib3.disable_warnings()

        cluster_provider: FaaSProviderConfig = self.config.providers[
            self.config.fedkeeper.provider
        ]
        deployment_manager = await get_deployment_manager(cluster_provider)

        evaluator_function = await deployment_manager.to_invocation_config(
            self.config.fedkeeper.evaluator.params
        )
        aggregator_function = await deployment_manager.to_invocation_config(
            self.config.fedkeeper.aggregator.params
        )
        invoker_function = await deployment_manager.to_invocation_config(
            self.config.fedkeeper.invoker.params
        )

        client_config_dao = ClientConfigDao(db=self.mongo_client)
        client_result_dao = ClientResultDao(db=self.mongo_client)
        if not session_id:
            session_id = str(uuid.uuid4())

            print(f"Initializing new fedkeeper experiment with id {session_id}")

            self._init_clients(session_id)
            self._init_model(session_id)
            round_id = 0
        else:
            try:
                round_id = ParameterDao(db=self.mongo_client).get_latest_round(
                    session_id=session_id
                )
            except PersistenceError:
                round_id = 0

        print(
            f"Starting or resuming training for session {session_id} and round {round_id}"
        )

        client_configs = list(client_config_dao.load_all(session_id=session_id))
        print(f"Found {len(client_configs)} registered clients for this session")

        should_abort = False

        log_metrics = []

        while not should_abort:
            round_start_time = time.time()
            clients_in_round = random.sample(
                client_configs, min(clients_per_round, len(client_configs))
            )

            await self._invoke_clients(
                clients_in_round, round_id, session_id, invoker_function
            )
            clients_finished_time = time.time()
            n_successful = client_result_dao.count_results_for_round(
                session_id=session_id, round_id=round_id
            )
            if n_successful < (clients_per_round - allowed_stragglers):
                raise Exception(
                    f"Only {n_successful} clients finished this round. Stopping training"
                )
            else:
                print(
                    f"{n_successful}/{len(clients_in_round)} clients successfully finished in time this round."
                )

            print(f"Invoking aggregator")
            aggregator_params = AggregatorFunctionParams(
                session_id=session_id, round_id=round_id, database=self.config.database
            )
            aggregator_start_time = time.time()
            aggregator_result = invoke_sync(
                aggregator_function,
                data=aggregator_params.dict(),
                session=retry_session(backoff_factor=1.0, retries=5),
            )
            aggregator_end_time = time.time()
            try:
                aggregator_result: AggregatorFunctionResult = (
                    AggregatorFunctionResult.parse_obj(aggregator_result)
                )
            except ValidationError as e:
                raise Exception(f"Aggregator failed with {e}")

            print(
                f"Aggregator combined result of {aggregator_result.num_clients} clients. "
            )
            round_id = aggregator_result.new_round_id

            print(f"Invoking evaluation function...")
            evaluator_params = EvaluatorParams(
                session_id=session_id,
                round_id=round_id,
                database=self.config.database,
                test_data=DatasetLoaderConfig(
                    type="mnist", params=MNISTConfig(split="test")
                ),
            )
            evaluator_start_time = time.time()
            evaluator_result_dict = invoke_sync(
                evaluator_function,
                data=evaluator_params.dict(),
                session=retry_session(backoff_factor=1.0),
            )
            evaluator_end_time = time.time()
            evaluator_result = EvaluatorResult.parse_obj(evaluator_result_dict)
            global_accuracy = evaluator_result.metrics.metrics.get("accuracy", None)
            global_loss = evaluator_result.metrics.metrics.get("loss", None)

            log_metrics.append(
                {
                    "session_id": session_id,
                    "round_id": round_id,
                    "round_seconds": time.time() - round_start_time,
                    "clients_finished_seconds": (
                        clients_finished_time - round_start_time
                    ),
                    "aggregator_seconds": aggregator_end_time - aggregator_start_time,
                    "evaluator_seconds": evaluator_end_time - evaluator_start_time,
                    "num_clients_round": len(clients_in_round),
                    "global_test_accuracy": global_accuracy,
                    "global_test_loss": global_loss,
                }
            )
            pd.DataFrame.from_records(self.client_timing_infos).to_csv(
                out_dir / f"clients_{session_id}.csv"
            )
            pd.DataFrame.from_records(log_metrics).to_csv(
                out_dir / f"timing_{session_id}.csv"
            )

            if global_accuracy > accuracy_threshold:
                should_abort = True
                print(f"Reached target test accuracy after {round_id} rounds!")

            print(
                f"Evaluator returned {evaluator_result}"
                f"Global accuracy: {global_accuracy}"
            )
            f"Starting new round {aggregator_result.new_round_id}"


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config",
    help="cluster config file",
    type=click.Path(),
    required=True,
)
@click.option("--session-id", type=str)
@click.option("--clients-per-round", type=int, default=10)
@click.option("--allowed-stragglers", type=int, default=0)
@click.option("--accuracy-threshold", type=float, default=0.99)
@click.option("--log-dir", type=click.Path(), default=None)
@click.option("--separate-invokers", type=bool, default=False)
def run(
    config: str,
    session_id: str,
    clients_per_round: int,
    allowed_stragglers: int,
    accuracy_threshold: float = 0.99,
    log_dir: str = None,
    separate_invokers: bool = False,
):
    config_path = Path(config).parent
    config: ClusterConfig = parse_yaml_file(config, model=ClusterConfig)
    fedkeeper = FedkeeperStrategy(config=config, separate_invokers=separate_invokers)

    # Create log directory
    log_dir = Path(log_dir) if log_dir else config_path / "logs"
    log_dir.mkdir(exist_ok=True)

    # Run experiments
    asyncio.run(
        fedkeeper.run(
            clients_per_round=clients_per_round,
            allowed_stragglers=allowed_stragglers,
            accuracy_threshold=accuracy_threshold,
            out_dir=log_dir,
            session_id=session_id,
        )
    )


@cli.command()
@click.option(
    "--config",
    help="cluster config file",
    type=click.Path(),
    required=True,
)
@click.option("--separate-invokers", type=bool, default=False)
def deploy(config: str, separate_invokers: bool):
    config: ClusterConfig = parse_yaml_file(config, model=ClusterConfig)

    fedkeeper = FedkeeperStrategy(config=config, separate_invokers=separate_invokers)
    asyncio.run(fedkeeper.deploy())


if __name__ == "__main__":
    cli()
