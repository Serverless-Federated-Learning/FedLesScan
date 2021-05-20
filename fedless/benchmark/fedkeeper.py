import os

import yaml
from requests import Session

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disable tensorflow logs

import time
import uuid
import random
import asyncio
from pathlib import Path
from sys import getsizeof
from typing import Iterator, List, Optional, Dict, Union, Tuple

import click
import numpy as np
import pydantic
import pymongo
import urllib3
import pandas as pd
import tensorflow as tf
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
    DatasetLoaderConfig,
    EvaluatorParams,
    EvaluatorResult,
    AggregatorFunctionResult,
    AggregatorFunctionParams,
    InvokerParams,
    SerializedParameters,
    WeightsSerializerConfig,
    NpzWeightsSerializerConfig,
    MNISTConfig,
    Parameters,
    TestMetrics,
)

# Model Definitions for Config files
from fedless.invocation import invoke_sync, retry_session, InvocationTimeOut
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
from fedless.data import DatasetLoaderBuilder
from fedless.auth import CognitoClient


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


class CognitoConfig(pydantic.BaseModel):
    user_pool_id: str
    region_name: str
    auth_endpoint: str
    invoker_client_id: str
    invoker_client_secret: str
    required_scopes: List[str] = ["client-functions/invoke"]


class ClusterConfig(pydantic.BaseModel):
    database: MongodbConnectionConfig
    clients: FedkeeperClientsConfig
    providers: Dict[str, FaaSProviderConfig]
    fedkeeper: FedkeeperFunctions
    cognito: Optional[CognitoConfig]


# Helper functions to create dataset shards / model


def create_mnist_train_data_loader_configs(
    n_devices: int, n_shards: int
) -> Iterator[DatasetLoaderConfig]:
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
        # noinspection PydanticTypeChecker,PyTypeChecker
        yield DatasetLoaderConfig(
            type="mnist", params=MNISTConfig(indices=indices.tolist())
        )


def create_mnist_cnn(num_classes=10):
    model = keras.models.Sequential(
        [
            keras.layers.Input((28, 28)),
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


class FederatedLearningStrategy:
    def __init__(
        self,
        model: keras.Model,
        client_data_configs: List[
            Union[DatasetLoaderConfig, Tuple[DatasetLoaderConfig, DatasetLoaderConfig]]
        ],
        test_data_config: Optional[DatasetLoaderConfig] = None,
    ):
        self.model = model
        self.test_data_config = test_data_config
        self._test_data: tf.data.Dataset
        self.client_data_configs = client_data_configs

    @property
    def cached_test_data(self) -> tf.data.Dataset:
        if self.test_data_config is None:
            raise ValueError(f"No global test dataset provided...")
        if not self._test_data:
            self._test_data = DatasetLoaderBuilder.from_config(
                self.test_data_config
            ).load()
        return self._test_data

    def evaluate_global(self, parameters: Parameters, batch_size: int = 32) -> Dict:
        self.model.set_weights(parameters)
        eval_dict = self.model.evaluate(
            self.cached_test_data.batch(batch_size), return_dict=True
        )
        return eval_dict

    def evaluate_clients(
        self, metrics: Iterator[TestMetrics], metric_names: List[str] = None
    ) -> Dict:

        if metric_names is None:
            metric_names = ["loss"]

        cardinalities, metrics = zip(
            *((metric.cardinality, metric.metrics) for metric in metrics)
        )
        result_dict = {}
        for metric_name in metric_names:
            values = [metric[metric_name] for metric in metrics]
            mean = np.average(values, weights=cardinalities)
            result_dict.update(
                {
                    f"mean_{metric_name}": mean,
                    f"all_{metric_name}": values,
                    f"median_{metric_name}": np.median(values),
                }
            )

        return result_dict


class FedkeeperStrategy(FederatedLearningStrategy):
    def __init__(
        self,
        model: keras.Model,
        client_data_configs: List[
            Union[DatasetLoaderConfig, Tuple[DatasetLoaderConfig, DatasetLoaderConfig]]
        ],
        config: ClusterConfig,
        test_data: Optional[DatasetLoaderConfig] = None,
    ):
        super(FedkeeperStrategy, self).__init__(
            model=model,
            client_data_configs=client_data_configs,
            test_data_config=test_data,
        )

        self.config = config
        self.evaluator_function: FunctionInvocationConfig = None
        self.aggregator_function: FunctionInvocationConfig = None
        self.invoker_function: FunctionInvocationConfig = None

        self.mongo_client = pymongo.MongoClient(
            host=config.database.host,
            port=config.database.port,
            username=config.database.username,
            password=config.database.password,
        )
        self.client_timing_infos = []
        self.cognito_auth_token: str = None

    def fetch_cognito_auth_token(self) -> str:
        if not self.config.cognito:
            raise ValueError(f"No cognito configuration given")
        cognito = CognitoClient(
            user_pool_id=self.config.cognito.user_pool_id,
            region_name=self.config.cognito.region_name,
        )
        self.cognito_auth_token = cognito.fetch_token_for_client(
            auth_endpoint=self.config.cognito.auth_endpoint,
            client_id=self.config.cognito.invoker_client_id,
            client_secret=self.config.cognito.invoker_client_secret,
            required_scopes=self.config.cognito.required_scopes,
        )
        return self.cognito_auth_token

    def _init_clients(
        self,
        session_id: str,
    ):
        client_config_dao = ClientConfigDao(db=self.mongo_client)

        clients = self.config.clients
        default_hyperparams = clients.hyperparams
        n_clients = sum(function.replicas for function in clients.functions)
        if n_clients != len(self.client_data_configs):
            raise ValueError(
                f"Found {n_clients} client functions but {len(self.client_data_configs)} "
                f"client data configs. Numbers must match"
            )
        print(
            f"{n_clients} found in total. Generating dataset shards and client configurations..."
        )
        client_data_config_iterator = iter(self.client_data_configs)
        for client in clients.functions:
            for client_replica_idx in range(client.replicas):
                client_hyperparams = client.hyperparams or default_hyperparams
                client_id = str(uuid.uuid4())
                data_config = next(client_data_config_iterator)
                test_config = None
                if isinstance(data_config, tuple):
                    data_config, test_config = data_config
                client_config = ClientConfig(
                    session_id=session_id,
                    client_id=client_id,
                    function=client.function,
                    data=data_config,
                    test_data=test_config,
                    hyperparams=client_hyperparams,
                )

                print(
                    f"Initializing client configurations with new id {client_id} of type "
                    f"{client.function.type} and {client.replicas} replicas"
                )
                client_config_dao.save(client_config)

    def _init_model(self, session_id: str):
        parameters_dao = ParameterDao(db=self.mongo_client)
        models_dao = ModelDao(db=self.mongo_client)

        weight_bytes = NpzWeightsSerializer().serialize(self.model.get_weights())
        weight_string = Base64StringConverter.to_str(weight_bytes)

        serialized_model = serialize_model(self.model)
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

        # Deploy invoker
        await deployment_manager.deploy(invoker.params),

        cluster_provider: FaaSProviderConfig = self.config.providers[
            self.config.fedkeeper.provider
        ]
        deployment_manager = await get_deployment_manager(cluster_provider)

        self.evaluator_function = await deployment_manager.to_invocation_config(
            self.config.fedkeeper.evaluator.params
        )
        self.aggregator_function = await deployment_manager.to_invocation_config(
            self.config.fedkeeper.aggregator.params
        )
        self.invoker_function = await deployment_manager.to_invocation_config(
            self.config.fedkeeper.invoker.params
        )
        print("Successfully deployed invoker function(s)")

    @run_in_executor
    def _call_invoker(
        self,
        params: InvokerParams,
        function: FunctionInvocationConfig,
        session: Optional[Session] = None,
    ):
        start_time = time.time()
        print(f"Client {params.client_id} invoked for round {params.round_id}")
        result = invoke_sync(
            function_config=function,
            data=params.dict(),
            session=retry_session(backoff_factor=1.0, retries=5, session=session),
            timeout=500,
        )
        print(f"Invoker received result from client {params.client_id}: {result}")
        self.client_timing_infos.append(
            {
                "client_id": params.client_id,
                "session_id": params.session_id,
                "seconds": time.time() - start_time,
                "config": function.json(),
                "round_id": params.round_id,
                "cardinality": result.get("cardinality", None),
                "privacy_guarantees": result.get("privacy_guarantees", None),
                "history": result.get("history:", None),
                "test_metrics": result.get("test_metrics", None),
            }
        )
        return result

    async def _invoke_clients(self, clients_in_round, round_id, session_id):
        print(f"Running round {round_id} with {len(clients_in_round)} clients")
        client_tasks = []
        http_headers = (
            {"Authorization": f"Bearer {self.fetch_cognito_auth_token()}"}
            if self.config.cognito
            else {}
        )
        for client in clients_in_round:
            invoker_params = InvokerParams(
                session_id=session_id,
                round_id=round_id,
                client_id=client.client_id,
                database=self.config.database,
                http_headers=http_headers,
            )

            async def g(params, invoker):
                return await self._call_invoker(params, invoker)

            task = asyncio.create_task(g(invoker_params, self.invoker_function))

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
        await self.deploy()

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

        with (out_dir / f"config_{session_id}.yaml").open("w") as f:
            yaml.dump(self.config.dict(), f)

        client_configs = list(client_config_dao.load_all(session_id=session_id))
        print(f"Found {len(client_configs)} registered clients for this session")

        should_abort = False

        log_metrics = []

        while not should_abort:
            round_start_time = time.time()
            clients_in_round = random.sample(
                client_configs, min(clients_per_round, len(client_configs))
            )
            try:
                await self._invoke_clients(clients_in_round, round_id, session_id)
            except InvocationTimeOut as e:
                print(e)
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
                self.aggregator_function,
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

            eval_infos = {}
            global_accuracy = None
            global_loss = None
            if aggregator_result.test_results:
                print(f"Computing test statistics from clients...")
                eval_dict = self.evaluate_clients(
                    metrics=aggregator_result.test_results,
                    metric_names=["loss", "accuracy"],
                )
                print(eval_dict)
                global_loss = eval_dict["mean_loss"]
                global_accuracy = eval_dict["mean_accuracy"]
                eval_infos.update(eval_dict)
            if self.test_data_config:
                print(f"Invoking evaluation function...")
                evaluator_params = EvaluatorParams(
                    session_id=session_id,
                    round_id=round_id,
                    database=self.config.database,
                    test_data=self.test_data_config,
                )
                evaluator_start_time = time.time()
                evaluator_result_dict = invoke_sync(
                    self.evaluator_function,
                    data=evaluator_params.dict(),
                    session=retry_session(backoff_factor=1.0, retries=5),
                )
                evaluator_end_time = time.time()
                evaluator_result = EvaluatorResult.parse_obj(evaluator_result_dict)
                global_accuracy = evaluator_result.metrics.metrics.get("accuracy", None)
                global_loss = evaluator_result.metrics.metrics.get("loss", None)
                eval_infos.update(
                    {
                        "evaluator_seconds": evaluator_end_time - evaluator_start_time,
                    }
                )

            log_metrics.append(
                {
                    "session_id": session_id,
                    "round_id": round_id,
                    "round_seconds": time.time() - round_start_time,
                    "clients_finished_seconds": (
                        clients_finished_time - round_start_time
                    ),
                    "aggregator_seconds": aggregator_end_time - aggregator_start_time,
                    "num_clients_round": len(clients_in_round),
                    "global_test_accuracy": global_accuracy,
                    "global_test_loss": global_loss,
                    **eval_infos,
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

            print(f"Global accuracy: {global_accuracy}, Global Loss: {global_loss}")
            f"Starting new round {aggregator_result.new_round_id}"
