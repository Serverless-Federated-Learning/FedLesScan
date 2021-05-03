import os
import time
from collections import defaultdict
from typing import Iterator, Tuple

import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disable tensorflow logs

from multiprocessing import Pool
import random

import click

from fedless.data import DatasetLoaderBuilder
from fedless.aggregation import FedAvgAggregator
from fedless.benchmark.fedkeeper import (
    create_mnist_train_data_loader_configs,
    create_mnist_cnn,
)
from fedless.client import default_handler
from fedless.models import (
    Hyperparams,
    ClientInvocationParams,
    DatasetLoaderConfig,
    MNISTConfig,
    SimpleModelLoaderConfig,
    ModelLoaderConfig,
    SerializedParameters,
    WeightsSerializerConfig,
    NpzWeightsSerializerConfig,
    LocalDifferentialPrivacyParams, ClientResult,
    EpsDelta
)
from fedless.serialization import (
    serialize_model,
    NpzWeightsSerializer,
    Base64StringConverter,
)


class NaiveAccounter:

    def __init__(self):
        self._client_guarantees = defaultdict(list)
        self._current_guarantees = dict()

    def update(self, results: Iterator[Tuple[MNISTConfig, EpsDelta]]):
        for data_config, guarantees in results:

            print(data_config, guarantees)
            key = data_config.json()
            self._client_guarantees[key].append(guarantees)

            if not key in self._current_guarantees:
                self._current_guarantees[key] = (guarantees.eps, guarantees.delta)
            else:
                eps, delta = self._current_guarantees[key]
                if delta != guarantees.delta:
                    print("Warning, eps is wrong!")
                self._current_guarantees[key] = (guarantees.eps + eps, guarantees.delta)
            print(key, self._current_guarantees[key])


@click.command()
@click.option("--devices", type=int, default=100)
@click.option("--epochs", type=int, default=100)
@click.option("--local-epochs", type=int, default=2)
@click.option("--local-batch-size", type=int, default=128)
@click.option("--clients-per-round", type=int, default=2)
@click.option("--l2-norm-clip", type=float, default=4.0)
@click.option("--noise-multiplier", type=float, default=1.0)
@click.option("--local-dp/--no-local-dp", type=bool, default=True)
def run(devices, epochs, local_epochs, local_batch_size, clients_per_round, l2_norm_clip, noise_multiplier, local_dp):
    # Setup
    privacy_params = (
        LocalDifferentialPrivacyParams(
            l2_norm_clip=l2_norm_clip, noise_multiplier=noise_multiplier, num_microbatches=1
        )
        if l2_norm_clip != 0.0 and noise_multiplier != 0.0 and local_dp
        else None
    )
    hyperparams = Hyperparams(
        batch_size=local_batch_size,
        epochs=local_epochs,
        metrics=["accuracy"],
        optimizer="Adam",
        local_privacy=privacy_params,
    )
    data_configs = list(create_mnist_train_data_loader_configs(n_devices=devices, n_shards=200))
    test_config = DatasetLoaderConfig(type="mnist", params=MNISTConfig(split="test"))
    test_set = DatasetLoaderBuilder.from_config(test_config).load()
    model = create_mnist_cnn()
    serialized_model = serialize_model(model)
    weight_bytes = NpzWeightsSerializer().serialize(model.get_weights())
    weight_string = Base64StringConverter.to_str(weight_bytes)
    params = SerializedParameters(
        blob=weight_string,
        serializer=WeightsSerializerConfig(
            type="npz", params=NpzWeightsSerializerConfig()
        ),
    )

    round_results = []
    test_accuracy = -1.0
    epoch = 0
    accounter = NaiveAccounter()
    start_time = time.time()
    while test_accuracy < 0.95 and epoch < epochs:
        model_loader = ModelLoaderConfig(
            type="simple",
            params=SimpleModelLoaderConfig(
                params=params,
                model=serialized_model.model_json,
                compiled=True,
                optimizer=serialized_model.optimizer,
                loss=serialized_model.loss,
                metrics=serialized_model.metrics,
            ),
        )

        invocation_params = []
        data_configs_for_round = random.sample(data_configs, clients_per_round)
        for data_config in data_configs_for_round:
            client_invocation_params = ClientInvocationParams(
                data=data_config, hyperparams=hyperparams, model=model_loader
            )
            invocation_params.append(
                (
                    client_invocation_params.data,
                    client_invocation_params.model,
                    client_invocation_params.hyperparams,
                    None,
                    NpzWeightsSerializer(),
                    Base64StringConverter(),
                    False,
                )
            )
        clients_invoked_time = time.time()
        with Pool() as p:
            results = p.starmap(default_handler, invocation_params)
        clients_finished_time = time.time()
        # for result in results:
        #    print(result.history)

        if local_dp:
            accounter.update(zip(data_configs_for_round, map(lambda result: result.privacy_guarantees, results)))

        new_parameters = FedAvgAggregator().aggregate(results)
        new_parameters_bytes = NpzWeightsSerializer().serialize(new_parameters)
        new_parameters_string = Base64StringConverter.to_str(new_parameters_bytes)
        params = SerializedParameters(
            blob=new_parameters_string,
            serializer=WeightsSerializerConfig(
                type="npz", params=NpzWeightsSerializerConfig()
            ),
        )

        model.set_weights(new_parameters)
        test_eval = model.evaluate(test_set.batch(32), return_dict=True, verbose=False)
        test_accuracy = test_eval["accuracy"]
        epoch += 1
        print(f"Epoch {epoch}/{epochs}: {test_eval}")
        round_results.append(
            {
                "test_loss": test_eval["loss"],
                "test_accuracy": test_eval["accuracy"],
                "epoch": epoch,
                "devices": devices,
                "epochs": epochs,
                "local_epochs": local_epochs,
                "clients_call_duration": clients_finished_time - clients_invoked_time,
                "clients_per_round": clients_per_round,
                "client_histories": [result.history for result in results],
                "privacy_params": privacy_params.json() if privacy_params else None,
                "privacy_guarantees": [
                    result.privacy_guarantees.json()
                    for result in results
                    if result.privacy_guarantees
                ],
            }
        )

        pd.DataFrame.from_records(round_results).to_csv(
            f"results_{devices}_{epochs}_{local_epochs}_{local_batch_size}"
            f"_{clients_per_round}_{l2_norm_clip}_{noise_multiplier}_{local_dp}_{start_time}.csv"
        )

if __name__ == "__main__":
    run()
