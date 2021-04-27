import os
import time

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
    LocalDifferentialPrivacyParams,
)
from fedless.serialization import (
    serialize_model,
    NpzWeightsSerializer,
    Base64StringConverter,
)


@click.command()
@click.option("--devices", type=int, default=100)
@click.option("--epochs", type=int, default=100)
@click.option("--local-epochs", type=int, default=10)
@click.option("--clients-per-round", type=int, default=5)
@click.option("--l2-norm-clip", type=float, default=2.0)
def run(devices, epochs, local_epochs, clients_per_round, l2_norm_clip):
    # Setup

    privacy_params = (
        LocalDifferentialPrivacyParams(
            l2_norm_clip=l2_norm_clip, noise_multiplier=1.0, num_microbatches=8
        )
        if l2_norm_clip != 0.0
        else None
    )
    hyperparams = Hyperparams(
        batch_size=32,
        epochs=local_epochs,
        metrics=["accuracy"],
        local_privacy=privacy_params,
    )
    splits = create_mnist_train_data_loader_configs(n_devices=devices, n_shards=200)
    data_configs = [
        DatasetLoaderConfig(type="mnist", params=MNISTConfig(indices=client_idx_list))
        for client_idx_list in splits
    ]
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
            f"results_{devices}_{epochs}_{local_epochs}_{clients_per_round}.csv"
        )


if __name__ == "__main__":
    run()
