import cProfile
import logging
import pstats
import time
from unittest.mock import patch

import click
import numpy as np
import pymongo_inmemory

from fedless.benchmark.fedkeeper import create_mnist_cnn
from fedless.benchmark.leaf import create_femnist_cnn
from fedless.cache import _clear_cache
from fedless.client import fedless_mongodb_handler
from fedless.data import DatasetLoaderBuilder
from fedless.models import (
    MongodbConnectionConfig,
    SerializedParameters,
    WeightsSerializerConfig,
    NpzWeightsSerializerConfig,
    BinaryStringFormat,
    ClientConfig,
    MNISTConfig,
    DatasetLoaderConfig,
    Hyperparams,
    FunctionInvocationConfig,
    GCloudFunctionConfig,
    LEAFConfig,
    LeafDataset,
)
from fedless.persistence import ParameterDao, ClientConfigDao, ModelDao
from fedless.serialization import serialize_model, NpzWeightsSerializer

logging.basicConfig(level=logging.DEBUG)


def run_handler(mongo_client, session_id, round_id, client_id):
    fedless_mongodb_handler(
        session_id=session_id,
        round_id=round_id,
        client_id=client_id,
        database=MongodbConnectionConfig(
            host=mongo_client.address[0],
            port=mongo_client.address[1],
            username="",
            password="",
        ),
    )


@click.command()
@click.option("--out", type=str, default="out")
@click.option("--preload-dataset/--no-preload-dataset", default=True)
@click.option("--dataset", type=str, default="femnist")
@click.option("--repeats", type=int, default=10)
@click.option("--batch-size", type=int, default=10)
@click.option("--epochs", type=int, default=5)
def main(out, preload_dataset, dataset, repeats, batch_size, epochs):
    mongo_client = pymongo_inmemory.MongoClient()
    session_id = "session-123"
    round_id = 0
    client_id = "client-123"

    config_dao = ClientConfigDao(mongo_client)
    model_dao = ModelDao(mongo_client)
    parameter_dao = ParameterDao(mongo_client)

    # model = create_femnist_cnn()
    if dataset.lower() == "femnist":
        model = create_femnist_cnn()
        data = DatasetLoaderConfig(
            type="leaf",
            params=LEAFConfig(
                dataset=LeafDataset.FEMNIST,
                location="http://138.246.235.163:31715/data/leaf/data/femnist/data/train/user_100_train_9.json",
            ),
        )
    elif dataset.lower() == "mnist":
        model = create_mnist_cnn()
        data = DatasetLoaderConfig(
            type="mnist", params=MNISTConfig(indices=list(range(600)))
        )
    else:
        raise NotImplementedError()

    serialized_model = serialize_model(model)
    weights = model.get_weights()
    weights_serialized = NpzWeightsSerializer(compressed=False).serialize(weights)
    params = SerializedParameters(
        blob=weights_serialized,
        serializer=WeightsSerializerConfig(
            type="npz", params=NpzWeightsSerializerConfig(compressed=False)
        ),
        string_format=BinaryStringFormat.NONE,
    )
    model_dao.save(session_id=session_id, model=serialized_model)
    parameter_dao.save(session_id=session_id, round_id=0, params=params)
    client_config = ClientConfig(
        client_id=client_id,
        session_id=session_id,
        function=FunctionInvocationConfig(
            type="gcloud", params=GCloudFunctionConfig(url="https://test.com")
        ),
        data=data,
        hyperparams=Hyperparams(batch_size=batch_size, epochs=epochs),
    )
    config_dao.save(client_config)

    if preload_dataset:
        DatasetLoaderBuilder.from_config(client_config.data).load()

    with patch("fedless.client.pymongo.MongoClient") as mockMongoClient:
        mongo_client.close = lambda *args, **kwargs: None
        mockMongoClient.return_value = mongo_client

        # Run once so tensorflow functions etc. are already initialized for all
        run_handler(mongo_client, session_id, round_id, client_id)
        _clear_cache()

        print(f"Running functions multiple times to collect runtimes")
        no_cache_durations = []
        for repeat in range(repeats):
            _clear_cache()
            tik = time.time_ns()
            run_handler(mongo_client, session_id, round_id, client_id)
            no_cache_durations.append(time.time_ns() - tik)

        print(
            f"Avg. duration with whole cache deleted {np.average(no_cache_durations) / 10 ** 9}"
        )

        cache_durations = []
        for repeat in range(repeats):
            tik = time.time_ns()
            run_handler(mongo_client, session_id, round_id, client_id)
            cache_durations.append(time.time_ns() - tik)
        print(
            f"Avg. duration with cache activated {np.average(cache_durations) / 10 ** 9}"
        )

        no_model_cache_durations = []
        for repeat in range(repeats):
            _clear_cache()
            DatasetLoaderBuilder.from_config(client_config.data).load()
            tik = time.time_ns()
            run_handler(mongo_client, session_id, round_id, client_id)
            no_model_cache_durations.append(time.time_ns() - tik)
        print(
            f"Avg. duration with cache only activated for data {np.average(no_model_cache_durations) / 10 ** 9}"
        )

        print(f"Running profilers...")
        _clear_cache()
        profiler = cProfile.Profile()
        profiler.enable()
        run_handler(mongo_client, session_id, round_id, client_id)
        profiler.disable()
        (
            pstats.Stats(profiler)
            .strip_dirs()
            .sort_stats("cumtime")
            .print_callees("client.py:.*(fedless_mongodb_handler)")
            .dump_stats(f"{out}-cold.prof")
        )

        profiler = cProfile.Profile()
        profiler.enable()
        run_handler(mongo_client, session_id, round_id, client_id)
        profiler.disable()
        (
            pstats.Stats(profiler)
            .strip_dirs()
            .sort_stats("cumtime")
            .print_callees("client.py:.*(fedless_mongodb_handler)")
            .dump_stats(f"{out}-warm.prof")
        )


if __name__ == "__main__":
    main()
