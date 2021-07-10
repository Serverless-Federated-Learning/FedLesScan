import cProfile
import logging
import pstats

import click
import pymongo_inmemory
from unittest.mock import patch

from fedless.client import fedless_mongodb_handler
from fedless.models import (
    MongodbConnectionConfig,
    SerializedModel,
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
from fedless.benchmark.leaf import create_femnist_cnn
from fedless.benchmark.fedkeeper import create_mnist_cnn
from fedless.data import MNIST, DatasetLoaderBuilder
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
@click.option("--out", type=click.Path(), required=True)
@click.option("--preload-dataset/--no-preload-dataset", default=True)
@click.option("--dataset", type=str, required=True)
def main(out, preload_dataset, dataset):
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
        hyperparams=Hyperparams(batch_size=10, epochs=5),
    )
    config_dao.save(client_config)

    if preload_dataset:
        DatasetLoaderBuilder.from_config(client_config.data).load()

    with patch("fedless.client.pymongo.MongoClient") as mockMongoClient:
        mockMongoClient.return_value = mongo_client
        profiler = cProfile.Profile()
        profiler.enable()
        run_handler(mongo_client, session_id, round_id, client_id)
        profiler.disable()
        (
            pstats.Stats(profiler)
            .strip_dirs()
            .sort_stats("cumtime")
            .print_callees("client.py:.*(fedless_mongodb_handler)")
            .dump_stats(out)
        )


if __name__ == "__main__":
    main()
