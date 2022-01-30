import logging
import sys

# import uuid
# from itertools import cycle
# from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# import click
import tensorflow as tf

from fedless.datasets.leaf.model import create_femnist_cnn, create_shakespeare_lstm
from fedless.datasets.mnist.dataset_loader import create_mnist_train_data_loader_configs
from fedless.datasets.mnist.model import create_mnist_cnn
from fedless.models import (
    BinaryStringFormat,
    DatasetLoaderConfig,
    LEAFConfig,
    MNISTConfig,
    MongodbConnectionConfig,
    NpzWeightsSerializerConfig,
    SerializedParameters,
    WeightsSerializerConfig,
)
from fedless.persistence.client_daos import ModelDao, ParameterDao
from fedless.serialization import (
    Base64StringConverter,
    NpzWeightsSerializer,
    serialize_model,
)

logger = logging.getLogger(__name__)


def create_model(dataset) -> tf.keras.Sequential:
    if dataset.lower() == "femnist":
        return create_femnist_cnn()
    elif dataset.lower() == "shakespeare":
        return create_shakespeare_lstm()
    elif dataset.lower() == "mnist":
        return create_mnist_cnn()
    else:
        raise NotImplementedError()


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


# only for global test data
def create_mnist_test_config(proxies) -> DatasetLoaderConfig:
    return DatasetLoaderConfig(
        type="mnist", params=MNISTConfig(split="test", proxies=proxies)
    )


FILE_SERVER = "http://138.246.235.175:81"


# noinspection PydanticTypeChecker,PyTypeChecker
def create_data_configs(
    dataset: str, clients: int, proxies: Optional[Dict] = None
) -> List[Union[DatasetLoaderConfig, Tuple[DatasetLoaderConfig, DatasetLoaderConfig]]]:
    dataset = dataset.lower()
    if dataset == "mnist":
        return list(
            create_mnist_train_data_loader_configs(
                n_devices=clients, n_shards=600, proxies=proxies
            )
        )
    elif dataset in ["femnist", "shakespeare"]:
        configs = []
        for client_idx in range(clients):
            train = DatasetLoaderConfig(
                type="leaf",
                params=LEAFConfig(
                    dataset=dataset,
                    location=f"{FILE_SERVER}/datasets/leaf/data/{dataset}/data/"
                    f"train/user_{client_idx}_train_9.json",
                ),
            )
            test = DatasetLoaderConfig(
                type="leaf",
                params=LEAFConfig(
                    dataset=dataset,
                    location=f"{FILE_SERVER}/datasets/leaf/data/{dataset}/data/"
                    f"test/user_{client_idx}_test_9.json",
                ),
            )
            configs.append((train, test))
        return configs
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")
