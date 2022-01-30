from typing import Iterator, Optional, Dict

import numpy as np
from tensorflow import keras
from fedless.datasets.dataset_loaders import DatasetLoader, DatasetNotLoadedError
import tensorflow as tf
from typing import  Dict, Iterator, List, Optional
from fedless.models import (
    DatasetLoaderConfig,
    MNISTConfig,
)

import requests
import tempfile
import os
from fedless.cache import cache


# Helper functions to create dataset shards / model


def create_mnist_train_data_loader_configs(
    n_devices: int, n_shards: int, proxies: Optional[Dict] = None
) -> Iterator[DatasetLoaderConfig]:
    if n_shards % n_devices != 0:
        raise ValueError(
            f"Can not equally distribute {n_shards} dataset shards among {n_devices} devices..."
        )

    (_, y_train), (_, _) = keras.datasets.mnist.load_data()
    num_train_examples, *_ = y_train.shape

    sorted_labels_idx = np.argsort(y_train, kind="stable")
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
            type="mnist", params=MNISTConfig(indices=indices.tolist(), proxies=proxies)
        )



class MNIST(DatasetLoader):
    def __init__(
        self,
        indices: Optional[List[int]] = None,
        split: str = "train",
        proxies: Optional[Dict] = None,
    ):
        self.split = split
        self.indices = indices
        self.proxies = proxies or {}

    @cache
    def load(self) -> tf.data.Dataset:
        response = requests.get(
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            proxies=self.proxies,
        )
        fp, path = tempfile.mkstemp()
        with os.fdopen(fp, "wb") as f:
            f.write(response.content)

        with np.load(path, allow_pickle=True) as f:
            x_train, y_train = f["x_train"], f["y_train"]
            x_test, y_test = f["x_test"], f["y_test"]

        if self.split.lower() == "train":
            features, labels = x_train, y_train
        elif self.split.lower() == "test":
            features, labels = x_test, y_test
        else:
            raise DatasetNotLoadedError(f"Mnist split {self.split} does not exist")

        if self.indices:
            features, labels = features[self.indices], labels[self.indices]

        def _scale_features(features, label):
            return tf.cast(features, tf.float32) / 255.0, tf.cast(label, tf.int32)

        ds = tf.data.Dataset.from_tensor_slices((features, labels))

        return ds.map(_scale_features)

