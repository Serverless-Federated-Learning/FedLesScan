import abc
import logging
from functools import reduce
from typing import Iterator 

import tensorflow as tf

from fedless.cache import cache
from fedless.datasets import leaf
from fedless.datasets.mnist.dataset_loader import MNIST
from fedless.models import LEAFConfig, DatasetLoaderConfig,  MNISTConfig

logger = logging.getLogger(__name__)


class DatasetNotLoadedError(Exception):
    """Dataset could not be loaded"""


class DatasetFormatError(DatasetNotLoadedError):
    """Source file containing data is malformed or otherwise invalid"""


def merge_datasets(datasets: Iterator[tf.data.Dataset]) -> tf.data.Dataset:
    """
    Merge the given datasets into one by concatenating them
    :param datasets: Iterator with all datasets
    :return: Final combined dataset
    :raises TypeError in tf.data.Dataset.concatenate
    """
    return reduce(tf.data.Dataset.concatenate, datasets)


class DatasetLoader(abc.ABC):
    """Load arbitrary datasets"""

    @abc.abstractmethod
    def load(self) -> tf.data.Dataset:
        """Load dataset"""
        pass

class DatasetLoaderBuilder:
    """Convenience class to construct loaders from config"""

    @staticmethod
    def from_config(config: DatasetLoaderConfig) -> DatasetLoader:
        """
        Construct loader from config
        :raises NotImplementedError if the loader does not exist
        """
        if config.type == "leaf":
            params: LEAFConfig = config.params
            return leaf(
                dataset=params.dataset,
                location=params.location,
                http_params=params.http_params,
                user_indices=params.user_indices,
            )
        elif config.type == "mnist":
            params: MNISTConfig = config.params
            return MNIST(
                split=params.split, indices=params.indices, proxies=params.proxies
            )
        else:
            raise NotImplementedError(
                f"Dataset loader {config.type} is not implemented"
            )
