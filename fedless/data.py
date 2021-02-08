import abc
import json
from enum import Enum
from functools import reduce
from json import JSONDecodeError
from pathlib import Path
from typing import Union, Dict, Iterator

import pydantic
import requests
import tensorflow as tf
from requests import RequestException

from fedless.validation import params_validate_types_match
from pydantic import validate_arguments, AnyHttpUrl, BaseModel, Field, validator


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


class LEAF(DatasetLoader):
    """
    Utility class to load and process the LEAF datasets as published in
    https://arxiv.org/pdf/1812.01097.pdf and https://github.com/TalwalkarLab/leaf
    """

    class LeafDataset(Enum):
        """
        Officially supported datasets
        """

        FEMNIST = "femnist"
        REDDIT = "reddit"
        CELEBA = "celeba"
        SHAKESPEARE = "shakespeare"
        SENT140 = "sent140"

    @validate_arguments
    def __init__(
        self,
        dataset: LeafDataset,
        location: Union[AnyHttpUrl, Path],
        http_params: Dict = None,
    ):
        """
        Create dataset loader for the specified source
        :param dataset: Dataset name, one of :py:class:`fedless.data.LEAF.LeafDataset`
        :param location: Location of dataset partition in form of a json file.
        :param http_params: Additional parameters to send with http request. Only used when location is an URL
         Use location:// to load from disk. For valid entries see :py:meth:`requests.get`
        """
        self.dataset = dataset
        self.source = location
        self.http_params = http_params

        if dataset != self.LeafDataset.FEMNIST:
            raise NotImplementedError()

    def _iter_dataset_files(self) -> Iterator[Union[AnyHttpUrl, Path]]:
        if isinstance(self.source, AnyHttpUrl):
            yield self.source
        elif isinstance(self.source, Path) and self.source.is_dir():
            for file in self.source.iterdir():
                if file.is_file() and file.suffix == ".json":
                    yield file
        else:
            yield self.source

    @staticmethod
    def _convert_dict_to_dataset(file_content: Dict) -> tf.data.Dataset:
        try:
            users = file_content["users"]
            user_data = file_content["user_data"]
            for user in users:
                data_for_user = user_data[user]
                yield tf.data.Dataset.from_tensor_slices(
                    (data_for_user["x"], data_for_user["y"])
                )
        except (KeyError, TypeError, ValueError) as e:
            raise DatasetFormatError(e) from e

    def _process_all_sources(self) -> Iterator[tf.data.Dataset]:
        for source in self._iter_dataset_files():
            file_content: Dict = self._read_source(source)
            for dataset in self._convert_dict_to_dataset(file_content):
                yield dataset

    def _read_source(self, source: Union[AnyHttpUrl, Path]) -> Dict:
        if isinstance(source, AnyHttpUrl):
            return self._fetch_url(source)
        else:
            return self._read_file_content(source)

    def _fetch_url(self, url: str):
        try:
            response = requests.get(url, params=self.http_params)
            response.raise_for_status()
            return response.json()
        except ValueError as e:
            raise DatasetFormatError(f"Invalid JSON returned from ${url}") from e
        except RequestException as e:
            raise DatasetNotLoadedError(e) from e

    @classmethod
    def _read_file_content(cls, path: Path) -> Dict:
        try:
            with path.open() as f:
                return json.load(f)
        except (JSONDecodeError, ValueError) as e:
            raise DatasetFormatError(e) from e
        except (IOError, OSError) as e:
            raise DatasetNotLoadedError(e) from e

    def load(self) -> tf.data.Dataset:
        """
        Load dataset
        :raise DatasetNotLoadedError when an error occurred in the process
        """
        sources = self._process_all_sources()
        try:
            return merge_datasets(sources)
        except TypeError as e:
            raise DatasetFormatError(e) from e


class LEAFConfig(BaseModel):
    """Configuration parameters for LEAF dataset loader"""

    type: str = Field("leaf", const=True)
    dataset: LEAF.LeafDataset
    location: Union[AnyHttpUrl, Path]
    http_params: Dict = None


class DatasetLoaderConfig(pydantic.BaseModel):
    """Configuration for arbitrary dataset loaders"""

    type: str
    params: Union[LEAFConfig]

    _params_type_matches_type = validator("params", allow_reuse=True)(
        params_validate_types_match
    )


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
            return LEAF(
                dataset=params.dataset,
                location=params.location,
                http_params=params.http_params,
            )
        else:
            raise NotImplementedError(
                f"Dataset loader {config.type} is not implemented"
            )
