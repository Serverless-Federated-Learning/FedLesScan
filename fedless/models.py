from __future__ import annotations

import abc
from typing import Union

import pydantic
import tensorflow as tf
from pydantic import Field, BaseModel, validator

from fedless.serialization import (
    ModelSerializer,
    SerializationError,
    ModelSerializerConfig,
    ModelSerializerBuilder,
    Base64StringConverter,
)
from fedless.validation import params_validate_types_match


class ModelLoadError(Exception):
    """Model could not be loaded"""


class ModelLoader(abc.ABC):
    """Load keras model from arbitrary source"""

    @abc.abstractmethod
    def load(self) -> tf.keras.Model:
        """Load model"""
        pass


class PayloadModelLoaderConfig(pydantic.BaseModel):
    """Configuration parameters required for :class:`PayloadModelLoader`"""

    type: str = Field("payload", const=True)
    payload: str
    serializer: ModelSerializerConfig = ModelSerializerConfig(type="h5")


class PayloadModelLoader(ModelLoader):
    """
    Send serialized models directly as part of the configuration object.
    Not advisable for large models.
    """

    def __init__(self, payload: str, serializer: ModelSerializer):
        self.payload = payload
        self.serializer = serializer

    @classmethod
    def from_config(cls, config: PayloadModelLoaderConfig) -> PayloadModelLoader:
        """Create loader from :class:`PayloadModelLoaderConfig`"""
        payload = config.payload
        serializer = ModelSerializerBuilder.from_config(config.serializer)
        return cls(payload=payload, serializer=serializer)

    def load(self) -> tf.keras.Model:
        """
        Deserialize payload and return model
        :raises ModelLoadError if payload is invalid or other error occurred during deserialization
        """
        try:
            raw_bytes = Base64StringConverter.from_str(self.payload)
            return self.serializer.deserialize(raw_bytes)
        except SerializationError as e:
            raise ModelLoadError("Model could not be deserialized") from e
        except ValueError as e:
            raise ModelLoadError("Malformed or otherwise invalid payload") from e


class ModelLoaderConfig(pydantic.BaseModel):
    """Configuration for arbitrary :class:`ModelLoader`'s"""

    type: str
    params: Union[PayloadModelLoaderConfig]

    _params_type_matches_type = validator("params", allow_reuse=True)(
        params_validate_types_match
    )


class ModelLoaderBuilder:
    """Convenience class to create loader from :class:`ModelLoaderConfig`"""

    @staticmethod
    def from_config(config: ModelLoaderConfig):
        """
        Construct a model loader from the config
        :raises NotImplementedError if the loader does not exist
        """
        if config.type == "payload":
            params: PayloadModelLoaderConfig = config.params
            return PayloadModelLoader.from_config(params)
        else:
            raise NotImplementedError(f"Model loader {config.type} is not implemented")
