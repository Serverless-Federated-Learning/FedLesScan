import abc
import base64
import binascii
import io
from typing import TypeVar, Callable, Union, Optional, List

import h5py
import numpy as np
import tensorflow as tf
from pydantic import BaseModel, Field, validator

from fedless.validation import params_validate_types_match


class SerializationError(Exception):
    """Object could not be (de)serialized"""


_RetT = TypeVar("_RetT")


def h5py_serialization_error_handler(
    func: Callable[..., _RetT]
) -> Callable[..., _RetT]:
    """
    Executes the function and wraps and rethrows any unhandled exception as a SerializationError
    :param func: Any serialization function
    :return: wrapped function
    """

    # noinspection PyMissingOrEmptyDocstring
    def new_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MemoryError:
            raise
        except Exception as e:
            # Catching general exceptions is bad practice but according to
            # https://docs.h5py.org/en/stable/faq.html#exceptions, we do not know which exceptions can be thrown
            # when opening/closing h5py files. As we're only dealing with in-memory representations, this should
            # hopefully be acceptable with the exception clause above
            raise SerializationError(e) from e

    return new_func


def wrap_exceptions_as_serialization_error(
    *exceptions: Exception.__class__,
) -> Callable[[Callable[..., _RetT]], Callable[..., _RetT]]:
    """
    Wrap and rethrow all specified exceptions as serialization errors.
    Can be used as a function decorator
    """

    def function_decorator(func: Callable[..., _RetT]) -> Callable[..., _RetT]:
        """Actual Function decorator responsible to wrap exceptions"""

        # noinspection PyMissingOrEmptyDocstring
        def new_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except tuple(exceptions) as e:
                raise SerializationError() from e

        return new_func

    return function_decorator


class StringSerializer(abc.ABC):
    """Convert raw bytes into string representation and vice versa"""

    @staticmethod
    @abc.abstractmethod
    def to_str(obj: bytes) -> str:
        """Convert raw bytes to string"""
        pass

    @staticmethod
    @abc.abstractmethod
    def from_str(rep: str) -> bytes:
        """Reconstruct raw bytes from string representation"""
        pass


class Base64StringConverter(StringSerializer):
    """
    Represents raw bytes as Base64 string representation. All created strings are valid ascii,
    this class can therefore also be used to send raw bytes in json payload via http
    """

    @staticmethod
    def to_str(obj: bytes) -> str:
        """
        Convert bytes object to base64 string
        :param obj: bytes
        :return: Base64 / ASCII string
        """
        encoding = base64.b64encode(obj)
        return encoding.decode(encoding="ascii")

    @staticmethod
    def from_str(rep: str) -> bytes:
        """
        Decodes Base64/ Ascii representation to original raw bytes
        :param rep: Base64/ ASCII string
        :return: bytes
        :raises ValueError if input is incorrectly padded or otherwise invalid
        """
        try:
            return base64.b64decode(rep)
        except binascii.Error:
            raise ValueError("Given string is not in base64 or incorrectly padded")


class ModelSerializer(abc.ABC):
    """
    Serialize tensorflow.keras.Model objects. Preserves architecture, weights and optimizer state.
    Please be aware that not possibly all implemented methods support custom components (layers, models, ...)
    """

    @abc.abstractmethod
    def serialize(self, model: tf.keras.Model) -> bytes:
        """Convert model into bytes"""
        pass

    @abc.abstractmethod
    def deserialize(self, blob: bytes) -> tf.keras.Model:
        """Reconstruct model from raw bytes"""
        pass


class H5FullModelSerializer(ModelSerializer):
    """
    Serializes the full model as an HDF5 file-string
    """

    class Config(BaseModel):
        """Configuration parameters for this serializer"""

        type: str = Field("h5", const=True)
        save_traces: bool = True

    def __init__(self, save_traces: bool = True):
        super().__init__()
        self.save_traces = save_traces

    @classmethod
    def from_config(cls, config: Config) -> "H5FullModelSerializer":
        """
        Create serializer from config
        :param config: configuration object
        :return: instantiated serializer
        """
        options = config.dict(exclude={"type"})
        return cls(**options)

    @h5py_serialization_error_handler
    def serialize(self, model: tf.keras.Model) -> bytes:
        """
        Save model, including weights and optimizer state, as raw bytes of h5py file
        :param model: Keras Model
        :return: raw bytes
        """
        with h5py.File(
            "does not matter", mode="w", driver="core", backing_store=False
        ) as h5file:
            model.save(
                filepath=h5file,
                include_optimizer=True,
                save_traces=self.save_traces,
            )
            return h5file.id.get_file_image()

    @h5py_serialization_error_handler
    def deserialize(self, blob: bytes) -> tf.keras.Model:
        """
        Reconstruct keras model from raw h5py file representation
        :param blob: bytes
        :return: Keras Model
        """
        fid = h5py.h5f.open_file_image(blob)
        with h5py.File(fid, mode="r+") as h5file:
            loaded_model = tf.keras.models.load_model(h5file)
        return loaded_model


class ModelSerializerConfig(BaseModel):
    """Configuration object for arbitrary model serializers of type :class:`ModelSerializer`"""

    type: str
    params: Optional[Union[H5FullModelSerializer.Config]]

    _params_type_matches_type = validator("params", allow_reuse=True)(
        params_validate_types_match
    )


class ModelSerializerBuilder:
    """Convenience class to directly create a serializer from its config"""

    @staticmethod
    def from_config(config: ModelSerializerConfig) -> ModelSerializer:
        """
        Create serializer from config
        :raises NotImplementedError if this serializer does not exist
        """
        if config.type == "h5":
            params: Optional[H5FullModelSerializer.Config] = config.params
            if config.params is not None:
                return H5FullModelSerializer.from_config(params)
            else:
                return H5FullModelSerializer()
        else:
            raise NotImplementedError(
                f"Serializer of type {config.type} does not exist"
            )


class WeightsSerializer(abc.ABC):
    """Serialize model weights/ list of numpy arrays as bytes"""

    @abc.abstractmethod
    def serialize(self, weights: List[np.ndarray]) -> bytes:
        """Convert into raw bytes"""
        pass

    @abc.abstractmethod
    def deserialize(self, blob: bytes) -> List[np.ndarray]:
        """Convert bytes into weights"""
        pass


class NpzWeightsSerializer(WeightsSerializer):
    """Serialize model weights as numpy npz object"""

    def __init__(self, compressed: bool = True):
        self.compressed = compressed

    def serialize(self, weights: List[np.ndarray]) -> bytes:
        """Convert into raw bytes, also supports basic compression"""
        with io.BytesIO() as f:
            if self.compressed:
                np.savez_compressed(f, *weights)
            else:
                np.savez(f, *weights)
            return f.getvalue()

    @wrap_exceptions_as_serialization_error(ValueError, IOError)
    def deserialize(self, blob: bytes) -> List[np.ndarray]:
        """Convert bytes into weights"""
        with io.BytesIO(blob) as f:
            npz_obj = np.load(f)
            return list(npz_obj.values())
