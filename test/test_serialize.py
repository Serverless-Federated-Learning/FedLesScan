import base64
import json
import random
from itertools import zip_longest
from json import JSONDecodeError
from typing import List, Tuple
from unittest.mock import patch

import h5py
import numpy as np
import pydantic
from _pytest.monkeypatch import MonkeyPatch
from pydantic import ValidationError

from fedless.serialization import (
    H5FullModelSerializer,
    ModelSerializer,
    SerializationError,
    Base64StringConverter,
    ModelSerializerConfig,
    ModelSerializerBuilder,
    NpzWeightsSerializer,
)
from .common import get_error_function
from .fixtures import *


def are_weights_equal(weight_old: List[np.ndarray], weights_new: List[np.ndarray]):
    for a, b in zip_longest(weight_old, weights_new):
        if not np.allclose(a, b):
            return False
    return True


def is_optimizer_state_preserved(
    optimizer_old: tf.keras.Model, optimizer_new: tf.keras.Model
):
    if optimizer_old is None or optimizer_new is None:
        return False
    if not optimizer_old.get_config() == optimizer_new.get_config():
        return False
    if not are_weights_equal(optimizer_old.get_weights(), optimizer_new.get_weights()):
        return False
    return True


def is_model_trainable(model: tf.keras.Model, data: Tuple[np.ndarray, np.ndarray]):
    features, labels = data
    try:
        model.fit(features, labels, batch_size=1)
    except (RuntimeError, ValueError):
        return False
    return True


def is_valid_json(json_str):
    try:
        json.loads(json_str)
    except (JSONDecodeError, ValueError):
        return False
    return True


def assert_models_equal(model: tf.keras.Model, model_re: tf.keras.Model):
    assert isinstance(model_re, tf.keras.Model)
    assert model_re.get_config() == model.get_config()
    assert are_weights_equal(model.get_weights(), model_re.get_weights())
    assert is_optimizer_state_preserved(model.optimizer, model_re.optimizer)


@pytest.fixture
def dummy_data() -> Tuple[np.ndarray, np.ndarray]:
    n_samples = 10
    return np.random.randn(n_samples, 3), np.random.randn(n_samples, 4)


def model_serializer_test_suite(
    model: tf.keras.Model,
    serializer: ModelSerializer,
    dummy_data: Tuple[np.ndarray, np.ndarray],
):
    blob = serializer.serialize(model)
    model_re = serializer.deserialize(blob)

    assert_models_equal(model, model_re)
    assert is_model_trainable(model, dummy_data)


def test_h5_serializer(simple_model, dummy_data: Tuple[np.ndarray, np.ndarray]):
    s = H5FullModelSerializer()
    model_serializer_test_suite(simple_model, s, dummy_data)


def test_h5_serializer_rethrows_exception(simple_model, monkeypatch: MonkeyPatch):
    s = H5FullModelSerializer()

    with pytest.raises(SerializationError):
        monkeypatch.setattr(simple_model, "save", get_error_function(ImportError))
        s.serialize(simple_model)

    with pytest.raises(SerializationError):
        monkeypatch.setattr(h5py.File, "__enter__", get_error_function(IOError))
        s.serialize(simple_model)


def test_h5_serializer_does_not_wrap_memory_error(
    simple_model, monkeypatch: MonkeyPatch
):
    s = H5FullModelSerializer()
    blob = s.serialize(simple_model)

    with pytest.raises(MemoryError):
        monkeypatch.setattr(h5py.File, "__enter__", get_error_function(MemoryError))
        s.deserialize(blob)


def test_h5_serializer_fails_on_invalid_blob(simple_model, monkeypatch: MonkeyPatch):
    s = H5FullModelSerializer()

    blob = s.serialize(simple_model)

    with pytest.raises(SerializationError):
        monkeypatch.setattr(tf.keras.models, "load_model", get_error_function(IOError))
        s.deserialize(blob)

    with pytest.raises(SerializationError):
        monkeypatch.setattr(
            tf.keras.models, "load_model", get_error_function(ImportError)
        )
        s.deserialize(blob)


def test_h5_serializer_config_type():
    config = H5FullModelSerializer.Config(save_traces=False)
    assert not config.save_traces

    config = H5FullModelSerializer.Config(type="h5", save_traces=True)
    assert config.save_traces and config.type == "h5"

    with pytest.raises(pydantic.ValidationError):
        H5FullModelSerializer.Config(type="s3")

    with pytest.raises(pydantic.ValidationError):
        # noinspection PyTypeChecker
        H5FullModelSerializer.Config(save_traces="Yes, please")


@pytest.mark.parametrize("save_traces", [True, False])
def test_h5_serializer_can_be_constructed_from_config(save_traces):
    config = H5FullModelSerializer.Config(save_traces=save_traces)
    serializer = H5FullModelSerializer.from_config(config)
    assert serializer is not None
    assert serializer.save_traces == save_traces


def test_model_serializer_config_types_must_match():
    with pytest.raises(ValidationError):
        config_dict = {"type": "s3", "params": {"type": "h5"}}
        ModelSerializerConfig.parse_obj(config_dict)


@pytest.mark.parametrize("serializer_config", [H5FullModelSerializer.Config()])
def test_model_serializer_config_types_can_be_created(serializer_config):
    config = ModelSerializerConfig(
        type=serializer_config.type, params=serializer_config
    )
    assert config is not None
    assert config.type == serializer_config.type
    assert config.params == serializer_config


def test_model_serializer_builder_fails_on_unknown_type():
    config = ModelSerializerConfig(type="invalid_type")
    with pytest.raises(NotImplementedError):
        ModelSerializerBuilder.from_config(config)


@patch("fedless.serialization.H5FullModelSerializer")
def test_model_serializer_builder_creates_object(serializer_mock):
    serializer_config = H5FullModelSerializer.Config()
    config = ModelSerializerConfig(type="h5", params=serializer_config)
    serializer = ModelSerializerBuilder.from_config(config)
    assert serializer is not None
    assert serializer_mock.from_config.called_with(serializer_config)


@patch("fedless.serialization.H5FullModelSerializer")
def test_model_serializer_builder_creates_object_without_params_specified(
    serializer_mock,
):
    config = ModelSerializerConfig(type="h5")
    serializer = ModelSerializerBuilder.from_config(config)
    assert serializer is not None
    assert serializer_mock.called_with()


@pytest.mark.parametrize("bytes_length", [0, 1, 16, 512])
def test_base64_converter_on_random_bytes(bytes_length):
    random_bytes = bytes([random.randrange(0, 256) for _ in range(bytes_length)])
    json_str = Base64StringConverter.to_str(random_bytes)
    assert Base64StringConverter.from_str(json_str) == random_bytes


def test_base64_converter_raises_error_on_internal_error(monkeypatch):
    monkeypatch.setattr(base64, "b64decode", get_error_function(ValueError))
    with pytest.raises(ValueError):
        valid_b64_str = "SSBoYXZlIHRoZSBoaWdoIGdyb3VuZCE"
        Base64StringConverter.from_str(valid_b64_str)


def test_base64_converter_throws_error_on_invalid_string():
    with pytest.raises(ValueError):
        invalid_b64_str = "!nv4l!D!"
        Base64StringConverter.from_str(invalid_b64_str)


@pytest.mark.parametrize("compressed", [True, False])
def test_npz_weights_serializer_restores_weights(
    simple_model: tf.keras.Model, compressed
):
    s = NpzWeightsSerializer(compressed=compressed)

    weights = simple_model.get_weights()
    reconstructured_weights = s.deserialize(s.serialize(weights))

    assert all(
        (np.allclose(a, b) for a, b in zip_longest(weights, reconstructured_weights))
    )


@pytest.mark.parametrize("compressed", [True, False])
def test_npz_weights_serialier_does_not_fail_on_empty_input(compressed):
    s = NpzWeightsSerializer(compressed=compressed)
    assert s.deserialize(s.serialize([])) == []


@pytest.mark.parametrize("compressed", [True, False])
def test_npz_weights_restores_types(compressed):
    weights = [
        np.random.randn(10, 15).astype(np.float32),
        np.random.randint(0, 32, (8, 5)),
        np.random.randint(0, 1, (5, 6)).astype(np.int8),
    ]

    s = NpzWeightsSerializer(compressed=compressed)
    reconstructured_weights = s.deserialize(s.serialize(weights))
    assert all(
        (
            np.allclose(a, b) and a.dtype == a.dtype
            for a, b in zip_longest(weights, reconstructured_weights)
        )
    )


@pytest.mark.parametrize("compressed", [True, False])
def test_npz_weights_serializer_wraps_loading_errors(
    compressed, monkeypatch: MonkeyPatch
):
    s = NpzWeightsSerializer(compressed=compressed)

    with pytest.raises(SerializationError):
        monkeypatch.setattr(np, "load", get_error_function(ValueError))
        s.deserialize(b"")

    with pytest.raises(SerializationError):
        monkeypatch.setattr(np, "load", get_error_function(IOError))
        s.deserialize(b"")
