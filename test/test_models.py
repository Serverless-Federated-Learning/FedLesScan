from unittest.mock import patch, MagicMock

import pydantic
import pytest
import tensorflow as tf

from .fixtures import simple_model
from fedless.models import (
    ModelLoaderBuilder,
    ModelLoaderConfig,
    PayloadModelLoader,
    PayloadModelLoaderConfig,
    ModelLoadError,
)
from fedless.serialization import (
    ModelSerializerConfig,
    ModelSerializerBuilder,
    ModelSerializer,
    SerializationError,
)


class SerializerStub(ModelSerializer):
    def __init__(self):
        self.model = None

    def serialize(self, model):
        self.model = model
        return b""

    def deserialize(self, blob: bytes):
        return self.model


def test_model_loader_builder_raises_not_implemented_error():
    config = MagicMock(ModelLoaderConfig)
    config.type = "does-not-exist"

    with pytest.raises(NotImplementedError):
        ModelLoaderBuilder.from_config(config)


@patch("fedless.models.PayloadModelLoader")
def test_model_loader_builder_returns_correct_object(payload_model_loader_mock):
    class LoaderStub:
        pass

    config_mock = ModelLoaderConfig(
        type="payload", params=PayloadModelLoaderConfig(payload="abc123")
    )

    payload_model_loader_mock.from_config.return_value = loader_stub = LoaderStub()

    model_loader = ModelLoaderBuilder.from_config(config_mock)
    assert model_loader == loader_stub
    payload_model_loader_mock.from_config.assert_called_with(config_mock.params)


@patch("fedless.validation.params_validate_types_match")
def test_model_loader_config_types_match(params_validate_types_match):
    payload_config = MagicMock(PayloadModelLoaderConfig)
    with pytest.raises(pydantic.ValidationError):
        ModelLoaderConfig(type="other", params=payload_config)
    assert params_validate_types_match.called_at_least_once


def test_model_loader_config_only_accepts_valid_configs():
    class FakeConfig(pydantic.BaseModel):
        type: str = "does-not-exist"
        attr: int

    with pytest.raises(pydantic.ValidationError):
        # noinspection PyTypeChecker
        ModelLoaderConfig(type="does-not-exist", params=FakeConfig(attr=2))


def test_payload_model_loader_config_type_fixed():
    with pytest.raises(pydantic.ValidationError):
        PayloadModelLoaderConfig(type="something-else", payload="")


def test_payload_model_loader_config_from_dict():
    config_dict = {
        "payload": "abc",
    }
    config = PayloadModelLoaderConfig.parse_obj(config_dict)
    assert config is not None
    assert config.payload == "abc"


def test_payload_model_loader_from_config_correct():
    with patch.object(ModelSerializerBuilder, "from_config") as from_config_mock:
        from_config_mock.return_value = serializer_stub = SerializerStub()

        config = PayloadModelLoaderConfig(
            payload="abc", serializer=ModelSerializerConfig(type="h5")
        )
        loader = PayloadModelLoader.from_config(config)
    assert loader is not None
    assert loader.payload == "abc"
    assert loader.serializer == serializer_stub


def test_payload_model_loader_fails_on_invalid_serializer():
    with pytest.raises(NotImplementedError):
        with patch.object(ModelSerializerBuilder, "from_config") as from_config_mock:
            from_config_mock.side_effect = NotImplementedError()

            PayloadModelLoader.from_config(
                PayloadModelLoaderConfig(
                    payload="abc", serializer=ModelSerializerConfig(type="h5")
                )
            )


@patch("fedless.models.Base64StringConverter")
@patch("fedless.models.ModelSerializerBuilder")
def test_payload_model_loader_works_correctly(
    serializer_builder_mock, string_converter_mock, simple_model
):
    serializer_stub = MagicMock(SerializerStub())
    serializer_stub.deserialize.return_value = simple_model
    serializer_builder_mock.from_config.return_value = serializer_stub
    string_converter_mock.from_str.return_value = b"0123abc"

    loader = PayloadModelLoader.from_config(
        PayloadModelLoaderConfig(
            payload="abc", serializer=ModelSerializerConfig(type="h5")
        )
    )

    model: tf.keras.Model = loader.load()

    string_converter_mock.from_str.assert_called_with("abc")
    serializer_stub.deserialize.assert_called_with(b"0123abc")
    assert model == simple_model


@patch("fedless.models.Base64StringConverter")
@patch("fedless.models.ModelSerializerBuilder")
def test_payload_model_loader_throws_model_error_when_serializer_fails(
    serializer_builder_mock, string_converter_mock
):
    with pytest.raises(ModelLoadError):
        string_converter_mock.from_str.return_value = b"0123abc"
        serializer_stub = MagicMock(SerializerStub())
        serializer_stub.deserialize.side_effect = SerializationError()
        serializer_builder_mock.from_config.return_value = serializer_stub

        loader = PayloadModelLoader.from_config(
            PayloadModelLoaderConfig(
                payload="abc", serializer=ModelSerializerConfig(type="h5")
            )
        )

        tf.keras.Model = loader.load()


@patch("fedless.models.Base64StringConverter")
def test_payload_model_loader_throws_model_error_for_invalid_payload(
    string_converter_mock,
):
    with pytest.raises(ModelLoadError):
        string_converter_mock.from_str.side_effect = ValueError()

        loader = PayloadModelLoader.from_config(
            PayloadModelLoaderConfig(
                payload="abc", serializer=ModelSerializerConfig(type="h5")
            )
        )

        tf.keras.Model = loader.load()
