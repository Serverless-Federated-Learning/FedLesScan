import base64
import itertools
import json
from copy import copy
from typing import List
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf
from pytest import fixture
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.data import Dataset

from fedless.client import (
    run,
    default_handler,
    Hyperparams,
    ClientError,
    create_http_success_response,
    ClientResult,
    create_http_user_error_response,
    lambda_proxy_handler,
    create_gcloud_http_success_response,
    create_gcloud_http_user_error_response,
    gcloud_http_error_handler,
    openwhisk_action_handler,
)
from fedless.data import DatasetNotLoadedError
from fedless.models import ModelLoadError
from fedless.serialization import SerializationError
from .stubs import (
    StringSerializerStub,
    WeightsSerializerStub,
    DatasetLoaderStub,
    ModelLoaderStub,
)

SAMPLES = 10
FEATURE_DIM = 5
CLASSES = 3


@fixture
def hyperparams():
    # noinspection PyTypeChecker
    return Hyperparams(batch_size=1, epochs=2)


@pytest.fixture
def dataset():
    features = np.random.randn(SAMPLES, FEATURE_DIM)
    labels = np.random.randint(low=0, high=CLASSES, size=SAMPLES)
    return Dataset.from_tensor_slices((features, labels))


@fixture
def data_loader(dataset):
    return DatasetLoaderStub(dataset)


@fixture
def model_loader(model):
    return ModelLoaderStub(model)


@pytest.fixture
def model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(FEATURE_DIM, input_shape=(FEATURE_DIM,)),
            tf.keras.layers.Dense(CLASSES, activation="softmax"),
        ]
    )
    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd")
    return model


@pytest.fixture
def weights_serializer():
    return WeightsSerializerStub([], b"1234")


@pytest.fixture
def string_serializer():
    return StringSerializerStub(b"", "serialized")


def test_run_returns_weights(
    model_loader, data_loader, hyperparams, weights_serializer, string_serializer
):
    result = run(
        data_loader=data_loader,
        model_loader=model_loader,
        hyperparams=hyperparams,
        weights_serializer=weights_serializer,
        string_serializer=string_serializer,
    )
    assert result.weights == string_serializer.string


def test_run_batches_data(
    model_loader: ModelLoaderStub,
    data_loader: DatasetLoaderStub,
    hyperparams,
    weights_serializer,
    string_serializer,
):
    with patch.object(
        data_loader.dataset, "batch", wraps=data_loader.dataset.batch
    ) as mocked_batch:
        run(
            data_loader=data_loader,
            model_loader=model_loader,
            hyperparams=hyperparams,
            weights_serializer=weights_serializer,
            string_serializer=string_serializer,
        )
        mocked_batch.assert_called_with(hyperparams.batch_size)


@pytest.mark.parametrize(
    ["optimizer", "loss", "metrics"],
    itertools.product(
        [SGD(learning_rate=0.01), Adam()],
        ["sparse_categorical_crossentropy"],
        [[], ["accuracy", "mse"]],
    ),
)
def test_run_overwrites_hyperparameters(
    model_loader: ModelLoaderStub,
    data_loader: DatasetLoaderStub,
    weights_serializer,
    string_serializer,
    optimizer: tf.keras.optimizers.Optimizer,
    loss: str,
    metrics: List[str],
):
    hyperparams = Hyperparams(
        batch_size=1,
        epochs=2,
        optimizer=tf.keras.optimizers.serialize(optimizer),
        loss=loss,
        metrics=metrics,
    )

    with patch.object(
        model_loader.model, "compile", wraps=model_loader.model.compile
    ) as mocked_compile, patch.object(tf.keras.optimizers, "get") as mock_get_optimizer:
        mock_get_optimizer.return_value = optimizer

        # Run training
        run(
            data_loader=data_loader,
            model_loader=model_loader,
            hyperparams=hyperparams,
            weights_serializer=weights_serializer,
            string_serializer=string_serializer,
        )

        mocked_compile.assert_called_with(
            optimizer=optimizer, loss=loss, metrics=metrics
        )


def test_run_does_not_overwrite_hyperparameters(
    model_loader: ModelLoaderStub,
    data_loader: DatasetLoaderStub,
    hyperparams,
    weights_serializer,
    string_serializer,
):
    model = model_loader.model
    original_metrics = copy(
        model.metrics
    )  # Needed as metrics are overwritten during compiling
    with patch.object(model, "compile", wraps=model.compile) as mocked_compile:
        run(
            data_loader=data_loader,
            model_loader=model_loader,
            hyperparams=hyperparams,
            weights_serializer=weights_serializer,
            string_serializer=string_serializer,
        )

        mocked_compile.assert_called_with(
            optimizer=model.optimizer, loss=model.loss, metrics=original_metrics
        )


@patch("fedless.client.DatasetLoaderBuilder")
@patch("fedless.client.ModelLoaderBuilder")
def test_default_handler_calls_run_correctly(
    model_builder,
    data_builder,
    hyperparams,
    dataset,
    model,
    weights_serializer,
    string_serializer,
):
    data_builder.from_config.return_value = data_loader = DatasetLoaderStub(dataset)
    model_builder.from_config.return_value = model_loader = ModelLoaderStub(model)
    with patch("fedless.client.run") as mock_run:
        # noinspection PyTypeChecker
        default_handler(None, None, hyperparams, weights_serializer, string_serializer)
        mock_run.assert_called_with(
            data_loader=data_loader,
            model_loader=model_loader,
            hyperparams=hyperparams,
            weights_serializer=weights_serializer,
            string_serializer=string_serializer,
        )


@pytest.mark.parametrize(
    "error",
    [
        NotImplementedError,
        DatasetNotLoadedError,
        ModelLoadError,
        RuntimeError,
        ValueError,
        SerializationError,
    ],
)
@patch("fedless.client.DatasetLoaderBuilder")
@patch("fedless.client.ModelLoaderBuilder")
@patch("fedless.client.run")
def test_default_handler_wraps_errors(
    mock_run, model_builder, data_builder, error, hyperparams
):
    # Mock loaders
    data_builder.from_config.return_value = data_loader = DatasetLoaderStub(dataset)
    model_builder.from_config.return_value = model_loader = ModelLoaderStub(model)

    # Set custom error message and make run throw it
    error_message = f"Error Message: {error.__name__}"
    mock_run.side_effect = error(error_message)

    with pytest.raises(ClientError, match=error_message):
        # noinspection PyTypeChecker
        default_handler(None, None, hyperparams, weights_serializer, string_serializer)


def test_http_client_result_reponse():
    result = ClientResult(weights="1234", history={"loss": [1.0]}, cardinality=12)
    response = create_http_success_response(result)

    assert response == {
        "statusCode": 200,
        "body": json.dumps(
            {"weights": "1234", "history": {"loss": [1.0]}, "cardinality": 12}
        ),
        "headers": {"Content-Type": "application/json"},
    }


@patch("traceback.format_exc")
def test_http_user_error_response(mock_format_exc):
    format_exc_mock_value = "Formatted exception information"
    mock_format_exc.return_value = format_exc_mock_value

    message = "Dataset did not load!"
    exception = ClientError(DatasetNotLoadedError(message))

    response = create_http_user_error_response(exception)

    assert response == {
        "statusCode": 400,
        "body": json.dumps(
            {
                "errorMessage": message,
                "errorType": str(ClientError.__name__),
                "details": format_exc_mock_value,
            }
        ),
        "headers": {"Content-Type": "application/json"},
    }


def test_lambda_proxy_decorator_returns_valid_response():
    def dummy_handler(event, context):
        return ClientResult(
            weights="1234", history={"loss": [0.0, 1.0]}, cardinality=12
        )

    patched_function = lambda_proxy_handler(dummy_handler)

    result_object = patched_function({}, {})
    assert result_object == {
        "statusCode": 200,
        "body": json.dumps(
            {"weights": "1234", "history": {"loss": [0.0, 1.0]}, "cardinality": 12}
        ),
        "headers": {"Content-Type": "application/json"},
    }


@patch("traceback.format_exc")
def test_lambda_proxy_decorator_returns_valid_error_dict(mock_format_exc):
    exception_message = "Dataset did not load!"
    format_exc_mock_value = "Formatted exception information"
    mock_format_exc.return_value = format_exc_mock_value

    def dummy_handler(event, context):
        raise ClientError(DatasetNotLoadedError(exception_message))

    patched_function = lambda_proxy_handler(dummy_handler)

    result_object = patched_function({}, {})
    assert result_object == {
        "statusCode": 400,
        "body": json.dumps(
            {
                "errorMessage": exception_message,
                "errorType": str(ClientError.__name__),
                "details": format_exc_mock_value,
            }
        ),
        "headers": {"Content-Type": "application/json"},
    }


def test_lambda_proxy_decorator_does_not_wrap_unknown_exception():
    def dummy_handler(event, context):
        raise MemoryError("fake memory error")

    patched_function = lambda_proxy_handler(dummy_handler)

    with pytest.raises(MemoryError, match="fake memory error"):
        patched_function({}, {})


def test_lambda_proxy_decorator_parses_body():
    def dummy_handler(event, context):
        assert isinstance(event, dict)
        assert isinstance(event["body"], dict)
        return ClientResult(weights="1234", history={"loss": [0.0, 1.0]})

    patched_function = lambda_proxy_handler(dummy_handler)
    patched_function({"body": '{"key": "value"}'}, {})


def test_gcloud_http_reponse():
    result = ClientResult(weights="1234", history={"loss": [1.0]}, cardinality=12)
    response = create_gcloud_http_success_response(result)

    assert response == (
        json.dumps({"weights": "1234", "history": {"loss": [1.0]}, "cardinality": 12}),
        200,
        {"Content-Type": "application/json"},
    )


@patch("traceback.format_exc")
def test_glcoud_http_user_error_response(mock_format_exc):
    format_exc_mock_value = "Formatted exception information"
    mock_format_exc.return_value = format_exc_mock_value

    message = "Dataset did not load!"
    exception = ClientError(DatasetNotLoadedError(message))

    response = create_gcloud_http_user_error_response(exception)

    assert response == (
        json.dumps(
            {
                "errorMessage": message,
                "errorType": str(ClientError.__name__),
                "details": format_exc_mock_value,
            }
        ),
        400,
        {"Content-Type": "application/json"},
    )


###################
# GCLOUD
###################
def test_gcloud_http_error_handler_decorator_returns_valid_response():
    def dummy_handler(request):
        return ClientResult(
            weights="1234", history={"loss": [0.0, 1.0]}, cardinality=12
        )

    patched_function = gcloud_http_error_handler(dummy_handler)

    result_object = patched_function(None)
    assert result_object == (
        json.dumps(
            {"weights": "1234", "history": {"loss": [0.0, 1.0]}, "cardinality": 12}
        ),
        200,
        {"Content-Type": "application/json"},
    )


@patch("traceback.format_exc")
def test_gcloud_http_error_handler_decorator_returns_valid_error_dict(mock_format_exc):
    exception_message = "Dataset did not load!"
    format_exc_mock_value = "Formatted exception information"
    mock_format_exc.return_value = format_exc_mock_value

    def dummy_handler(request):
        raise ClientError(DatasetNotLoadedError(exception_message))

    patched_function = gcloud_http_error_handler(dummy_handler)

    result_object = patched_function(None)
    assert result_object == (
        json.dumps(
            {
                "errorMessage": exception_message,
                "errorType": str(ClientError.__name__),
                "details": format_exc_mock_value,
            }
        ),
        400,
        {"Content-Type": "application/json"},
    )


def test_gcloud_http_error_handler_decorator_does_not_wrap_unknown_exception():
    def dummy_handler(request):
        raise MemoryError("fake memory error")

    patched_function = gcloud_http_error_handler(dummy_handler)

    with pytest.raises(MemoryError, match="fake memory error"):
        patched_function(None)


###################
# OPENWHISK
###################
def test_openwhisk_http_error_handler_decorator_returns_valid_response():
    def dummy_handler(params):
        return ClientResult(
            weights="1234", history={"loss": [0.0, 1.0]}, cardinality=12
        )

    patched_function = openwhisk_action_handler(dummy_handler)

    result_object = patched_function({})
    assert result_object == {
        "statusCode": 200,
        "body": json.dumps(
            {"weights": "1234", "history": {"loss": [0.0, 1.0]}, "cardinality": 12}
        ),
        "headers": {"Content-Type": "application/json"},
    }


@patch("traceback.format_exc")
def test_openwhisk_http_error_handler_decorator_returns_valid_error_dict(
    mock_format_exc,
):
    exception_message = "Dataset did not load!"
    format_exc_mock_value = "Formatted exception information"
    mock_format_exc.return_value = format_exc_mock_value

    def dummy_handler(request):
        raise ClientError(DatasetNotLoadedError(exception_message))

    patched_function = openwhisk_action_handler(dummy_handler)

    result_object = patched_function({})
    assert result_object == {
        "statusCode": 400,
        "body": json.dumps(
            {
                "errorMessage": exception_message,
                "errorType": str(ClientError.__name__),
                "details": format_exc_mock_value,
            }
        ),
        "headers": {"Content-Type": "application/json"},
    }


def test_openwhisk_http_error_handler_decorator_does_not_wrap_unknown_exception():
    def dummy_handler(request):
        raise MemoryError("fake memory error")

    patched_function = openwhisk_action_handler(dummy_handler)

    with pytest.raises(MemoryError, match="fake memory error"):
        patched_function({})


def test_openwhisk_web_action_handler_accepts_direct_invocation():
    def dummy_handler(params):
        assert isinstance(params, dict)
        assert params["body"] == {"key": "value"}
        return ClientResult(
            weights="1234", history={"loss": [0.0, 1.0]}, cardinality=12
        )

    patched_function = openwhisk_action_handler(dummy_handler)
    patched_function({"key": "value"})


def test_openwhisk_web_action_handler_accepts_empty_body():
    def dummy_handler(params):
        return ClientResult(
            weights="1234", history={"loss": [0.0, 1.0]}, cardinality=12
        )

    patched_function = openwhisk_action_handler(dummy_handler)
    patched_function({})


def test_openwhisk_web_action_handler_converts_web_request_body():
    def dummy_handler(params):
        assert isinstance(params, dict)
        assert params["body"] == {"key": "value"}
        return ClientResult(
            weights="1234", history={"loss": [0.0, 1.0]}, cardinality=12
        )

    patched_function = openwhisk_action_handler(dummy_handler)
    patched_function({"__ow_body": json.dumps({"key": "value"})})


def test_openwhisk_web_action_handler_converts_base64_encoded_web_request_body():
    def dummy_handler(params):
        assert isinstance(params, dict)
        assert params["body"] == {"key": "value"}
        return ClientResult(
            weights="1234", history={"loss": [0.0, 1.0]}, cardinality=12
        )

    patched_function = openwhisk_action_handler(dummy_handler)
    patched_function(
        {
            "__ow_body": base64.b64encode(
                bytes(json.dumps({"key": "value"}), encoding="utf-8")
            )
        }
    )
