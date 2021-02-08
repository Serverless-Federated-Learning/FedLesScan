import base64
import binascii
import json
import traceback
from json import JSONDecodeError
from typing import Optional, Dict, Union, List, Callable, Any, Tuple, Iterator

import pydantic
import tensorflow.keras as keras
from pydantic import Field, ValidationError
from tensorflow.python.keras.callbacks import History

from fedless.data import (
    DatasetLoader,
    DatasetLoaderBuilder,
    DatasetLoaderConfig,
    DatasetNotLoadedError,
)
from fedless.models import (
    ModelLoader,
    ModelLoaderConfig,
    ModelLoaderBuilder,
    ModelLoadError,
)
from fedless.serialization import (
    WeightsSerializer,
    StringSerializer,
    NpzWeightsSerializer,
    Base64StringConverter,
    SerializationError,
)


class Hyperparams(pydantic.BaseModel):
    """Parameters for training and some data processing"""

    batch_size: pydantic.PositiveInt
    epochs: pydantic.PositiveInt
    shuffle_data: bool = True
    optimizer: Optional[Union[str, Dict]] = Field(
        default=None,
        description="Optimizer, either string with name of optimizer or "
        "a config dictionary retrieved via tf.keras.optimizers.serialize. ",
    )
    loss: Optional[str] = Field(
        default=None,
        description="Name of loss function, see https://www.tensorflow.org/api_docs/python/tf/keras/losses",
    )
    metrics: Optional[List[str]] = Field(
        default=None,
        description="List of metrics to be evaluated by the model",
    )


class ClientConfig(pydantic.BaseModel):
    """Convenience class to directly parse and serialize loaders and hyperparameters"""

    data: DatasetLoaderConfig
    model: ModelLoaderConfig
    hyperparams: Hyperparams


class ClientResult(pydantic.BaseModel):
    """Result of client function execution"""

    weights: str
    history: Optional[Dict]
    cardinality: int = Field(
        description="tf.data.INFINITE_CARDINALITY if the dataset contains an infinite number of elements or "
        "tf.data.UNKNOWN_CARDINALITY if the analysis fails to determine the number of elements in the dataset "
        "(e.g. when the dataset source is a file). "
        "Source: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cardinality"
    )


class ClientError(Exception):
    """Error in client code"""


def default_handler(
    data_config: DatasetLoaderConfig,
    model_config: ModelLoaderConfig,
    hyperparams: Hyperparams,
    weights_serializer: WeightsSerializer = NpzWeightsSerializer(),
    string_serializer: StringSerializer = Base64StringConverter(),
) -> ClientResult:
    """
    Basic handler that only requires data and model loader configs plus hyperparams.
    Uses Npz weight serializer + Base64 encoding by default
    :raises ClientError if something failed during execution
    """
    data_loader = DatasetLoaderBuilder.from_config(data_config)
    model_loader = ModelLoaderBuilder.from_config(model_config)

    try:
        return run(
            data_loader=data_loader,
            model_loader=model_loader,
            hyperparams=hyperparams,
            weights_serializer=weights_serializer,
            string_serializer=string_serializer,
        )
    except (
        NotImplementedError,
        DatasetNotLoadedError,
        ModelLoadError,
        RuntimeError,
        ValueError,
        SerializationError,
    ) as e:
        raise ClientError(e) from e


def run(
    data_loader: DatasetLoader,
    model_loader: ModelLoader,
    hyperparams: Hyperparams,
    weights_serializer: WeightsSerializer,
    string_serializer: StringSerializer,
) -> ClientResult:
    """
    Loads model and data, trains the model and returns serialized weights wrapped as :class:`ClientResult`

    :raises DatasetNotLoadedError, ModelLoadError, RuntimeError if the model was never compiled,
     ValueError if input data is invalid or shape does not match the one expected by the model, SerializationError
    """
    # Load data and model
    dataset = data_loader.load()
    model = model_loader.load()

    # Set configured optimizer if specified
    loss = hyperparams.loss or model.loss
    optimizer = (
        keras.optimizers.get(hyperparams.optimizer)
        if hyperparams.optimizer
        else model.optimizer
    )
    metrics = (
        hyperparams.metrics or model.compiled_metrics.metrics
    )  # compiled_metrics are explicitly defined by the user
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Batch data, necessary or model fitting will fail
    dataset = dataset.batch(hyperparams.batch_size)

    # Train Model
    # RuntimeError, ValueError
    history: History = model.fit(
        dataset, epochs=hyperparams.epochs, shuffle=hyperparams.shuffle_data
    )

    # serialization error
    weights_bytes = weights_serializer.serialize(model.get_weights())
    weights_string = string_serializer.to_str(weights_bytes)

    return ClientResult(
        weights=weights_string,
        history=history.history,
        cardinality=dataset.cardinality(),
    )


def create_http_success_response(result: ClientResult, status: int = 200):
    """Creates successful response compatible with API gateway lambda-proxy integration"""
    return {
        "statusCode": status,
        "body": result.json(),
        "headers": {"Content-Type": "application/json"},
    }


def format_exception_for_user(exception: Exception) -> Dict:
    """Create dictionary with information about the exception to be returned to a user"""
    return {
        "errorMessage": str(exception),
        "errorType": str(exception.__class__.__name__),
        "details": traceback.format_exc(),
    }


def create_http_user_error_response(exception: Exception, status: int = 400):
    """Create error response for given exception. Compatible with API gateway lambda-proxy integration"""
    return {
        "statusCode": status,
        "body": json.dumps(format_exception_for_user(exception)),
        "headers": {"Content-Type": "application/json"},
    }


def lambda_proxy_handler(
    func: Callable[[Dict, Any], ClientResult]
) -> Callable[[Dict, Any], Dict]:
    """
    Creates response objects from results that are compatible with API gateway's lambda-proxy integration.
    If necessary parses and validates event object
    :param func: Lambda handler with only difference that it should return :class:`ClientResult` directly.
    :return: Fully valid lambda handler function
    """

    def patched_func(event, context):
        try:
            if "body" in event and isinstance(event["body"], str):
                event["body"] = json.loads(event["body"])
            result: ClientResult = func(event, context)
            return create_http_success_response(result)
        except (ValidationError, ClientError) as e:
            return create_http_user_error_response(e)

    return patched_func


def create_gcloud_http_success_response(
    result: ClientResult, status: int = 200
) -> Tuple[Union[str, bytes, dict], int, Union[Dict, List]]:
    """
    Create object that can be converted into Flask response object by Google Cloud API
    See https://flask.palletsprojects.com/en/1.1.x/api/#flask.Flask.make_response for more info
    """
    return result.json(), status, {"Content-Type": "application/json"}


def create_gcloud_http_user_error_response(
    exception: Exception, status: int = 400
) -> Tuple[Union[str, bytes, dict], int, Union[Dict, List]]:
    """
    Create object from exception that can be converted into Flask response object by Google Cloud API
    See https://flask.palletsprojects.com/en/1.1.x/api/#flask.Flask.make_response for more info
    """
    return (
        json.dumps(format_exception_for_user(exception)),
        status,
        {"Content-Type": "application/json"},
    )


# noinspection PyUnresolvedReferences
def gcloud_http_error_handler(
    func: Callable[["flask.Request"], ClientResult]
) -> Callable[["flask.Request"], Dict]:
    """
    Creates response tuples from results that are compatible with gcloud's flask integration.
    :param func: Gcloud http handler with only difference that it should return :class:`ClientResult` directly.
    :return: Fully valid gcloud handler function
    """

    def patched_func(*args, **kwargs):
        try:
            result: ClientResult = func(*args, **kwargs)
            return create_gcloud_http_success_response(result)
        except (ValidationError, ClientError) as e:
            return create_gcloud_http_user_error_response(e)

    return patched_func


def openwhisk_action_handler(
    func: Callable[[Dict], ClientResult]
) -> Callable[[Dict], Dict]:
    """
    Creates response objects from results that are compatible with Openwhisk's Web Actions and direct invocation
    :param func: Openwhisk function handler with only difference that it should return :class:`ClientResult` directly.
    :return: Fully valid openwhisk handler function
    """

    def patched_func(params: Dict):
        # Put parameters or request body under key "body". Strip __ow_ prefix for web action support
        # See https://github.com/apache/openwhisk/blob/master/docs/webactions.md#http-context for more info
        if any(map(lambda name: name.startswith("__ow_"), params.keys())):
            params = {key[len("__ow_") :]: value for key, value in params.items()}
        else:
            params = {"body": params}

        try:
            if isinstance(params["body"], (str, bytes)):
                # Openwhisk sometimes base64 encodes the body
                try:
                    params["body"] = json.loads(params["body"])
                except JSONDecodeError:
                    params["body"] = json.loads(base64.b64decode(params["body"]))

            result: ClientResult = func(params)
            return create_http_success_response(result)
        except (ValidationError, ClientError, JSONDecodeError, binascii.Error) as e:
            return create_http_user_error_response(e)

    return patched_func
