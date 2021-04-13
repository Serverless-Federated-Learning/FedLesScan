import tensorflow.keras as keras
from tensorflow.python.keras.callbacks import History

from fedless.data import (
    DatasetLoader,
    DatasetLoaderBuilder,
    DatasetNotLoadedError,
)
from fedless.models import (
    DatasetLoaderConfig,
    ModelLoaderConfig,
    Hyperparams,
    ClientResult,
    SerializedParameters,
    TestMetrics,
)
from fedless.serialization import (
    ModelLoadError,
    ModelLoader,
    ModelLoaderBuilder,
    WeightsSerializer,
    StringSerializer,
    NpzWeightsSerializer,
    Base64StringConverter,
    SerializationError,
)


class ClientError(Exception):
    """Error in client code"""


def default_handler(
    data_config: DatasetLoaderConfig,
    model_config: ModelLoaderConfig,
    hyperparams: Hyperparams,
    test_data_config: DatasetLoaderConfig = None,
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
    test_data_loader = (
        DatasetLoaderBuilder.from_config(test_data_config) if test_data_config else None
    )

    try:
        return run(
            data_loader=data_loader,
            model_loader=model_loader,
            hyperparams=hyperparams,
            weights_serializer=weights_serializer,
            string_serializer=string_serializer,
            test_data_loader=test_data_loader,
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
    validation_split: float = None,
    test_data_loader: DatasetLoader = None,
) -> ClientResult:
    """
    Loads model and data, trains the model and returns serialized parameters wrapped as :class:`ClientResult`

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
    if validation_split:
        cardinality = float(dataset.cardinality())
        train_validation_split_idx = int(cardinality - cardinality * validation_split)
        train_dataset = dataset.take(train_validation_split_idx)
        val_dataset = dataset.skip(train_validation_split_idx)
        train_dataset = train_dataset.batch(hyperparams.batch_size)
        val_dataset = val_dataset.batch(hyperparams.batch_size)
        train_cardinality = train_validation_split_idx
    else:
        train_dataset = dataset.batch(hyperparams.batch_size)
        train_cardinality = dataset.cardinality()
        val_dataset = None

    # Train Model
    # RuntimeError, ValueError
    history: History = model.fit(
        train_dataset,
        epochs=hyperparams.epochs,
        shuffle=hyperparams.shuffle_data,
        validation_data=val_dataset,
    )

    test_metrics = None
    if test_data_loader:
        test_dataset = test_data_loader.load()
        metrics = model.evaluate(
            test_dataset.batch(hyperparams.batch_size), return_dict=True
        )
        test_metrics = TestMetrics(
            cardinality=test_dataset.cardinality(), metrics=metrics
        )

    # serialization error
    weights_bytes = weights_serializer.serialize(model.get_weights())
    weights_string = string_serializer.to_str(weights_bytes)

    return ClientResult(
        parameters=SerializedParameters(
            blob=weights_string,
            serializer=weights_serializer.get_config(),
            string_format=string_serializer.get_format(),
        ),
        history=history.history,
        test_metrics=test_metrics,
        cardinality=train_cardinality,
    )