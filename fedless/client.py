import math
from typing import Optional

import pymongo
import tensorflow.keras as keras
from absl import app
from tensorflow.python.keras.callbacks import History
from tensorflow_privacy import (
    DPKerasAdamOptimizer,
    DPKerasAdagradOptimizer,
    DPKerasSGDOptimizer,
    compute_rdp,
)
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
    apply_dp_sgd_analysis,
)

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
    LocalPrivacyGuarantees,
    MongodbConnectionConfig,
    SimpleModelLoaderConfig,
    ClientInvocationParams,
    InvocationResult,
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
from fedless.persistence import (
    PersistenceError,
    ClientConfigDao,
    ModelDao,
    ParameterDao,
    ClientResultDao,
)


class ClientError(Exception):
    """Error in client code"""


def fedless_mongodb_handler(
    session_id: str,
    round_id: int,
    client_id: str,
    database: MongodbConnectionConfig,
):
    """
    Basic handler that only requires data and model loader configs plus hyperparams.
    Uses Npz weight serializer + Base64 encoding by default
    :raises ClientError if something failed during execution
    """
    db = pymongo.MongoClient(
        host=database.host,
        port=database.port,
        username=database.username,
        password=database.password,
    )

    try:
        # Create daos to access database
        config_dao = ClientConfigDao(db=db)
        model_dao = ModelDao(db=db)
        parameter_dao = ParameterDao(db=db)
        results_dao = ClientResultDao(db=db)

        # Load model and latest weights
        model = model_dao.load(session_id=session_id)
        latest_params = parameter_dao.load_latest(session_id)
        model = ModelLoaderConfig(
            type="simple",
            params=SimpleModelLoaderConfig(
                params=latest_params,
                model=model.model_json,
                compiled=True,
                optimizer=model.optimizer,
                loss=model.loss,
                metrics=model.metrics,
            ),
        )

        # Load client configuration and prepare call statements
        client_config = config_dao.load(id=client_id)
        client_params = ClientInvocationParams(
            data=client_config.data,
            model=model,
            hyperparams=client_config.hyperparams,
            test_data=client_config.test_data,
        )

        data_loader = DatasetLoaderBuilder.from_config(client_params.data)
        model_loader = ModelLoaderBuilder.from_config(client_params.model)
        test_data_loader = (
            DatasetLoaderBuilder.from_config(client_params.test_data)
            if client_params.test_data
            else None
        )
        weights_serializer: WeightsSerializer = NpzWeightsSerializer()
        string_serializer: StringSerializer = Base64StringConverter()
        verbose: bool = True
        client_result = run(
            data_loader=data_loader,
            model_loader=model_loader,
            hyperparams=client_params.hyperparams,
            weights_serializer=weights_serializer,
            string_serializer=string_serializer,
            test_data_loader=test_data_loader,
            verbose=verbose,
        )

        results_dao.save(
            session_id=session_id,
            round_id=round_id,
            client_id=client_id,
            result=client_result,
        )

        return InvocationResult(
            session_id=session_id,
            round_id=round_id,
            client_id=client_id,
        )

    except (
        NotImplementedError,
        DatasetNotLoadedError,
        ModelLoadError,
        RuntimeError,
        ValueError,
        SerializationError,
        PersistenceError,
    ) as e:
        raise ClientError(e) from e
    finally:
        db.close()


def default_handler(
    data_config: DatasetLoaderConfig,
    model_config: ModelLoaderConfig,
    hyperparams: Hyperparams,
    test_data_config: DatasetLoaderConfig = None,
    weights_serializer: WeightsSerializer = NpzWeightsSerializer(),
    string_serializer: StringSerializer = Base64StringConverter(),
    verbose: bool = True,
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
            verbose=verbose,
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
    verbose: bool = True,
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
    loss = keras.losses.get(hyperparams.loss) if hyperparams.loss else model.loss
    optimizer = (
        keras.optimizers.get(hyperparams.optimizer)
        if hyperparams.optimizer
        else model.optimizer
    )
    metrics = (
        hyperparams.metrics or model.compiled_metrics.metrics
    )  # compiled_metrics are explicitly defined by the user

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

    privacy_guarantees: Optional[LocalPrivacyGuarantees] = None
    if hyperparams.local_privacy:
        privacy_params = hyperparams.local_privacy
        opt_config = optimizer.get_config()
        opt_name = opt_config.get("name", "unknown")
        if opt_name == "Adam":
            optimizer = DPKerasAdamOptimizer(
                l2_norm_clip=privacy_params.l2_norm_clip,
                noise_multiplier=privacy_params.noise_multiplier,
                num_microbatches=privacy_params.num_microbatches,
                **opt_config,
            )
        elif opt_name == "Adagrad":
            optimizer = DPKerasAdagradOptimizer(
                l2_norm_clip=privacy_params.l2_norm_clip,
                noise_multiplier=privacy_params.noise_multiplier,
                num_microbatches=privacy_params.num_microbatches,
                **opt_config,
            )
        elif opt_name == "SGD":
            optimizer = DPKerasSGDOptimizer(
                l2_norm_clip=privacy_params.l2_norm_clip,
                noise_multiplier=privacy_params.noise_multiplier,
                num_microbatches=privacy_params.num_microbatches,
                **opt_config,
            )
        else:
            raise ValueError(
                f"No DP variant for optimizer {opt_name} found in TF Privacy..."
            )

        delta = 1.0 / (int(train_cardinality) ** 1.1)
        q = hyperparams.batch_size / train_cardinality  # q - the sampling ratio.
        if q > 1:
            raise app.UsageError("n must be larger than the batch size.")
        orders = (
            [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
            + list(range(5, 64))
            + [128, 256, 512]
        )
        steps = int(
            math.ceil(hyperparams.epochs * train_cardinality / hyperparams.batch_size)
        )
        eps, opt_order = apply_dp_sgd_analysis(
            q, privacy_params.noise_multiplier, steps, orders, delta
        )
        rdp = compute_rdp(
            q,
            noise_multiplier=privacy_params.noise_multiplier,
            steps=steps,
            orders=orders,
        )
        privacy_guarantees = LocalPrivacyGuarantees(
            eps=eps, delta=delta, rdp=rdp.tolist(), orders=orders, steps=steps
        )

        # Manually set loss' reduction method to None to support per-example loss calculation
        # Required to enable different microbatch sizes
        loss_serialized = keras.losses.serialize(keras.losses.get(loss))
        loss_name = (
            loss_serialized
            if isinstance(loss_serialized, str)
            else loss_serialized.get("config", dict()).get("name", "unknown")
        )
        if loss_name == "sparse_categorical_crossentropy":
            loss = keras.losses.SparseCategoricalCrossentropy(
                reduction=keras.losses.Reduction.NONE
            )
        elif loss_name == "categorical_crossentropy":
            loss = keras.losses.CategoricalCrossentropy(
                reduction=keras.losses.Reduction.NONE
            )
        elif loss_name == "mean_squared_error":
            loss = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)
        else:
            raise ValueError(f"Unkown loss type {loss_name}")

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Train Model
    # RuntimeError, ValueError
    history: History = model.fit(
        train_dataset,
        epochs=hyperparams.epochs,
        shuffle=hyperparams.shuffle_data,
        validation_data=val_dataset,
        verbose=verbose,
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
        privacy_guarantees=privacy_guarantees,
    )
