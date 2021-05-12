import logging

import tensorflow as tf

import pandas as pd
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import (
    compute_dp_sgd_privacy,
)

from fedless.benchmark.fedkeeper import create_mnist_cnn
from fedless.client import run
from fedless.data import DatasetLoaderBuilder
from fedless.models import (
    Hyperparams,
    DatasetLoaderConfig,
    ModelLoaderConfig,
    MNISTConfig,
    SimpleModelLoaderConfig,
    WeightsSerializerConfig,
    NpzWeightsSerializerConfig,
    LocalDifferentialPrivacyParams,
)
from fedless.serialization import Base64StringConverter
from fedless.serialization import (
    NpzWeightsSerializer,
    serialize_model,
    ModelLoaderBuilder,
)
from fedless.models import SerializedParameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_exp(
    noise_multiplier, l2_norm_clip, batch_size, epochs, num_microbatches, n, delta
):
    model = create_mnist_cnn()
    weight_bytes = NpzWeightsSerializer().serialize(model.get_weights())
    weight_string = Base64StringConverter.to_str(weight_bytes)
    serialized_model = serialize_model(model)
    params = SerializedParameters(
        blob=weight_string,
        serializer=WeightsSerializerConfig(
            type="npz", params=NpzWeightsSerializerConfig()
        ),
    )
    data = DatasetLoaderConfig(type="mnist", params=MNISTConfig(indices=list(range(n))))
    test_data_config = DatasetLoaderConfig(
        type="mnist", params=MNISTConfig(split="test")
    )
    keras_model = model
    model = ModelLoaderConfig(
        type="simple",
        params=SimpleModelLoaderConfig(
            params=params,
            model=model.to_json(),
            compiled=True,
            optimizer=serialized_model.optimizer,
            loss=serialized_model.loss,
            metrics=serialized_model.metrics,
        ),
    )

    result = run(
        data_loader=DatasetLoaderBuilder.from_config(data),
        model_loader=ModelLoaderBuilder.from_config(model),
        hyperparams=Hyperparams(
            batch_size=batch_size,
            epochs=epochs,
            # loss="sparse_categorical_crossentropy",
            # optimizer="adam",
            local_privacy=LocalDifferentialPrivacyParams(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
            ),
        ),
        weights_serializer=NpzWeightsSerializer(),
        string_serializer=Base64StringConverter(),
        test_data_loader=DatasetLoaderBuilder.from_config(test_data_config),
    )

    print(result.history)
    print(result.test_metrics)
    print(result.privacy_guarantees)
    eps, opt_order = compute_dp_sgd_privacy(
        n=n,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        epochs=epochs,
        delta=delta,
    )
    # parameters = deserialize_parameters(result.parameters)
    # keras_model.set_weights(parameters)
    # train, test = data_loader.load(), test_data_loader.load()
    # x_train = train.map(lambda x, y: x)
    # x_test = test.map(lambda x, y: x)
    # y_train = list(train.map(lambda x, y: y).as_numpy_iterator())
    # y_test = list(test.map(lambda x, y: y).as_numpy_iterator())
    # train_predict = keras_model.predict(x_train.batch(128))
    # test_predict = keras_model.predict(x_test.batch(128))
    # loss_train = keras.losses.sparse_categorical_crossentropy(
    #    y_train, train_predict
    # ).numpy()
    # loss_test = keras.losses.sparse_categorical_crossentropy(
    #    y_test, test_predict
    # ).numpy()
    # labels_train = y_train  # .argmax(axis=1)
    # labels_test = y_test  # .argmax(axis=1)
    #
    # attacks_result = mia.run_attacks(
    #    AttackInputData(
    #        loss_train=loss_train,
    #        loss_test=loss_test,
    #        labels_train=np.asanyarray(labels_train),
    #        labels_test=np.asanyarray(labels_test),
    #    )
    # )

    return result, eps  # , attacks_result.summary()


if __name__ == "__main__":
    results = []

    learning_rate = 0.25
    noise_multiplier = 1.0
    l2_norm_clip = 5.0
    batch_size = 10
    epochs = 10
    num_microbatches = 1  # 256 error
    n = 1000

    # for epochs in [epochs]:
    #    for noise_multiplier in [0.01, 0.5, 1.0, 2.0, 10.0]:
    #        for l2_norm_clip in [0.1, 1.0, 2.0, 5.0, 10.0]:
    for epochs in [10]:
        for noise_multiplier in reversed([0.01, 0.5, 1.0, 2.0, 10.0]):
            for l2_norm_clip in [0.1, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]:
                for num_microbatches in [1, 5]:
                    for delta in [1.0 / n, 1.0 / (float(n) ** 1.1)]:
                        for n in [100, 500, 1000]:
                            result, eps = run_exp(
                                noise_multiplier=noise_multiplier,
                                l2_norm_clip=l2_norm_clip,
                                batch_size=batch_size,
                                epochs=epochs,
                                num_microbatches=num_microbatches,
                                n=n,
                                delta=delta,
                            )
                            results.append(
                                {
                                    "learning_rate": learning_rate,
                                    "noise_multiplier": noise_multiplier,
                                    "l2_norm_clip": l2_norm_clip,
                                    "batch_size": batch_size,
                                    "epochs": epochs,
                                    "num_microbatches": num_microbatches,
                                    "n": n,
                                    "test_metrics": result.test_metrics,
                                    "history": result.history,
                                    "cardinality": result.cardinality,
                                    "eps": eps,
                                    "delta": delta,
                                }
                            )

                            pd.DataFrame.from_records(results).to_csv("results.csv")

                            # print(result)
                            print(epochs, noise_multiplier)
                            print(eps)
                        # print(results)
