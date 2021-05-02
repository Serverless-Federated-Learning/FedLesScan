from typing import Iterable, Optional, Dict

from tensorflow import keras

from fedless.data import LEAF
from fedless.models import LeafDataset, LEAFConfig


def create_femnist_cnn(num_classes: int = 62, small: bool = False):
    model = keras.Sequential()
    model.add(keras.layers.Input((28 * 28,)))
    model.add(keras.layers.Reshape((28, 28, 1)))
    model.add(
        keras.layers.Convolution2D(
            filters=(16 if small else 32),
            kernel_size=(5, 5),
            padding="same",
            activation="relu",
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(
        keras.layers.Convolution2D(
            filters=(32 if small else 64),
            kernel_size=(5, 5),
            padding="same",
            activation="relu",
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense((512 if small else 2048), activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def create_shakespeare_lstm(
    units: int = 256,
    vocab_size: int = 82,
    sequence_length: int = 80,
    embedding_size: int = 8,
):
    model = keras.Sequential()
    keras.Input(shape=(sequence_length, vocab_size))
    model.add(
        keras.layers.Embedding(
            vocab_size,
            embedding_size,
        )
    )
    model.add(keras.layers.LSTM(units, return_sequences=True))
    model.add(keras.layers.LSTM(units))
    model.add(keras.layers.Dense(vocab_size, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def split_shakespear_source_by_users(
    url: str, http_params: Optional[Dict]
) -> Iterable[LEAFConfig]:
    loader = LEAF(dataset=LeafDataset.SHAKESPEARE, location=url)
    loader.load()

    for i, _ in enumerate(loader.users):
        yield LEAFConfig(
            dataset=LeafDataset.SHAKESPEARE,
            location=url,
            http_params=http_params,
            user_indices=[i],
        )
