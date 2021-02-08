import pytest
import tensorflow as tf


@pytest.fixture
def simple_model() -> tf.keras.Model:
    """Used in test_serialize.py"""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(5, input_shape=(3,)),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Softmax(),
        ]
    )
    model.compile(loss="mse", optimizer="sgd")
    return model
