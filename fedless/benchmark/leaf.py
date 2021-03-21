from tensorflow import keras


def create_femnist_cnn(num_classes: int = 62):
    model = keras.Sequential()
    model.add(keras.layers.Input((28 * 28,)))
    model.add(keras.layers.Reshape((28, 28, 1)))
    model.add(
        keras.layers.Convolution2D(
            # filters=32, kernel_size=(5, 5), padding="same", activation="relu"
            filters=16,
            kernel_size=(5, 5),
            padding="same",
            activation="relu",
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(
        keras.layers.Convolution2D(
            # filters=64, kernel_size=(5, 5), padding="same", activation="relu"
            filters=16,
            kernel_size=(5, 5),
            padding="same",
            activation="relu",
        )
    )
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
