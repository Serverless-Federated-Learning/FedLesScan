import logging
import time

import click
import pandas as pd
from keras.utils.losses_utils import ReductionV2
from tensorflow import keras
from tensorflow_privacy import (
    DPKerasAdamOptimizer,
    VectorizedDPKerasAdamOptimizer,
)

from fedless.benchmark.fedkeeper import create_mnist_cnn
from fedless.client import logger
from fedless.data import DatasetLoaderBuilder
from fedless.models import (
    Hyperparams,
    DatasetLoaderConfig,
    MNISTConfig,
    LocalDifferentialPrivacyParams,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
def start():
    repeats = 10
    batch_size = 64
    model = create_mnist_cnn()

    data = DatasetLoaderConfig(
        type="mnist", params=MNISTConfig(indices=list(range(600)))
    )

    data_loader = DatasetLoaderBuilder.from_config(data)
    # Load data and model
    logger.debug(f"Loading dataset...")
    dataset = data_loader.load()
    logger.debug(f"Finished loading dataset. Loading model...")
    # Set configured optimizer if specified
    original_optimizer = keras.optimizers.Adam()
    original_model = model
    train_dataset = dataset.batch(batch_size)

    times = []
    for name, params in [
        (
            "none",
            Hyperparams(batch_size=batch_size, epochs=10, metrics=["accuracy"]),
        ),
        (
            "vanilla",
            Hyperparams(
                batch_size=batch_size,
                epochs=10,
                metrics=["accuracy"],
                local_privacy=LocalDifferentialPrivacyParams(
                    l2_norm_clip=1.0, noise_multiplier=1.0, num_microbatches=1
                ),
            ),
        ),
        (
            "vanilla-8-microbatches",
            Hyperparams(
                batch_size=batch_size,
                epochs=10,
                metrics=["accuracy"],
                local_privacy=LocalDifferentialPrivacyParams(
                    l2_norm_clip=1.0, noise_multiplier=1.0, num_microbatches=8
                ),
            ),
        ),
        (
            "vectorized",
            Hyperparams(
                batch_size=batch_size,
                epochs=10,
                metrics=["accuracy"],
                local_privacy=LocalDifferentialPrivacyParams(
                    l2_norm_clip=1.0, noise_multiplier=1.0, num_microbatches=1
                ),
            ),
        ),
        (
            "vectorized-8-microbatches",
            Hyperparams(
                batch_size=batch_size,
                epochs=10,
                metrics=["accuracy"],
                local_privacy=LocalDifferentialPrivacyParams(
                    l2_norm_clip=1.0, noise_multiplier=1.0, num_microbatches=8
                ),
            ),
        ),
    ]:
        for i in range(repeats):
            model = keras.models.clone_model(original_model)
            model.set_weights(original_model.get_weights())
            optimizer = original_optimizer
            start_time = time.time()
            if params.local_privacy:
                privacy_params = params.local_privacy
                opt_config = optimizer.get_config()
                if "vanilla" in name:
                    print(f"Using unoptimized dp variant")
                    optimizer = DPKerasAdamOptimizer(
                        l2_norm_clip=privacy_params.l2_norm_clip,
                        noise_multiplier=privacy_params.noise_multiplier,
                        num_microbatches=privacy_params.num_microbatches,
                        **opt_config,
                    )
                elif "vector" in name:
                    print(f"Using vectorized dp variant")
                    optimizer = VectorizedDPKerasAdamOptimizer(
                        l2_norm_clip=privacy_params.l2_norm_clip,
                        noise_multiplier=privacy_params.noise_multiplier,
                        num_microbatches=privacy_params.num_microbatches,
                        **opt_config,
                    )
                model.compile(
                    optimizer=optimizer,
                    loss=keras.losses.SparseCategoricalCrossentropy(
                        reduction=ReductionV2.NONE,
                    ),
                )
            else:
                model.compile(
                    loss="sparse_categorical_crossentropy", optimizer=optimizer
                )
            # Train Model
            # RuntimeError, ValueError
            model.fit(
                train_dataset,
                epochs=params.epochs,
                shuffle=params.shuffle_data,
                verbose=True,
            )
            duration = time.time() - start_time
            times.append({"name": name, "duration": duration})
            pd.DataFrame.from_records(times).to_csv("timings.csv")


if __name__ == "__main__":
    start()
