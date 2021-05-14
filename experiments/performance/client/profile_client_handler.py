import cProfile
import logging
import pstats
import time

import click

from fedless.benchmark.fedkeeper import create_mnist_cnn
from fedless.benchmark.leaf import create_shakespeare_lstm
from fedless.client import default_handler
from fedless.models import (
    Hyperparams,
    DatasetLoaderConfig,
    ModelLoaderConfig,
    MNISTConfig,
    SimpleModelLoaderConfig,
    SerializedParameters,
    WeightsSerializerConfig,
    NpzWeightsSerializerConfig,
    LocalDifferentialPrivacyParams,
    LEAFConfig,
)
from fedless.serialization import (
    Base64StringConverter,
    NpzWeightsSerializer,
    serialize_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--out", type=click.Path(), required=True)
@click.option("--ldp/--no-ldp", type=bool, default=False)
@click.option("--local-epochs", type=int, default=2)
@click.option("--runs", type=int, default=5)
@click.option("--dataset", type=str, default="mnist")
def run(out, ldp, local_epochs, runs, dataset):
    model = (
        create_mnist_cnn() if dataset.lower() == "mnist" else create_shakespeare_lstm()
    )
    weight_bytes = NpzWeightsSerializer().serialize(model.get_weights())
    weight_string = Base64StringConverter.to_str(weight_bytes)
    serialized_model = serialize_model(model)
    params = SerializedParameters(
        blob=weight_string,
        serializer=WeightsSerializerConfig(
            type="npz", params=NpzWeightsSerializerConfig()
        ),
    )
    if dataset.lower() == "mnist":
        data = DatasetLoaderConfig(
            type="mnist", params=MNISTConfig(indices=list(range(600)))
        )
    elif dataset.lower() == "shakespeare":
        data = DatasetLoaderConfig(
            type="leaf",
            params=LEAFConfig(
                dataset="shakespeare",
                location="http://138.246.235.163:31715/data/leaf/data/shakespeare/data/train/all_data_niid_0_keep_64_train_9.json",
                user_indices=[0],
            ),
        )
    else:
        raise NotImplementedError(f"{dataset} not implemented for benchmark")
    model = ModelLoaderConfig(
        type="simple",
        params=SimpleModelLoaderConfig(
            params=params,
            model=model.to_json(),
            compiled=True,
            optimizer="Adam",
            loss=serialized_model.loss,
            metrics=serialized_model.metrics,
        ),
    )
    start_time_ns = time.time_ns()
    # cProfile.run()
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(runs):
        result = default_handler(
            data_config=data,
            model_config=model,
            hyperparams=Hyperparams(
                batch_size=32,
                epochs=local_epochs,
                metrics=["accuracy"],
                local_privacy=(
                    LocalDifferentialPrivacyParams(
                        l2_norm_clip=2.0, noise_multiplier=1.0, num_microbatches=1
                    )
                )
                if ldp
                else None,
            ),
        )
        print(f"Model trained on {result.cardinality} samples per epoch")
    print(
        f"Execution took {((time.time_ns() - start_time_ns) / 10 ** 9) / runs} seconds"
    )
    profiler.disable()
    (
        pstats.Stats(profiler)
        .strip_dirs()
        .sort_stats("cumtime")
        .print_callees("client.py:.*(run)")
        .dump_stats(out)
    )


if __name__ == "__main__":
    run()