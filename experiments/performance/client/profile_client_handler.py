import cProfile
import logging
import pstats
import time

from fedless.benchmark.fedkeeper import create_mnist_cnn
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
)
from fedless.serialization import Base64StringConverter
from fedless.serialization import NpzWeightsSerializer, serialize_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
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

    data = DatasetLoaderConfig(
        type="mnist", params=MNISTConfig(indices=list(range(100)))
    )
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

    start_time_ns = time.time_ns()
    # cProfile.run()
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(2):
        result = default_handler(
            data_config=data,
            model_config=model,
            hyperparams=Hyperparams(batch_size=32, epochs=2, metrics=["accuracy"]),
        )
    profiler.disable()
    (
        pstats.Stats(profiler)
        .strip_dirs()
        .sort_stats("cumtime")
        .print_callees("client.py:.*(run)")
        .dump_stats("program.prof")
    )
    print(f"Execution took {(time.time_ns() - start_time_ns) / 10**9} seconds")
