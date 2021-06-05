import os
import time

from pymongo_inmemory import MongoClient

from fedless.persistence import ClientResultDao
from fedless.models import (
    ClientResult,
    SerializedParameters,
    NpzWeightsSerializerConfig,
)
from fedless.serialization import (
    NpzWeightsSerializer,
    WeightsSerializerConfig,
    Base64StringConverter,
)
from fedless.benchmark.leaf import create_femnist_cnn
from fedless.benchmark.fedkeeper import create_mnist_cnn


def avg(l):
    return sum(l) / len(l)


def run():
    serialization_times = []
    saving_times = []
    loading_times = []
    with MongoClient() as client:
        dao = ClientResultDao(client)

        model = create_mnist_cnn()
        weights = model.get_weights()

        for _ in range(10):
            tic = time.perf_counter()
            serialized_params = SerializedParameters(
                blob=NpzWeightsSerializer(compressed=False).serialize(weights),
                serializer=WeightsSerializerConfig(
                    type="npz", params=NpzWeightsSerializerConfig(compressed=False)
                ),
            )
            toc = time.perf_counter()
            print(f"Serialization took {toc - tic} seconds")
            serialization_times.append(toc - tic)
            result = ClientResult(parameters=serialized_params, cardinality=123)

            tic = time.perf_counter()
            dao.save(
                session_id="123",
                round_id=0,
                client_id="123client",
                result=result.dict(),
            )
            toc = time.perf_counter()
            print(f"Saving took {toc - tic} seconds")
            saving_times.append(toc - tic)

            tic = time.perf_counter()
            list(dao.load_results_for_round(session_id="123", round_id=0))
            toc = time.perf_counter()
            print(f"Loading took {toc - tic} seconds")
            loading_times.append(toc - tic)

    print(f"Serialization: {avg(serialization_times)}")
    print(f"Saving: {avg(saving_times)}")
    print(f"Loading: {avg(loading_times)}")


if __name__ == "__main__":
    run()
