import sys
from typing import Iterator

import click

from fedless.aggregation import (
    FedAvgAggregator,
    StreamFedAvgAggregator,
)
from fedless.benchmark.leaf import create_femnist_cnn
from fedless.models import (
    ClientResult,
    SerializedParameters,
    NpzWeightsSerializerConfig,
    WeightsSerializerConfig,
)
from fedless.serialization import NpzWeightsSerializer, Base64StringConverter


@click.command()
@click.option("--stream/--no-stream", default=False)
@click.option("--num-models", default=20)
@click.option("--large-models/--small-models", default=False)
def run(stream: bool, num_models: int, large_models: bool):
    def create_models() -> Iterator[ClientResult]:
        for i in range(num_models):
            params = create_femnist_cnn(small=not large_models).get_weights()
            weights_bytes = NpzWeightsSerializer().serialize(params)
            blob = Base64StringConverter.to_str(weights_bytes)
            yield ClientResult(
                parameters=SerializedParameters(
                    blob=blob,
                    serializer=WeightsSerializerConfig(
                        type="npz", params=NpzWeightsSerializerConfig()
                    ),
                ),
                cardinality=i,
            )

    aggregator = StreamFedAvgAggregator() if stream else FedAvgAggregator()

    final_params = aggregator.aggregate(create_models())
    print(f"Got final parameters!")


if __name__ == "__main__":
    run()
