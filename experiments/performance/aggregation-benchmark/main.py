import random
import timeit
from functools import reduce
from typing import List

import numpy as np

from fedless.benchmark.leaf import create_femnist_cnn


def aggregate(
    parameters: List[List[np.ndarray]], weights: List[float]
) -> List[np.ndarray]:
    return [
        np.average(params_for_layer, axis=0, weights=weights)
        for params_for_layer in zip(*parameters)
    ]


def aggregate_flower(
    parameters: List[List[np.ndarray]], weights: List[float]
) -> List[List[np.ndarray]]:
    num_examples_total = sum(weights)
    weighted_weights = [
        [layer * num_examples for layer in weights]
        for weights, num_examples in zip(parameters, weights)
    ]

    weights_prime = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


if __name__ == "__main__":
    clients = 100
    weights = [random.randint(1, 32) for _ in range(clients)]
    parameters = [create_femnist_cnn().get_weights() for _ in range(clients)]

    agg = aggregate(parameters, weights)
    time_non_jit = timeit.timeit(lambda: aggregate(parameters, weights), number=10)
    print(f"Vanilla: {time_non_jit}")
    time_flower = timeit.timeit(
        lambda: aggregate_flower(parameters, weights), number=10
    )
    print(f"Flower: {time_flower}")
