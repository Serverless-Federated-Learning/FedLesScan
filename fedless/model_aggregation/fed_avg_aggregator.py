import abc
import logging
from functools import reduce
from typing import Iterator, Optional, List, Tuple

import numpy as np
import tensorflow as tf

from fedless.model_aggregation.exceptions import UnknownCardinalityError
from fedless.models import Parameters, ClientResult, TestMetrics

from fedless.serialization import deserialize_parameters

from parameter_aggregator import ParameterAggregator


class FedAvgAggregator(ParameterAggregator):
    def _aggregate(
        self, parameters: List[List[np.ndarray]], weights: List[float]
    ) -> List[np.ndarray]:
        # Partially from https://github.com/adap/flower/blob/
        # 570788c9a827611230bfa78f624a89a6630555fd/src/py/flwr/server/strategy/aggregate.py#L26
        num_examples_total = sum(weights)
        weighted_weights = [
            [layer * num_examples for layer in weights]
            for weights, num_examples in zip(parameters, weights)
        ]

        # noinspection PydanticTypeChecker,PyTypeChecker
        weights_prime: List[np.ndarray] = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime

    def aggregate(
        self,
        client_results: Iterator[ClientResult],
        default_cardinality: Optional[float] = None,
    ) -> Tuple[Parameters, Optional[List[TestMetrics]]]:

        client_parameters: List[List[np.ndarray]] = []
        client_cardinalities: List[int] = []
        client_metrics: List[TestMetrics] = []
        for client_result in client_results:
            params = deserialize_parameters(client_result.parameters)
            del client_result.parameters
            cardinality = client_result.cardinality

            # Check if cardinality is valid and handle accordingly
            if cardinality in [
                tf.data.UNKNOWN_CARDINALITY,
                tf.data.INFINITE_CARDINALITY,
            ]:
                if not default_cardinality:
                    raise UnknownCardinalityError(
                        f"Cardinality for client result invalid. "
                    )
                else:
                    cardinality = default_cardinality

            client_parameters.append(params)
            client_cardinalities.append(cardinality)
            if client_result.test_metrics:
                client_metrics.append(client_result.test_metrics)

        return (
            self._aggregate(client_parameters, client_cardinalities),
            client_metrics or None,
        )


class StreamFedAvgAggregator(FedAvgAggregator):
    def __init__(self, chunk_size: int = 25):
        self.chunk_size = chunk_size

    def chunks(self, iterator: Iterator, n) -> Iterator[List]:
        buffer = []
        for el in iterator:
            if len(buffer) < n:
                buffer.append(el)
            if len(buffer) == n:
                yield buffer
                buffer = []
        else:
            if len(buffer) > 0:
                yield buffer

    def aggregate(
        self,
        client_results: Iterator[ClientResult],
        default_cardinality: Optional[float] = None,
    ) -> Tuple[Parameters, Optional[List[TestMetrics]]]:

        curr_global_params: Parameters = None
        curr_sum_weights = 0
        client_metrics: List[TestMetrics] = []
        for results_chunk in self.chunks(client_results, self.chunk_size):
            params_buffer, card_buffer = [], []
            for client_result in results_chunk:
                params = deserialize_parameters(client_result.parameters)
                del client_result.parameters
                cardinality = client_result.cardinality

                # Check if cardinality is valid and handle accordingly
                if cardinality in [
                    tf.data.UNKNOWN_CARDINALITY,
                    tf.data.INFINITE_CARDINALITY,
                ]:
                    if not default_cardinality:
                        raise UnknownCardinalityError(
                            f"Cardinality for client result invalid. "
                        )
                    else:
                        cardinality = default_cardinality

                params_buffer.append(params)
                card_buffer.append(cardinality)
                if client_result.test_metrics:
                    client_metrics.append(client_result.test_metrics)
            if curr_global_params is None:
                curr_global_params = self._aggregate(params_buffer, card_buffer)
            else:
                curr_global_params = self._aggregate(
                    [curr_global_params, *params_buffer],
                    [curr_sum_weights, *card_buffer],
                )
            curr_sum_weights += sum(card_buffer)

        return curr_global_params, client_metrics or None
