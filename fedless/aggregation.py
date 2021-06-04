import abc
from itertools import tee
from typing import Iterator, Optional, List, Tuple

import numpy as np
import pymongo
import tensorflow as tf

from fedless.models import (
    Parameters,
    ClientResult,
    MongodbConnectionConfig,
    WeightsSerializerConfig,
    AggregatorFunctionResult,
    SerializedParameters,
    TestMetrics,
)
from fedless.serialization import (
    deserialize_parameters,
    Base64StringConverter,
    WeightsSerializerBuilder,
    SerializationError,
)
from fedless.persistence import ClientResultDao, ParameterDao, PersistenceError


class AggregationError(Exception):
    pass


class InsufficientClientResults(AggregationError):
    pass


class UnknownCardinalityError(AggregationError):
    pass


class InvalidParameterShapeError(AggregationError):
    pass


def default_aggregation_handler(
    session_id: str,
    round_id: int,
    database: MongodbConnectionConfig,
    serializer: WeightsSerializerConfig,
    online: bool = False,
) -> AggregatorFunctionResult:
    mongo_client = pymongo.MongoClient(
        host=database.host,
        port=database.port,
        username=database.username,
        password=database.password,
    )
    try:

        result_dao = ClientResultDao(mongo_client)
        parameter_dao = ParameterDao(mongo_client)

        previous_results: Iterator[ClientResult] = result_dao.load_results_for_round(
            session_id=session_id, round_id=round_id
        )

        if not previous_results:
            raise InsufficientClientResults(
                f"Found no client results for session {session_id} and round {round_id}"
            )
        aggregator = FedAvgAggregator()
        if online:
            aggregator = StreamFedAvgAggregator()
        else:
            previous_results = (
                list(previous_results)
                if not isinstance(previous_results, list)
                else previous_results
            )
        new_parameters, test_results = aggregator.aggregate(previous_results)
        serialized_params_str = Base64StringConverter.to_str(
            WeightsSerializerBuilder.from_config(serializer).serialize(new_parameters)
        )

        serialized_params = SerializedParameters(
            blob=serialized_params_str, serializer=serializer
        )

        new_round_id = round_id + 1
        parameter_dao.save(
            session_id=session_id, round_id=new_round_id, params=serialized_params
        )

        return AggregatorFunctionResult(
            new_round_id=new_round_id,
            num_clients=result_dao.count_results_for_round(
                session_id=session_id, round_id=round_id
            ),
            test_results=test_results,
        )
    except (SerializationError, PersistenceError) as e:
        raise AggregationError(e) from e
    finally:
        mongo_client.close()


class ParameterAggregator(abc.ABC):
    @abc.abstractmethod
    def aggregate(
        self, client_results: Iterator[ClientResult]
    ) -> Tuple[Parameters, Optional[List[TestMetrics]]]:
        pass


class FedAvgAggregator(ParameterAggregator):
    def _aggregate(
        self, parameters: List[List[np.ndarray]], weights: List[float]
    ) -> List[np.ndarray]:
        return [
            np.average(params_for_layer, axis=0, weights=weights)
            for params_for_layer in zip(*parameters)
        ]

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
    def aggregate(
        self,
        client_results: Iterator[ClientResult],
        default_cardinality: Optional[float] = None,
    ) -> Tuple[Parameters, Optional[List[TestMetrics]]]:

        curr_global_params: Parameters = None
        curr_sum_weights = 0
        client_metrics: List[TestMetrics] = []
        for client_result in client_results:
            params = deserialize_parameters(client_result.parameters)
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

            if curr_global_params is None:
                curr_global_params = params
                curr_sum_weights = cardinality
            else:
                curr_global_params = self._aggregate(
                    [curr_global_params, params], [curr_sum_weights, cardinality]
                )
                curr_sum_weights += cardinality

            if client_result.test_metrics:
                client_metrics.append(client_result.test_metrics)

        return curr_global_params, client_metrics or None
