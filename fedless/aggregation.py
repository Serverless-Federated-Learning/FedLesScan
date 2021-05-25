import abc
from typing import Iterator, Optional, List

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

        previous_results: List[ClientResult] = result_dao.load_results_for_round(
            session_id=session_id, round_id=round_id
        )

        if not previous_results:
            raise InsufficientClientResults(
                f"Found no client results for session {session_id} and round {round_id}"
            )

        new_parameters = FedAvgAggregator().aggregate(previous_results)
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

        test_results = [
            result.test_metrics
            for result in previous_results
            if result.test_metrics is not None
        ]

        return AggregatorFunctionResult(
            new_round_id=new_round_id,
            num_clients=result_dao.count_results_for_round(
                session_id=session_id, round_id=round_id
            ),
            test_results=test_results or None,
        )
    except (SerializationError, PersistenceError) as e:
        raise AggregationError(e) from e
    finally:
        mongo_client.close()


class ParameterAggregator(abc.ABC):
    @abc.abstractmethod
    def aggregate(self, client_results: Iterator[ClientResult]) -> Parameters:
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
    ) -> Parameters:

        client_parameters: List[List[np.ndarray]] = []
        client_cardinalities: List[int] = []
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

        return self._aggregate(client_parameters, client_cardinalities)


class StreamFedAvgAggregator(FedAvgAggregator):
    def aggregate(
        self,
        client_results: Iterator[ClientResult],
        default_cardinality: Optional[float] = None,
    ) -> Parameters:

        curr_global_params: Parameters = None
        curr_sum_weights = 0
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

        return curr_global_params
