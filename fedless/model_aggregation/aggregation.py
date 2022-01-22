import logging
from typing import Iterator, Optional, List

import pymongo
import tensorflow as tf

from fedless.datasets.dataset_loaders import DatasetLoaderBuilder
from fedless.models import (
    ClientResult,
    MongodbConnectionConfig,
    WeightsSerializerConfig,
    AggregatorFunctionResult,
    SerializedParameters,
    TestMetrics,
    DatasetLoaderConfig,
    SerializedModel,
)
from fedless.persistence import (
    ClientResultDao,
    ParameterDao,
    PersistenceError,
    ModelDao,
)
from fedless.serialization import (
    WeightsSerializerBuilder,
    SerializationError,
)
from fedless.model_aggregation.exceptions import (
    InsufficientClientResults,
    AggregationError,
)

from fedless.model_aggregation.fed_avg_aggregator import (
    FedAvgAggregator,
    StreamFedAvgAggregator,
)

logger = logging.getLogger(__name__)


def default_aggregation_handler(
    session_id: str,
    round_id: int,
    database: MongodbConnectionConfig,
    serializer: WeightsSerializerConfig,
    online: bool = False,
    test_data: Optional[DatasetLoaderConfig] = None,
    test_batch_size: int = 512,
    delete_results_after_finish: bool = True,
) -> AggregatorFunctionResult:
    mongo_client = pymongo.MongoClient(
        host=database.host,
        port=database.port,
        username=database.username,
        password=database.password,
    )
    logger.info(f"Aggregator invoked for session {session_id} and round {round_id}")
    try:

        result_dao = ClientResultDao(mongo_client)
        parameter_dao = ParameterDao(mongo_client)
        logger.debug(f"Establishing database connection")
        # TODO load all results in db not just in round
        previous_results: Iterator[ClientResult] = result_dao.load_results_for_round(
            session_id=session_id, round_id=round_id
        )

        if not previous_results:
            raise InsufficientClientResults(
                f"Found no client results for session {session_id} and round {round_id}"
            )
        aggregator = FedAvgAggregator()
        if online:
            logger.debug(f"Using online aggregation")
            aggregator = StreamFedAvgAggregator()
        else:
            logger.debug(f"Loading results from database...")
            previous_results = (
                list(previous_results)
                if not isinstance(previous_results, list)
                else previous_results
            )
            logger.debug(f"Loading of {len(previous_results)} results finished")
        logger.debug(f"Starting aggregation...")
        new_parameters, test_results = aggregator.aggregate(previous_results)
        logger.debug(f"Aggregation finished")

        global_test_metrics = None
        if test_data:
            logger.debug(f"Evaluating model")
            model_dao = ModelDao(mongo_client)
            # Load model and latest weights
            serialized_model: SerializedModel = model_dao.load(session_id=session_id)
            test_data = DatasetLoaderBuilder.from_config(test_data).load()
            cardinality = test_data.cardinality()
            test_data = test_data.batch(test_batch_size)
            model: tf.keras.Model = tf.keras.models.model_from_json(
                serialized_model.model_json
            )
            model.set_weights(new_parameters)
            if not serialized_model.loss or not serialized_model.optimizer:
                raise AggregationError("If compiled=True, a loss has to be specified")
            model.compile(
                optimizer=tf.keras.optimizers.get(serialized_model.optimizer),
                loss=tf.keras.losses.get(serialized_model.loss),
                metrics=serialized_model.metrics or [],
            )
            evaluation_result = model.evaluate(test_data, return_dict=True)
            global_test_metrics = TestMetrics(
                cardinality=cardinality, metrics=evaluation_result
            )

        logger.debug(f"Serializing model")
        serialized_params_str = WeightsSerializerBuilder.from_config(
            serializer
        ).serialize(new_parameters)

        serialized_params = SerializedParameters(
            blob=serialized_params_str, serializer=serializer
        )

        new_round_id = round_id + 1
        logger.debug(f"Saving model to database")
        parameter_dao.save(
            session_id=session_id, round_id=new_round_id, params=serialized_params
        )
        logger.debug(f"Finished...")

        results_processed = result_dao.count_results_for_round(
            session_id=session_id, round_id=round_id
        )
        if delete_results_after_finish:
            logger.debug(f"Deleting individual results...")
            result_dao.delete_results_for_round(
                session_id=session_id, round_id=round_id
            )

        return AggregatorFunctionResult(
            new_round_id=new_round_id,
            num_clients=results_processed,
            test_results=test_results,
            global_test_results=global_test_metrics,
        )
    except (SerializationError, PersistenceError) as e:
        raise AggregationError(e) from e
    finally:
        mongo_client.close()
