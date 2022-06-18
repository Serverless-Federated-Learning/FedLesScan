import logging
from typing import Optional

import pymongo
import tensorflow as tf

from fedless.datasets.benchmark_configurator import DatasetLoaderBuilder
from fedless.aggregator.stall_aware_aggregation import (
    StallAwareAggregator,
    StreamStallAwareAggregator,
)
from fedless.common.models import (
    MongodbConnectionConfig,
    WeightsSerializerConfig,
    AggregatorFunctionResult,
    SerializedParameters,
    TestMetrics,
    DatasetLoaderConfig,
    SerializedModel,
)
from fedless.common.models.aggregation_models import (
    AggregationHyperParams,
    AggregationStrategy,
)
from fedless.common.persistence import (
    ClientResultDao,
    ParameterDao,
    PersistenceError,
    ModelDao,
)
from fedless.common.serialization import (
    WeightsSerializerBuilder,
    SerializationError,
)
from fedless.aggregator.exceptions import AggregationError

from fedless.aggregator.fed_avg_aggregator import (
    FedAvgAggregator,
    StreamFedAvgAggregator,
)

logger = logging.getLogger(__name__)


def default_aggregation_handler(
    session_id: str,
    round_id: int,
    database: MongodbConnectionConfig,
    serializer: WeightsSerializerConfig,
    test_data: Optional[DatasetLoaderConfig] = None,
    delete_results_after_finish: bool = True,
    aggregation_strategy: AggregationStrategy = AggregationStrategy.PER_ROUND,
    aggregation_hyper_params: AggregationHyperParams = None,
) -> AggregatorFunctionResult:

    # mongo_client = pymongo.MongoClient(
    #     host=database.host,
    #     port=database.port,
    #     username=database.username,
    #     password=database.password,
    # )

    mongo_client = pymongo.MongoClient(database.url)
    logger.info(f"Aggregator invoked for session {session_id} and round {round_id}")
    try:

        result_dao = ClientResultDao(mongo_client)
        parameter_dao = ParameterDao(mongo_client)
        # logger.debug(f"Establishing database connection")
        # aggregator = FedAvgAggregator()
        aggregator = (
            StallAwareAggregator(round_id, aggregation_hyper_params)
            if aggregation_strategy == AggregationStrategy.PER_SESSION
            else FedAvgAggregator()
        )
        previous_dic, previous_results = aggregator.select_aggregation_candidates(
            mongo_client, session_id, round_id
        )
        if aggregation_hyper_params.aggregate_online:
            logger.debug(f"Using online aggregation")
            aggregator = (
                StreamStallAwareAggregator(round_id, aggregation_hyper_params)
                if aggregation_strategy == AggregationStrategy.PER_SESSION
                else StreamFedAvgAggregator()
            )
        else:
            logger.debug(f"Loading results from database...")
            previous_results = (
                list(previous_results)
                if not isinstance(previous_results, list)
                else previous_results
            )
            logger.debug(f"Loading of {len(previous_results)} results finished")
        logger.debug(f"Starting aggregation...")
        new_parameters, test_results = aggregator.aggregate(
            previous_results, previous_dic
        )
        logger.debug(f"Aggregation finished")

        global_test_metrics = None
        if test_data:
            logger.debug(f"Evaluating model")
            model_dao = ModelDao(mongo_client)
            # Load model and latest weights
            serialized_model: SerializedModel = model_dao.load(session_id=session_id)
            test_data = DatasetLoaderBuilder.from_config(test_data).load()
            cardinality = test_data.cardinality()
            test_data = test_data.batch(aggregation_hyper_params.test_batch_size)
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

        # cleanup and count results
        delete_func = result_dao.delete_results_for_round
        count_func = result_dao.count_results_for_round
        cleanup_params = {"session_id": session_id, "round_id": round_id}
        if aggregation_strategy == AggregationStrategy.PER_SESSION:
            delete_func = result_dao.delete_results_for_session
            count_func = result_dao.count_results_for_session
            cleanup_params.pop("round_id", None)

        results_processed = count_func(**cleanup_params)
        if delete_results_after_finish:
            logger.debug(f"Deleting individual results...")
            delete_func(**cleanup_params)
            # result_dao.delete_results_for_round(
            #     session_id=session_id, round_id=round_id
            # )

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
