import logging

from pydantic import ValidationError


from fedless.aggregator.aggregation import (
    default_aggregation_handler,
    AggregationError,
)
from fedless.common.models import AggregatorFunctionParams
from fedless.common.providers import openfaas_action_handler

logging.basicConfig(level=logging.DEBUG)


@openfaas_action_handler(caught_exceptions=(ValidationError, AggregationError))
def handle(event, context):
    config = AggregatorFunctionParams.parse_raw(event.body)

    return default_aggregation_handler(
        session_id=config.session_id,
        round_id=config.round_id,
        database=config.database,
        serializer=config.serializer,
        online=config.online,
        test_data=config.test_data,
        test_batch_size=config.test_batch_size,
        aggregation_strategy=config.aggregation_strategy,
    )
