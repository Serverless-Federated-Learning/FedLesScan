from typing import Optional
from build.lib.fedless.models import AggregatorFunctionParams

from fedless.model_aggregation.aggregation import default_aggregation_handler

from fedless.models import (
    MongodbConnectionConfig,
    WeightsSerializerConfig,
    DatasetLoaderConfig,
)


class MockAggregator:
    def __init__(
        self,
        params: AggregatorFunctionParams,
        delete_results_after_finish: bool = True,
    ):
        self.session_id = params.session_id
        self.round_id = params.round_id
        self.database = params.database
        self.serializer = params.serializer
        self.online = params.online
        self.test_data = params.test_data
        self.test_batch_size = params.test_batch_size
        self.delete_results_after_finish = delete_results_after_finish

    def run_aggregator(self):
        return default_aggregation_handler(
            self.session_id,
            self.round_id,
            self.database,
            self.serializer,
            self.online,
            self.test_data,
            self.test_batch_size,
            self.delete_results_after_finish,
        )