from enum import Enum

from fedless.common.models.models import (
    MongodbConnectionConfig,
    WeightsSerializerConfig,
    NpzWeightsSerializerConfig,
    DatasetLoaderConfig,
    TestMetrics,
)
from pydantic import BaseModel
from typing import Optional, List


class AggregationStrategy(str, Enum):
    PER_ROUND = "per_round"
    PER_SESSION = "per_session"  # enhanced with staleness aware aggregation


class AggregationHyperParams(BaseModel):
    tolerance: int = 0
    aggregate_online: bool = False
    test_batch_size: int = 10


class AggregatorFunctionParams(BaseModel):
    session_id: str
    round_id: int
    database: MongodbConnectionConfig
    serializer: WeightsSerializerConfig = WeightsSerializerConfig(
        type="npz", params=NpzWeightsSerializerConfig(compressed=False)
    )
    test_data: Optional[DatasetLoaderConfig]
    aggregation_hyper_params: AggregationHyperParams
    aggregation_strategy: AggregationStrategy = AggregationStrategy.PER_ROUND


class AggregatorFunctionResult(BaseModel):
    new_round_id: int
    num_clients: int
    test_results: Optional[List[TestMetrics]]
    global_test_results: Optional[TestMetrics]
