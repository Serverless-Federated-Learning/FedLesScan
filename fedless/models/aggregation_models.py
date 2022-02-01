from fedless.models.models import *

class AggregationStrategy(str, Enum):
    PER_ROUND = "per_round"
    PER_SESSION = 'per_session'
    
class AggregatorFunctionParams(BaseModel):
    session_id: str
    round_id: int
    database: MongodbConnectionConfig
    serializer: WeightsSerializerConfig = WeightsSerializerConfig(
        type="npz", params=NpzWeightsSerializerConfig(compressed=False)
    )
    online: bool = False
    test_data: Optional[DatasetLoaderConfig]
    test_batch_size: int = 512
    aggregation_strategy: AggregationStrategy = AggregationStrategy.PER_ROUND

class AggregatorFunctionResult(BaseModel):
    new_round_id: int
    num_clients: int
    test_results: Optional[List[TestMetrics]]
    global_test_results: Optional[TestMetrics]
