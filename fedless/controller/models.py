from typing import Optional, List

import pydantic

from fedless.common.models import (
    FunctionDeploymentConfig,
    FunctionInvocationConfig,
    Hyperparams,
    MongodbConnectionConfig,
    AggregationHyperParams
)


class ClientFunctionConfig(pydantic.BaseModel):
    function: FunctionInvocationConfig
    hyperparams: Optional[Hyperparams]
    replicas: int = 1


class ClientFunctionConfigList(pydantic.BaseModel):
    functions: List[ClientFunctionConfig]
    hyperparams: Optional[Hyperparams]
    
class AggregationFunctionConfig(pydantic.BaseModel): 
    function: FunctionInvocationConfig
    hyperparams: Optional[AggregationHyperParams]   


class CognitoConfig(pydantic.BaseModel):
    user_pool_id: str
    region_name: str
    auth_endpoint: str
    invoker_client_id: str
    invoker_client_secret: str
    required_scopes: List[str] = ["client-functions/invoke"]
class ServerFunctions(pydantic.BaseModel):
    provider: Optional[str]
    invoker: Optional[FunctionDeploymentConfig]
    evaluator: FunctionInvocationConfig
    aggregator: AggregationFunctionConfig
    
class ExperimentConfig(pydantic.BaseModel):
    cognito: Optional[CognitoConfig] = None
    database: MongodbConnectionConfig
    # cluster: OpenwhiskClusterConfig
    server: ServerFunctions
    clients: ClientFunctionConfigList
