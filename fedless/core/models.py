from typing import Optional, List, Dict

import pydantic

from fedless.models import (
    FunctionDeploymentConfig,
    FunctionInvocationConfig,
    Hyperparams,
    MongodbConnectionConfig,
)


class ServerFunctions(pydantic.BaseModel):
    provider: Optional[str]
    invoker: Optional[FunctionDeploymentConfig]
    evaluator: FunctionInvocationConfig
    aggregator: FunctionInvocationConfig


class FedkeeperClientConfig(pydantic.BaseModel):
    function: FunctionInvocationConfig
    hyperparams: Optional[Hyperparams]
    replicas: int = 1


class FedkeeperClientsConfig(pydantic.BaseModel):
    functions: List[FedkeeperClientConfig]
    hyperparams: Optional[Hyperparams]


class CognitoConfig(pydantic.BaseModel):
    user_pool_id: str
    region_name: str
    auth_endpoint: str
    invoker_client_id: str
    invoker_client_secret: str
    required_scopes: List[str] = ["client-functions/invoke"]

# todo remove
# class ClusterConfig(pydantic.BaseModel):
#     database: MongodbConnectionConfig
#     clients: FedkeeperClientsConfig
#     providers: Dict[str, FaaSProviderConfig]
#     function: ServerFunctions
#     cognito: Optional[CognitoConfig]


class ExperimentConfig(pydantic.BaseModel):
    cognito: Optional[CognitoConfig] = None
    database: MongodbConnectionConfig
    # cluster: OpenwhiskClusterConfig
    server: ServerFunctions
    clients: FedkeeperClientsConfig
