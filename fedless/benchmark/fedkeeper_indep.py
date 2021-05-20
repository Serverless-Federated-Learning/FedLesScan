import os
from typing import Optional, Dict

import asyncio
import pydantic
from requests import Session

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disable tensorflow logs


from fedless.models import (
    FunctionDeploymentConfig,
    MongodbConnectionConfig,
    FaaSProviderConfig,
    InvokerParams,
)
from fedless.benchmark.fedkeeper import (
    FedkeeperStrategy,
    CognitoConfig,
    FedkeeperClientsConfig,
    get_deployment_manager,
)


class FedlessClusterFunctions(pydantic.BaseModel):
    provider: str
    evaluator: FunctionDeploymentConfig
    aggregator: FunctionDeploymentConfig


class ClusterConfig(pydantic.BaseModel):
    database: MongodbConnectionConfig
    clients: FedkeeperClientsConfig
    providers: Dict[str, FaaSProviderConfig]
    fedkeeper: FedlessClusterFunctions
    cognito: Optional[CognitoConfig]


class FedlessStrategy(FedkeeperStrategy):
    async def deploy(self):
        providers = self.config.providers
        evaluator = self.config.fedkeeper.evaluator
        aggregator = self.config.fedkeeper.aggregator

        # Check if referenced provider actually exist
        if self.config.fedkeeper.provider not in providers.keys():
            raise KeyError(f"Provider {self.config.fedkeeper.provider} not specified")

        # Get deployment-manager
        cluster_provider: FaaSProviderConfig = providers[self.config.fedkeeper.provider]
        deployment_manager = await get_deployment_manager(cluster_provider)

        # Deploy or update evaluator and aggregator
        await asyncio.gather(
            deployment_manager.deploy(evaluator.params),
            deployment_manager.deploy(aggregator.params),
        )

        print("Successfully deployed evaluator and aggregator")

        cluster_provider: FaaSProviderConfig = self.config.providers[
            self.config.fedkeeper.provider
        ]
        deployment_manager = await get_deployment_manager(cluster_provider)

        self.evaluator_function = await deployment_manager.to_invocation_config(
            self.config.fedkeeper.evaluator.params
        )
        self.aggregator_function = await deployment_manager.to_invocation_config(
            self.config.fedkeeper.aggregator.params
        )

    async def _invoke_clients(self, clients_in_round, round_id, session_id):
        print(f"Running round {round_id} with {len(clients_in_round)} clients")
        client_tasks = []
        session = Session()
        session.headers = (
            {"Authorization": f"Bearer {self.fetch_cognito_auth_token()}"}
            if self.config.cognito
            else {}
        )
        for client in clients_in_round:
            invoker_params = InvokerParams(
                session_id=session_id,
                round_id=round_id,
                client_id=client.client_id,
                database=self.config.database,
            )

            async def g(params, invoker):
                return await self._call_invoker(params, invoker, session=session)

            task = asyncio.create_task(g(invoker_params, client.function))

            client_tasks.append(task)
        await asyncio.wait(client_tasks)
        return client_tasks
