import asyncio
import logging
import random
import time
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import urllib3
from pydantic import ValidationError
from requests import Session

from fedless.benchmark.common import fetch_cognito_auth_token
from fedless.benchmark.models import CognitoConfig
from fedless.strategies.serverless_strategy import ServerlessFlStrategy
from fedless.invocation import retry_session, InvocationError
from fedless.mocks.mock_client import MockClient
from fedless.models import (
    ClientConfig,
    MongodbConnectionConfig,
    FunctionDeploymentConfig,
    DatasetLoaderConfig,
    InvocationResult,
    InvokerParams,
)
from fedless.providers import FaaSProvider

logger = logging.getLogger(__name__)


class MockFedlessStrategy(ServerlessFlStrategy):
    def __init__(
        self,
        provider: FaaSProvider,
        clients: List[ClientConfig],
        mongodb_config: MongodbConnectionConfig,
        evaluator_config: FunctionDeploymentConfig,
        aggregator_config: FunctionDeploymentConfig,
        client_timeout: float = 300,
        cognito: Optional[CognitoConfig] = None,
        global_test_data: Optional[DatasetLoaderConfig] = None,
        aggregator_params: Optional[Dict] = None,
        allowed_stragglers: int = 0,
        session: Optional[str] = None,
        save_dir: Optional[Path] = None,
        proxies: Dict = None,
        invocation_delay: float = None,
        evaluation_timeout: float = 30.0,
    ):
        super().__init__(
            provider=provider,
            session=session,
            clients=clients,
            mongodb_config=mongodb_config,
            global_test_data=global_test_data,
            aggregator_params=aggregator_params,
            evaluator_config=evaluator_config,
            aggregator_config=aggregator_config,
            client_timeout=client_timeout,
            allowed_stragglers=allowed_stragglers,
            save_dir=save_dir,
            proxies=proxies,
            invocation_delay=invocation_delay,
        )
        self.cognito = cognito
        self.evaluation_timeout = evaluation_timeout

    async def deploy_all_functions(self, *args, **kwargs):
        logger.info(f"Deploying fedless functions...")
        logger.info(f"Deploying aggregator and evaluator")
        self._aggregator = await self.provider.deploy(self.aggregator_config.params)
        self._evaluator = await self.provider.deploy(self.evaluator_config.params)

    # function with closure for easier logging
    async def _inv_mock(self, data:InvokerParams, session:Session, round, client_id):
        try:
            if self.invocation_delay:
                await asyncio.sleep(random.uniform(0.0, self.invocation_delay))
            t_start = time.time()
            logger.info(f"***->>> invoking client ${client_id} with time out ${self.client_timeout}")
            cl = MockClient(data)
            res =  await cl.runClient()
            # res = await self.invoke_async(
            #     function,
            #     data,
            #     session=session,
            #     timeout=self.client_timeout
            #     if not data.evaluate_only
            #     else self.evaluation_timeout,
            # )
            dt_call = time.time() - t_start
            self.client_timings.append(
                {
                    "client_id": client_id,
                    "session_id": self.session,
                    "invocation_time": t_start,
                    "function": {"function":"mock"},
                    "seconds": dt_call,
                    "eval": data.evaluate_only,
                    "round": round,
                }
            )
            return res
        except InvocationError as e:
            return str(e)

    async def call_clients(
        self, round: int, clients: List[ClientConfig], evaluate_only: bool = False
    ) -> Tuple[List[InvocationResult], List[str]]:
        urllib3.disable_warnings()
        tasks = []

        for client in clients:
            session = Session()
            session.proxies.update(self.proxies)
            # session = retry_session(backoff_factor=1.0, session=session)
            params = InvokerParams(
                session_id=self.session,
                round_id=round,
                client_id=client.client_id,
                database=self.mongodb_config,
                http_proxies=self.proxies,
                evaluate_only=evaluate_only,
            )


            tasks.append(
                asyncio.create_task(
                    self._inv_mock(
                        data=params,
                        session=session,
                        round=round,
                        client_id=client.client_id,
                    )
                )
            )

        done, pending = await asyncio.wait(tasks)
        results = list(map(lambda f: f.result(), done))
        suc, errs = [], []
        for res in results:
            try:
                suc.append(InvocationResult.parse_obj(res))
            except ValidationError:
                errs.append(res)
        return suc, errs

    async def evaluate_clients(self, round: int, clients: List[ClientConfig]) -> Dict:
        succ, fails = await self.call_clients(round, clients, evaluate_only=True)
        logger.info(
            f"{len(succ)} client evaluations returned, {len(fails)} failures... {fails}"
        )
        client_metrics = [res.test_metrics for res in succ]
        return self.aggregate_metrics(
            metrics=client_metrics, metric_names=["loss", "accuracy"]
        )
