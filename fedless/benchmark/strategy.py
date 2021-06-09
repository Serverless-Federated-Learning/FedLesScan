import asyncio
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import urllib3
from pydantic import ValidationError
from requests import Session

from fedless.benchmark.common import run_in_executor, fetch_cognito_auth_token
from fedless.benchmark.models import CognitoConfig
from fedless.invocation import (
    invoke_sync,
    retry_session,
    InvocationError,
)
from fedless.models import (
    TestMetrics,
    FunctionInvocationConfig,
    FunctionDeploymentConfig,
    ClientConfig,
    AggregatorFunctionResult,
    AggregatorFunctionParams,
    MongodbConnectionConfig,
    DatasetLoaderConfig,
    EvaluatorParams,
    EvaluatorResult,
    InvokerParams,
    InvocationResult,
)
from fedless.providers import FaaSProvider

logger = logging.getLogger(__name__)


class FLStrategy(ABC):
    def __init__(self, clients):
        self.clients = clients

    def aggregate_metrics(
        self, metrics: List[TestMetrics], metric_names: Optional[List[str]] = None
    ) -> Dict:
        if metric_names is None:
            metric_names = ["loss"]

        cardinalities, metrics = zip(
            *((metric.cardinality, metric.metrics) for metric in metrics)
        )
        result_dict = {}
        for metric_name in metric_names:
            values = [metric[metric_name] for metric in metrics]
            mean = np.average(values, weights=cardinalities)
            result_dict.update(
                {
                    f"mean_{metric_name}": mean,
                    f"all_{metric_name}": values,
                    f"median_{metric_name}": np.median(values),
                }
            )
        return result_dict

    @abstractmethod
    async def fit_round(self, round: int, clients: List) -> Tuple[float, float, Dict]:
        """
        :return: (loss, accuracy, metrics) tuple
        """

    def sample_clients(self, clients: int, pool: List) -> List:
        return random.sample(pool, clients)

    async def fit(
        self,
        n_clients_in_round: int,
        max_rounds: int,
        max_accuracy: Optional[float] = None,
    ):
        for round in range(max_rounds):
            clients = self.sample_clients(n_clients_in_round, self.clients)
            logger.info(f"Sampled {len(clients)} for round {round}")
            loss, accuracy, metrics = await self.fit_round(round, clients)
            logger.info(
                f"Round {round} finished. Global loss={loss}, accuracy={accuracy}"
            )

            if max_accuracy and accuracy >= max_accuracy:
                logger.info(
                    f"Reached accuracy {accuracy} after {round + 1} rounds. Aborting..."
                )
                break


class ServerlessFlStrategy(FLStrategy, ABC):
    def __init__(
        self,
        clients: List,
        provider: FaaSProvider,
        mongodb_config: MongodbConnectionConfig,
        evaluator_config: FunctionDeploymentConfig,
        aggregator_config: FunctionDeploymentConfig,
        client_timeout: float = 300,
        allowed_stragglers: int = 0,
        global_test_data: Optional[DatasetLoaderConfig] = None,
        aggregator_params: Optional[Dict] = None,
        session: Optional[str] = None,
        save_dir: Optional[Path] = None,
        proxies: Dict = None,
    ):
        super().__init__(clients=clients)
        urllib3.disable_warnings()
        self.session: str = session or str(uuid.uuid4())
        self.provider = provider
        self.log_metrics = []
        self.client_timings = []
        self.allowed_stragglers = allowed_stragglers

        self.mongodb_config = mongodb_config
        self.aggregator_params = aggregator_params or {}
        self.global_test_data = global_test_data

        self._aggregator: Optional[FunctionInvocationConfig] = None
        self._evaluator: Optional[FunctionInvocationConfig] = None

        self.evaluator_config: FunctionDeploymentConfig = evaluator_config
        self.aggregator_config: FunctionDeploymentConfig = aggregator_config
        self.client_timeout: float = client_timeout
        self.clients: List[ClientConfig] = clients
        self.save_dir = save_dir
        self.proxies = proxies or {}

    @abstractmethod
    async def deploy_all_functions(self, *args, **kwargs):
        pass

    def save_round_results(
        self, session: str, round: int, dir: Optional[Path] = None, **kwargs
    ) -> None:
        self.log_metrics.append({"session_id": session, "round_id": round, **kwargs})

        if not dir:
            dir = Path.cwd()
        pd.DataFrame.from_records(self.log_metrics).to_csv(
            dir / f"timing_{session}.csv"
        )
        pd.DataFrame.from_records(self.client_timings).to_csv(
            dir / f"clients_{session}.csv"
        )

    @run_in_executor
    def _async_call_request(
        self,
        function: FunctionInvocationConfig,
        data: Dict,
        session: Optional[Session] = None,
        timeout: float = 300,
    ) -> Dict:
        session = retry_session(backoff_factor=0.5, retries=5, session=session)
        return invoke_sync(
            function_config=function, data=data, session=session, timeout=timeout
        )

    @property
    def aggregator(self) -> FunctionInvocationConfig:
        if not self._aggregator:
            raise ValueError()
        return self._aggregator

    @property
    def evaluator(self) -> FunctionInvocationConfig:
        if not self._evaluator:
            raise ValueError()
        return self._evaluator

    def call_aggregator(self, round: int) -> AggregatorFunctionResult:
        params = AggregatorFunctionParams(
            session_id=self.session,
            round_id=round,
            database=self.mongodb_config,
            **self.aggregator_params,
        )
        session = Session()
        # session.proxies.update(self.proxies)
        result = invoke_sync(
            self.aggregator,
            data=params.dict(),
            session=retry_session(backoff_factor=1.0, retries=5, session=session),
        )
        try:
            return AggregatorFunctionResult.parse_obj(result)
        except ValidationError as e:
            raise ValueError(f"Aggregator returned invalid result.") from e

    def call_evaluator(self, round: int) -> EvaluatorResult:
        params = EvaluatorParams(
            session_id=self.session,
            round_id=round + 1,
            database=self.mongodb_config,
            test_data=self.global_test_data,
        )
        session = Session()
        # session.proxies.update(self.proxies)
        result = invoke_sync(
            self.evaluator,
            data=params.dict(),
            session=retry_session(backoff_factor=1.0, retries=5, session=session),
        )
        try:
            return EvaluatorResult.parse_obj(result)
        except ValidationError as e:
            raise ValueError(f"Evaluator returned invalid result.") from e

    async def invoke_async(
        self,
        function: FunctionInvocationConfig,
        data: Dict,
        session: Optional[Session] = None,
        timeout: float = 300,
    ) -> Dict:
        return await self._async_call_request(function, data, session, timeout=timeout)

    async def fit_round(
        self, round: int, clients: List[ClientConfig]
    ) -> Tuple[float, float, Dict]:
        round_start_time = time.time()
        metrics_misc = {}
        loss, acc = None, None

        # Invoke clients
        t_clients_start = time.time()
        succs, errors = await self.call_clients(round, clients)

        if len(succs) < (len(clients) - self.allowed_stragglers):
            logger.error(errors)
            raise Exception(
                f"Only {len(succs)}/{len(clients)} clients finished this round, "
                f"required are {len(clients) - self.allowed_stragglers}."
            )
        logger.info(
            f"Received results from {len(succs)}/{len(clients)} client functions"
        )

        t_clients_end = time.time()

        logger.info(f"Invoking Aggregator")
        t_agg_start = time.time()
        agg_res: AggregatorFunctionResult = self.call_aggregator(round)
        t_agg_end = time.time()
        logger.info(f"Aggregator combined result of {agg_res.num_clients} clients.")
        metrics_misc["aggregator_seconds"] = t_agg_end - t_agg_start

        if self.global_test_data:
            logger.info(f"Running global evaluator function")
            t_eval_start = time.time()
            eval_res = self.call_evaluator(round)
            t_eval_end = time.time()
            metrics_misc["evaluator_seconds"] = t_eval_end - t_eval_start
            loss = eval_res.metrics.metrics.get("loss")
            acc = eval_res.metrics.metrics.get("accuracy")
        else:
            logger.info(f"Computing test statistics from clients")
            if not agg_res.test_results:
                raise ValueError(
                    f"Clients or aggregator did not return local test results..."
                )
            metrics = self.aggregate_metrics(
                metrics=agg_res.test_results, metric_names=["loss", "accuracy"]
            )
            loss = metrics.get("mean_loss")
            acc = metrics.get("mean_accuracy")

        metrics_misc.update(
            {
                "round_seconds": time.time() - round_start_time,
                "clients_finished_seconds": t_clients_end - t_clients_start,
                "num_clients_round": len(clients),
                "global_test_accuracy": acc,
                "global_test_loss": loss,
            }
        )

        logger.info(f"Round {round}: loss={loss}, acc={acc}")
        self.save_round_results(
            session=self.session,
            round=round,
            dir=self.save_dir,
            **metrics_misc,
        )
        return loss, acc, metrics_misc

    @abstractmethod
    async def call_clients(
        self, round: int, clients: List[ClientConfig]
    ) -> Tuple[List[InvocationResult], List[str]]:
        pass


class FedkeeperStrategy(ServerlessFlStrategy):
    def __init__(
        self,
        provider: FaaSProvider,
        clients: List[ClientConfig],
        invoker_config: FunctionDeploymentConfig,
        mongodb_config: MongodbConnectionConfig,
        evaluator_config: FunctionDeploymentConfig,
        aggregator_config: FunctionDeploymentConfig,
        client_timeout: float = 300,
        global_test_data: Optional[DatasetLoaderConfig] = None,
        aggregator_params: Optional[Dict] = None,
        allowed_stragglers: int = 0,
        use_separate_invokers: bool = True,
        session: Optional[str] = None,
        save_dir: Optional[Path] = None,
        proxies: Dict = None,
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
        )
        self.use_separate_invokers = use_separate_invokers
        self.invoker_config: FunctionDeploymentConfig = invoker_config

        # Will be set during deployment
        self._invoker: Optional[FunctionInvocationConfig] = None
        self._client_to_invoker: Optional[Dict[str, FunctionInvocationConfig]] = None

    @property
    def client_to_invoker(self) -> Dict[str, FunctionInvocationConfig]:
        if self._client_to_invoker is None:
            raise ValueError()
        return self._client_to_invoker

    async def deploy_all_functions(self, *args, **kwargs):
        logger.info(f"Deploying fedkeeper functions...")
        logger.info(f"Deploying aggregator and evaluator")
        self._aggregator = await self.provider.deploy(self.aggregator_config.params)
        self._evaluator = await self.provider.deploy(self.evaluator_config.params)

        if self.use_separate_invokers:
            client_ids = [client.client_id for client in self.clients]
            logger.info(
                f"Found {len(client_ids)} client functions. Deploying one invoker each"
            )
            client_invoker_mappings = []
            for ix, client in enumerate(client_ids):
                invoker_config = self.invoker_config.copy(deep=True)

                invoker_config.params.name = f"{invoker_config.params.name}-{ix}"
                invoker_invocation_config = await self.provider.deploy(
                    invoker_config.params
                )
                client_invoker_mappings.append((client, invoker_invocation_config))
                logger.debug(f"Deployed invoker {invoker_config.params.name}")
            self._client_to_invoker = {c: inv for (c, inv) in client_invoker_mappings}
        else:
            invoker = await self.provider.deploy(self.invoker_config.params)
            self._client_to_invoker = defaultdict(lambda: invoker)
            logger.debug(f"Deployed invoker {invoker.params.name}")

    async def call_clients(
        self, round: int, clients: List[ClientConfig]
    ) -> Tuple[List[InvocationResult], List[str]]:
        urllib3.disable_warnings()
        tasks = []

        for client in clients:
            session = Session()
            # session.proxies.update(self.proxies)
            session = retry_session(session=session)
            params = InvokerParams(
                session_id=self.session,
                round_id=round,
                client_id=client.client_id,
                database=self.mongodb_config,
                http_proxies=self.proxies,
            )
            invoker = self.client_to_invoker[client.client_id]

            # function with closure for easier logging
            async def _inv(function, data, session):
                try:
                    t_start = time.time()
                    res = await self.invoke_async(
                        function, data, session=session, timeout=self.client_timeout
                    )
                    dt_call = time.time() - t_start
                    self.client_timings.append(
                        {
                            "client_id": client.client_id,
                            "session_id": self.session,
                            "seconds": dt_call,
                        }
                    )
                    return res
                except InvocationError as e:
                    return str(e)

            tasks.append(
                asyncio.create_task(
                    _inv(function=invoker, data=params.dict(), session=session)
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


class FedlessStrategy(ServerlessFlStrategy):
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
        )
        self.cognito = cognito

    async def deploy_all_functions(self, *args, **kwargs):
        logger.info(f"Deploying fedless functions...")
        logger.info(f"Deploying aggregator and evaluator")
        self._aggregator = await self.provider.deploy(self.aggregator_config.params)
        self._evaluator = await self.provider.deploy(self.evaluator_config.params)

    async def call_clients(
        self, round: int, clients: List[ClientConfig]
    ) -> Tuple[List[InvocationResult], List[str]]:
        urllib3.disable_warnings()
        tasks = []

        http_headers = {}
        if self.cognito:
            token = fetch_cognito_auth_token(
                user_pool_id=self.cognito.user_pool_id,
                region_name=self.cognito.region_name,
                auth_endpoint=self.cognito.auth_endpoint,
                invoker_client_id=self.cognito.invoker_client_id,
                invoker_client_secret=self.cognito.invoker_client_secret,
                required_scopes=self.cognito.required_scopes,
            )
            http_headers = {"Authorization": f"Bearer {token}"}

        for client in clients:
            session = Session()
            session.headers.update(http_headers)
            session.proxies.update(self.proxies)
            session = retry_session(backoff_factor=1.0, session=session)
            params = InvokerParams(
                session_id=self.session,
                round_id=round,
                client_id=client.client_id,
                database=self.mongodb_config,
                http_proxies=self.proxies,
            )

            # function with closure for easier logging
            async def _inv(function, data, session):
                try:
                    t_start = time.time()
                    res = await self.invoke_async(
                        function, data, session=session, timeout=self.client_timeout
                    )
                    dt_call = time.time() - t_start
                    self.client_timings.append(
                        {
                            "client_id": client.client_id,
                            "session_id": self.session,
                            "seconds": dt_call,
                        }
                    )
                    return res
                except InvocationError as e:
                    return str(e)

            tasks.append(
                asyncio.create_task(
                    _inv(function=client.function, data=params.dict(), session=session)
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
