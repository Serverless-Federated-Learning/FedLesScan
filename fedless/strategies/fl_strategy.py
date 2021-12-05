import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd


from fedless.benchmark.common import run_in_executor
from fedless.invocation import retry_session, invoke_sync
from fedless.models import (
    TestMetrics,
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
            #TODO straggler identification scheme here
            clients = self.sample_clients(n_clients_in_round, self.clients)
            logger.info(f"Sampled {len(clients)} for round {round}")
            #TODO straggler mitigation scheme
            loss, accuracy, metrics = await self.fit_round(round, clients)
            logger.info(
                f"Round {round} finished. Global loss={loss}, accuracy={accuracy}"
            )

            if max_accuracy and accuracy >= max_accuracy:
                logger.info(
                    f"Reached accuracy {accuracy} after {round + 1} rounds. Aborting..."
                )
                break
