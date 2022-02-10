import abc
from typing import Iterator, Optional, List, Tuple

import pymongo_inmemory

from fedless.models import (
    Parameters,
    ClientResult,
    TestMetrics,
)


class ParameterAggregator(abc.ABC):

    """
    return the dict containing client sepcs and clientresults containing the files to agregate
    """

    @abc.abstractmethod
    def select_aggregation_candidates(self, **kwargs) -> Iterator:
        pass

    @abc.abstractmethod
    def aggregate(
        self, client_results: Iterator[ClientResult], client_feats: List[dict]
    ) -> Tuple[Parameters, Optional[List[TestMetrics]]]:
        pass
