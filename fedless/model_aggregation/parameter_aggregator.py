import abc
from typing import Iterator, Optional, List, Tuple

from fedless.models import (
    Parameters,
    ClientResult,
    TestMetrics,
)


class ParameterAggregator(abc.ABC):
    @abc.abstractmethod
    def aggregate(
        self, client_results: Iterator[ClientResult]
    ) -> Tuple[Parameters, Optional[List[TestMetrics]]]:
        pass
