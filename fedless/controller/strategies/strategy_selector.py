from fedless.common.models import AggregationStrategy
from fedless.controller.strategies.Intelligent_selection import DBScanClientSelection

from fedless.controller.strategies.fedless_strategy import FedlessStrategy

from fedless.controller.strategies.Intelligent_selection import (
    DBScanClientSelection,
    RandomClientSelection,
)


def select_strategy(strategy: str, invocation_attrs: dict):
    switcher = {
        "fedlesscan": FedlessStrategy(
            selection_strategy=DBScanClientSelection(
                invocation_attrs["mongodb_config"],
                invocation_attrs["session"],
                invocation_attrs["save_dir"],
            ),
            aggregation_strategy=AggregationStrategy.PER_SESSION,
            **invocation_attrs,
        ),
        "fedavg": FedlessStrategy(
            selection_strategy=RandomClientSelection(),
            aggregation_strategy=AggregationStrategy.PER_ROUND,
            **invocation_attrs,
        ),
        "fedprox": FedlessStrategy(
            selection_strategy=RandomClientSelection(),
            aggregation_strategy=AggregationStrategy.PER_ROUND,
            **invocation_attrs,
        ),
    }

    # default to fedless strategy
    return switcher.get(strategy, switcher["fedlesscan"])
