from fedless.models import AggregationStrategy
from fedless.strategies.Intelligent_selection import DBScanClientSelection
# from fedless.strategies.fedkeeper_strategy import FedkeeperStrategy
from fedless.strategies.fedless_strategy import FedlessStrategy
# from fedless.mocks.mock_fedless_strategy import MockFedlessStrategy

from fedless.strategies.Intelligent_selection import (
    DBScanClientSelection,
    RandomClientSelection,
)


def select_strategy(strategy: str, invocation_attrs: dict):
    switcher = {
        # "fedkeeper": FedkeeperStrategy(**invocation_attrs),
        "fedless_enhanced": FedlessStrategy(
            selection_strategy=DBScanClientSelection(
                invocation_attrs["mongodb_config"], invocation_attrs["session"],invocation_attrs["save_dir"]
            ),
            aggregation_strategy=AggregationStrategy.PER_SESSION,
            **invocation_attrs,
        ),
        "fedless": FedlessStrategy(
            selection_strategy=RandomClientSelection(),
            aggregation_strategy=AggregationStrategy.PER_ROUND,
            **invocation_attrs,
        ),
        # "fedless_mock": MockFedlessStrategy(
        #     selection_strategy=DBScanClientSelection(
        #         invocation_attrs["mongodb_config"], invocation_attrs["session"],invocation_attrs["save_dir"]
        #     ),
        #     aggregation_strategy=AggregationStrategy.PER_SESSION,
        #     **invocation_attrs,
        # ),
    }

    # default to fedless strategy
    return switcher.get(strategy, switcher["fedless_enhanced"])
