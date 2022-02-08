from fedless.strategies.Intelligent_selection import DBScanClientSelection
from fedless.strategies.fedkeeper_strategy import FedkeeperStrategy
from fedless.strategies.fedless_strategy import FedlessStrategy
from fedless.mocks.mock_fedless_strategy import MockFedlessStrategy

from fedless.strategies.Intelligent_selection import (
    DBScanClientSelection,
    RandomClientSelection,
)


def selectStrategy(strategy: str, invocation_attrs: dict):
    # todo fix fedkeeper args
    switcher = {
        # "fedkeeper": FedkeeperStrategy(**invocation_attrs),
        "fedless_enhanced": FedlessStrategy(
            selectionStrategy=DBScanClientSelection(
                invocation_attrs["mongodb_config"], invocation_attrs["session"]
            ),
            **invocation_attrs,
        ),
        "fedless": FedlessStrategy(
            selectionStrategy=RandomClientSelection(), **invocation_attrs
        ),
        "fedless_mock": MockFedlessStrategy(
            selectionStrategy=DBScanClientSelection(
                invocation_attrs["mongodb_config"], invocation_attrs["session"]
            ),
            **invocation_attrs,
        ),
    }

    # default to fedless strategy
    return switcher.get(strategy, switcher["fedless"])
