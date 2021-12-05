from fedless.strategies.fedkeeper import FedkeeperStrategy
from fedless.strategies.fedless_strategy import FedlessStrategy
from fedless.mocks.mock_fedless_strategy import MockFedlessStrategy



def selectStrategy(strategy:str, invocation_attrs:dict):
    # todo fix fedkeeper args
    switcher = {
        # "fedkeeper": FedkeeperStrategy(**invocation_attrs),
        "fedless": FedlessStrategy(**invocation_attrs),
        "fedless_mock":MockFedlessStrategy(**invocation_attrs)
    }

    # default to fedless strategy
    return switcher.get(strategy,switcher["fedless"])