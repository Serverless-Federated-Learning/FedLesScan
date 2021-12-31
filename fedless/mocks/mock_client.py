from typing import Any
from fedless.client import fedless_mongodb_handler
from fedless.models import InvokerParams


class MockClient:
    def __init__(self, config: InvokerParams, delay: int = 0):
        self.config = config
        self.delay = delay

    async def runClient(self):

        return fedless_mongodb_handler(
            session_id=self.config.session_id,
            round_id=self.config.round_id,
            client_id=self.config.client_id,
            database=self.config.database,
            evaluate_only=self.config.evaluate_only,
        )
