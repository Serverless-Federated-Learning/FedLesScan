import logging

import azure.functions
from pydantic import ValidationError

from fedless.providers import azure_handler
from fedless.models import InvokerParams
from fedless.client import fedless_mongodb_handler, ClientError

logging.basicConfig(level=logging.DEBUG)


@azure_handler(caught_exceptions=(ValidationError, ValueError, ClientError))
def main(req: azure.functions.HttpRequest):
    params = InvokerParams.parse_obj(req.get_json())

    return fedless_mongodb_handler(
        session_id=params.session_id,
        round_id=params.round_id,
        client_id=params.client_id,
        database=params.database,
    )
