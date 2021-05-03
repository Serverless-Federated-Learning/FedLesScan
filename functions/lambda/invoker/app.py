from pydantic import ValidationError

from fedless.invocation import (
    InvocationError,
    function_invoker_handler,
)
from fedless.providers import lambda_proxy_handler
from fedless.models import InvokerParams


@lambda_proxy_handler(caught_exceptions=(ValidationError, InvocationError))
def handler(event, context):
    """
    Train client on given data and model and return :class:`fedless.client.ClientResult`.
    Relies on :meth:`fedless.client.lambda_proxy_handler` decorator
    for return object conversion and error handling
    :return Response dictionary compatible with API gateway's lambda-proxy integration
    """
    config = InvokerParams.parse_obj(event["body"])

    return function_invoker_handler(
        session_id=config.session_id,
        round_id=config.round_id,
        client_id=config.client_id,
        database=config.database,
        http_headers=config.http_headers
    )
