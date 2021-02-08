from fedless.client import (
    default_handler,
    ClientConfig,
    lambda_proxy_handler,
)


@lambda_proxy_handler
def handler(event, context):
    """
    Train client on given data and model and return :class:`fedless.client.ClientResult`.
    Relies on :meth:`fedless.client.lambda_proxy_handler` decorator
    for return object conversion and error handling
    :return Response dictionary compatible with API gateway's lambda-proxy integration
    """
    config = ClientConfig.parse_obj(event["body"])

    return default_handler(
        data_config=config.data,
        model_config=config.model,
        hyperparams=config.hyperparams,
    )
