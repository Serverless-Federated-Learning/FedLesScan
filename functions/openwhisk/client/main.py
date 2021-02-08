from fedless.client import (
    default_handler,
    ClientConfig,
    openwhisk_action_handler,
)


@openwhisk_action_handler
def main(request):
    """
    Train client on given data and model and return :class:`fedless.client.ClientResult`.
    Relies on :meth:`fedless.client.openwhisk_action_handler` decorator
    for return object conversion and error handling
    :return Response dictionary containing http response
    """
    config = ClientConfig.parse_obj(request["body"])

    return default_handler(
        data_config=config.data,
        model_config=config.model,
        hyperparams=config.hyperparams,
    )
