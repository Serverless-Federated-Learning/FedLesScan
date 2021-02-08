from fedless.client import (
    ClientConfig,
    default_handler,
    gcloud_http_error_handler,
)


@gcloud_http_error_handler
def http(request):
    """Example function running training on client and returning result (weights, history, ...)

    :param request: flask.Request object. Body has to be a serialized ClientConfig
    :returns :class:`fedless.client.ClientResult` that gets wrapped by the decorator as a tuple with HTTP response
        infos. In case of an exception this tuple comprises an error code and information about the exception.
    """
    body: bytes = request.get_data()
    config = ClientConfig.parse_raw(body)
    return default_handler(
        data_config=config.data,
        model_config=config.model,
        hyperparams=config.hyperparams,
    )
