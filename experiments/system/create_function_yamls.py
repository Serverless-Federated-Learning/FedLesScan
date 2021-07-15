import yaml
from fedless.models import (
    ApiGatewayLambdaFunctionConfig,
    FunctionInvocationConfig,
    GCloudFunctionConfig,
    OpenwhiskWebActionConfig,
    AzureFunctionHTTPConfig,
)

if __name__ == "__main__":
    functions = []

    for i in range(10):
        config = FunctionInvocationConfig(
            type="gcloud",
            params=GCloudFunctionConfig(
                url=f"https://europe-west3-federatedlearning-319719.cloudfunctions.net/http-indep-secure-{i + 1}"
            ),
        )
        functions.append({"function": config.dict()})

    # Azure
    for i in range(5):
        config = FunctionInvocationConfig(
            type="azure",
            params=AzureFunctionHTTPConfig(
                trigger_url=f"https://fedless-client-{i + 1}.azurewebsites.net/api/client-indep-secure"
            ),
        )
        functions.append({"function": config.dict()})

    # IBM Openwhisk
    for i in range(10):
        config = FunctionInvocationConfig(
            type="openwhisk-web",
            params=OpenwhiskWebActionConfig(
                endpoint=f"https://eu-de.functions.cloud.ibm.com/api/v1/web/09b13c7b-7fcf-455e-b974-3b2348dc0152/"
                f"default/client-indep-secure-{i + 1}.json",
                token="vxA/+lrxBi2CUmdpmrdPKgz8VT49vqY8aSnK33fW",
                self_signed_cert=False,
            ),
        )
        functions.append({"function": config.dict()})

    for i in range(10):
        config = FunctionInvocationConfig(
            type="lambda",
            params=ApiGatewayLambdaFunctionConfig(
                apigateway=f"https://6ptskoqcbf.execute-api.eu-central-1.amazonaws.com/client-indep-secure-{i + 1}",
                # api_key="HPKpJnrlGQ7fnsh1h5BJBaW3TvcUM9Nf2WKOmUkD",
            ),
        )
        functions.append({"function": config.dict()})

    # LRZ Openwhisk
    # for i in range(10):
    #    config = FunctionInvocationConfig(
    #        type="openwhisk-web",
    #        params=OpenwhiskWebActionConfig(
    #            endpoint=f"https://138.246.233.207:31001/api/v1/web/guest/default/client-{i + 1}.json",
    #            token="PEETox0e/24aVK6+Xy9KERZMnnnwlHJa620wbk28",
    #        ),
    #    )
    #    functions.append({"function": config.dict()})
    #

    print(yaml.dump(functions))
