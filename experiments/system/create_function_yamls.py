import yaml
from fedless.models import (
    ApiGatewayLambdaFunctionConfig,
    FunctionInvocationConfig,
    GCloudFunctionConfig,
    OpenwhiskWebActionConfig,
)

if __name__ == "__main__":
    functions = []
    for i in range(20):
        config = FunctionInvocationConfig(
            type="lambda",
            params=ApiGatewayLambdaFunctionConfig(
                apigateway=f"https://sfeenj9g31.execute-api.eu-central-1.amazonaws.com/dev/federated/client-{i + 1}",
                api_key="dlTu9R1Lc41IrLhJpj9Mv79dFwpShckP40dieHHe",
            ),
        )
        functions.append({"function": config.dict()})

    for i in range(70):
        config = FunctionInvocationConfig(
            type="gcloud",
            params=GCloudFunctionConfig(
                url=f"https://europe-west3-thesis-303614.cloudfunctions.net/http-{i}"
            ),
        )
        functions.append({"function": config.dict()})

    for i in range(10):
        config = FunctionInvocationConfig(
            type="openwhisk-web",
            params=OpenwhiskWebActionConfig(
                endpoint=f"https://138.246.233.207:31001/api/v1/web/guest/default/client-{i}.json",
                token="rHMzaoFGivt7GszbnnO9YRcsxU61lEUd9XAVpF8U",
            ),
        )
        functions.append({"function": config.dict()})

    print(yaml.dump(functions))
