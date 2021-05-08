import yaml
from fedless.models import ApiGatewayLambdaFunctionConfig, FunctionInvocationConfig, GCloudFunctionConfig

if __name__ == "__main__":
    num = 50
    base_url = "https://sfeenj9g31.execute-api.eu-central-1.amazonaws.com/dev/federated/client-"
    functions = []
    for i in range(num):
        url = f"{base_url}{i + 1}"
        config = FunctionInvocationConfig(type="lambda", params=ApiGatewayLambdaFunctionConfig(
            apigateway=url,
            api_key="dlTu9R1Lc41IrLhJpj9Mv79dFwpShckP40dieHHe"
        ))
        functions.append({"function": config.dict()})

    for i in range(45):
        url = f"{base_url}{i + 1}"
        config = FunctionInvocationConfig(type="gcloud", params=GCloudFunctionConfig(
            url=f"https://us-central1-thesis-303614.cloudfunctions.net/http-{i}"
        ))
        functions.append({"function": config.dict()})



    print(yaml.dump(functions))
