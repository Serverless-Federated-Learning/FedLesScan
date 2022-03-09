import yaml
from fedless.models import (
    ApiGatewayLambdaFunctionConfig,
    FunctionInvocationConfig,
    GCloudFunctionConfig,
    OpenwhiskWebActionConfig,
    AzureFunctionHTTPConfig,
    OpenFaasFunctionConfig,
)

if __name__ == "__main__":
    functions = []

    # # LRZ Openwhisk
    # for i in range(5):
    #     config = FunctionInvocationConfig(
    #         type="openwhisk-web",
    #         params=OpenwhiskWebActionConfig(
    #             endpoint=f"https://138.246.233.207:31001/api/v1/web/guest/default/client-indep-secure-{i+1}.json",
    #             token="XCtAsCYFHFxk8xef18nRFVU5cc2Tj+Cmc5PoSOCD",
    #         ),
    #     )
    #     functions.append({"function": config.dict()})

    file_name = "my_funcs.yaml"
    # Gcloud
    for i in range(100):
        config = FunctionInvocationConfig(
            type="gcloud",
            params=GCloudFunctionConfig(
                url=f"https://http-indep-{i}-n2nf4txpja-ez.a.run.app/"
            ),
        )
        functions.append({"function": config.dict()})

    # for i in range(50):
    #     config = FunctionInvocationConfig(
    #         type="openfaas",
    #         params=OpenFaasFunctionConfig(
    #             url=f"http://127.0.0.1:31112/function/client-indep-{i}"
    #         ),
    #     )
    #     functions.append({"function": config.dict()})

    with open(file_name, "w") as file1:
        file1.write(yaml.dump(functions))
