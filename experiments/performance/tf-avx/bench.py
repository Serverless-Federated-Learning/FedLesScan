import time
from collections import defaultdict

import pandas as pd
from requests import Session

from fedless.auth import CognitoClient
from fedless.invocation import invoke_sync, InvocationError
from fedless.models import (
    FunctionInvocationConfig,
    GCloudFunctionConfig,
    InvokerParams,
    MongodbConnectionConfig,
)

if __name__ == "__main__":

    cognito = CognitoClient(user_pool_id="eu-west-1_Ay4jDCguX", region_name="eu-west-1")
    token = cognito.fetch_token_for_client(
        auth_endpoint=f"https://fedless-pool-2.auth.eu-west-1.amazoncognito.com/oauth2/token",
        client_id="16q863gk2kggfen78uf5aa2o23",
        client_secret="fg7bq2dr4p1idovtcspn8ablf76bd3g9nu0d2o11ldl2bnbhqtl",
        required_scopes=["client-functions/invoke"],
    )
    data = InvokerParams(
        session_id="27d9e62f-d611-404d-9dcb-0e5e2d187e82",
        round_id=0,
        client_id="d4820d0e-093d-451b-a814-dd06eecce6d4",
        database=MongodbConnectionConfig(host="138.246.233.217"),
    ).dict()
    session = Session()
    session.headers = {"Authorization": f"Bearer {token}"}
    # session = retry_session(session=session, retries=1)

    timings = defaultdict(list)
    n_rounds = 10
    for url in [
        "https://europe-west3-thesis-303614.cloudfunctions.net/http-indep-secure-tf4-avx-1",
        "https://europe-west3-thesis-303614.cloudfunctions.net/http-indep-secure-tf4-vanilla-1",
        "https://europe-west3-thesis-303614.cloudfunctions.net/http-indep-secure-tf5-axx-enabled-1",
        "https://europe-west3-thesis-303614.cloudfunctions.net/http-indep-secure-tf5-axx-disabled-1",
        "https://europe-west3-thesis-303614.cloudfunctions.net/http-indep-secure-tf4-avx512-1",
    ]:
        for i in range(n_rounds):
            start_time = time.time()
            try:
                result = invoke_sync(
                    FunctionInvocationConfig(
                        type="gcloud",
                        params=GCloudFunctionConfig(url=url),
                    ),
                    data=data,
                    session=session,
                )
                print(result)
                duration = time.time() - start_time
                print(
                    f"Execution of function {url} took {duration} seconds ({i + 1}/{n_rounds})"
                )
                timings[url].append(duration)
                df = pd.DataFrame.from_records(
                    list(
                        {"fun": url, "time": t} for url in timings for t in timings[url]
                    )
                )
                df.to_csv("timings.csv")
                print(f'{df.groupby("fun").mean()}')
            except InvocationError as e:
                print(e)
    print(timings)
    for url in timings:
        print(url, sum(timings[url]) / len(timings[url]))
