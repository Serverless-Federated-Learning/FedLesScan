import fedless
from fedless.models import TestMetrics


def handle(event, context):
    return {
        "statusCode": 200,
        "body": f"Hello from OpenFaaS! {TestMetrics(cardinality=2, metrics={'acc': 2.0}).json()}",
    }
