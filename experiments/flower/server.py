from typing import Optional, Tuple, List, Dict

import click
import flwr as fl
from flwr.common import EvaluateRes, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg

from fedless.data import MNIST
from fedless.benchmark.fedkeeper import create_mnist_cnn


# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    test_set = MNIST(split="test").load().batch(32)

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(test_set)
        return loss, accuracy

    return evaluate


class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        print(rnd, results, failures)
        loss_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.loss,
                    evaluate_res.accuracy,
                )
                for _, evaluate_res in results
            ]
        )
        accuracy_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.metrics.get("accuracy", 0.0),
                    evaluate_res.loss,
                )
                for _, evaluate_res in results
            ]
        )
        return loss_aggregated, {"accuracy": accuracy_aggregated}


@click.command()
@click.option("--dataset", type=str)
@click.option("--min-num-clients", type=int, default=5)
@click.option("--rounds", type=int, default=200)
@click.option("--port", type=int, default=31532)
def run(dataset, min_num_clients, rounds, port):
    client_manager = fl.server.SimpleClientManager()
    strategy = AggregateCustomMetricStrategy(
        min_fit_clients=min_num_clients,
        min_eval_clients=min_num_clients,
        min_available_clients=min_num_clients,
        eval_fn=get_eval_fn(create_mnist_cnn()) if dataset.lower() == "mnist" else None,
    )
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)

    # Run server
    fl.server.start_server(
        f"[::]:{port}",
        server,
        config={"num_rounds": rounds},
    )


if __name__ == "__main__":
    run()