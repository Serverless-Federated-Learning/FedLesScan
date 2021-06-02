from typing import Optional, Tuple

import click
import flwr as fl

from fedless.data import MNIST
from fedless.benchmark.fedkeeper import create_mnist_cnn


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    test_set = MNIST(split="test").load().batch(128)

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(test_set)
        return loss, accuracy

    return evaluate


@click.command()
@click.option("--dataset", type=str)
@click.option("--min-num-clients", type=int, default=5)
@click.option("--rounds", type=int, default=200)
@click.option("--port", type=int, default=31532)
def run(dataset, min_num_clients, rounds, port):
    client_manager = fl.server.SimpleClientManager()
    strategy = fl.server.strategy.FedAvg(
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
