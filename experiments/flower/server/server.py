import flower as fl


def run(sample_fraction, min_sample_size, min_num_clients, rounds):
    client_manager = fl.server.SimpleClientManager()
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=sample_fraction,
        min_fit_clients=min_sample_size,
        min_available_clients=min_num_clients,
        # eval_fn=get_eval_fn(testset),
        # on_fit_config_fn=fit_config,
    )
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)

    # Run server
    fl.server.start_server(
        "127.0.0.1:8005",
        server,
        config={"num_rounds": rounds},
    )


if __name__ == "__main__":
    run()
