import flwr as fl
import tensorflow as tf

from fedless.benchmark.leaf import create_shakespeare_lstm
from fedless.data import LEAFConfig, DatasetLoaderConfig
from fedless.data import DatasetLoaderBuilder


class FedlessClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
    ):
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.model = model

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.dataset, epochs=1, batch_size=32, steps_per_epoch=3)
        return self.model.get_weights(), self.dataset.cardinality(), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_dataset)
        return loss, self.test_dataset.cardinality(), {"accuracy": accuracy}


if __name__ == "__main__":
    model = create_shakespeare_lstm()

    dataset_config = DatasetLoaderConfig(
        type="leaf",
        params=LEAFConfig(
            dataset="shakespeare",
            location="http://138.246.235.163:31715/data/leaf/data/shakespeare/data/test/user_0_all_data_niid_05_keep_64_test_9.json",
        ),
    )

    train_set = DatasetLoaderBuilder.from_config(dataset_config).load()
    test_set = train_set
    client = FedlessClient(model=model, dataset=train_set, test_dataset=test_set)

    fl.client.start_client("192.0.0.1", client)
