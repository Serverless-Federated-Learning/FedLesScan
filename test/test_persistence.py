import os
from unittest import mock
from unittest.mock import patch

import mongomock
import pymongo
import pytest
from pymongo.errors import ConnectionFailure

from fedless.client import ClientResult
from fedless.persistence import (
    MongodbConnectionConfig,
    MongodbClientResultStorage,
    StorageConnectionError,
    ResultAlreadyExistsException,
    ResultNotLoadedException,
    PersistenceError,
)


@pytest.fixture
def client_result():
    return ClientResult(
        weights="abc123",
        history={"loss": [1.0, 0.5, 0.0]},
        cardinality=12,
    )


@pytest.fixture
def mongodb_config():
    config = MongodbConnectionConfig(
        host="localhost", port=27001, username="admin", password="password123"
    )
    with mongomock.patch(servers=((config.host, config.port),)):
        yield config


def test_mongodb_connection_config_format_url():
    config = MongodbConnectionConfig(
        host="localhost", port=27001, username="admin", password="password123@#"
    )
    assert config.url == "mongodb://admin:password123%40%23@localhost:27001"


# noinspection TimingAttack
def test_mongodb_config_fills_in_environment_variables():
    with mock.patch.dict(
        os.environ,
        {
            "FEDLESS_MONGODB_HOST": "localhost",
            "FEDLESS_MONGODB_PORT": "27001",
            "FEDLESS_MONGODB_USERNAME": "admin",
            "FEDLESS_MONGODB_PASSWORD": "password123",
        },
    ):
        config = MongodbConnectionConfig(password="password12345")
        assert config.host == "localhost"
        assert config.port == 27001
        assert config.username == "admin"
        assert config.password == "password12345"


def test_mongodb_storage_insert_result(client_result, mongodb_config):
    with MongodbClientResultStorage(
        mongodb_config, database="db", collection="results"
    ) as storage:
        storage.save("key-1", client_result)
        returned_result = storage.load("key-1")
        assert returned_result == client_result


def test_mongodb_storage_works_with_url(client_result, mongodb_config):
    with MongodbClientResultStorage(
        mongodb_config.url, database="db", collection="results"
    ) as storage:
        storage.save("key-1", client_result)
        returned_result = storage.load("key-1")
        assert returned_result == client_result


def test_mongodb_storage_throws_error_on_invalid_name():
    config = MongodbConnectionConfig(
        host="localhost", port=27101, username="admin", password="pw"
    )
    with pytest.raises(StorageConnectionError):
        MongodbClientResultStorage(config, database="db.", collection="col")


def test_mongodb_storage_save_throws_error_on_duplicate_key(
    mongodb_config, client_result
):
    storage = MongodbClientResultStorage(
        mongodb_config, database="db", collection="results"
    )

    storage.save("client-0", client_result)

    with pytest.raises(ResultAlreadyExistsException):
        storage.save("client-0", client_result, overwrite=False)


def test_mongodb_storage_save_overwrites_key_by_default(mongodb_config, client_result):
    storage = MongodbClientResultStorage(
        mongodb_config, database="db", collection="results"
    )

    storage.save("client-0", client_result)
    storage.save("client-0", client_result)

    assert storage.load("client-0") == client_result


def test_mongodb_storage_wraps_connection_error(mongodb_config, client_result):
    storage = MongodbClientResultStorage(
        mongodb_config, database="db", collection="results"
    )

    with pytest.raises(StorageConnectionError), patch.object(
        storage._collection, "replace_one"
    ) as mock_replace:
        mock_replace.side_effect = ConnectionFailure
        storage.save("client-0", client_result)

    with pytest.raises(StorageConnectionError), patch.object(
        storage._collection, "find_one"
    ) as mock_find:
        mock_find.side_effect = ConnectionFailure
        storage.load("client-0")


def test_mongodb_throws_error_on_missing_key(mongodb_config):
    with MongodbClientResultStorage(
        mongodb_config, database="db", collection="results"
    ) as storage:
        with pytest.raises(ResultNotLoadedException):
            storage.load("non-existent-key")


def test_mongodb_wraps_other_mongo_errors(mongodb_config):
    storage = MongodbClientResultStorage(
        mongodb_config, database="db", collection="results"
    )
    for error_class in [
        pymongo.errors.ProtocolError,
        pymongo.errors.NetworkTimeout,
        pymongo.errors.DocumentTooLarge,
    ]:
        with pytest.raises(PersistenceError):
            with patch.object(storage._collection, "find_one") as mock_find:
                mock_find.side_effect = error_class
                storage.load("client-0")
            with patch.object(storage._collection, "replace_one") as mock_replace:
                mock_replace.side_effect = error_class
                storage.load("client-0")
