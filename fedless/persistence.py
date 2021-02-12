import abc
from contextlib import AbstractContextManager
from typing import Any, Union, Callable
from urllib import parse

import pydantic
import pymongo
from pydantic import Field
from pymongo.collection import Collection
from pymongo.errors import InvalidName, PyMongoError, ConnectionFailure, BSONError

from fedless.client import ClientResult


class PersistenceError(Exception):
    """Base exception for persistence errors"""


class StorageConnectionError(PersistenceError):
    """Connection to storage resource (e.g. database) could not be established"""


class ResultNotStoredException(PersistenceError):
    """Client result could not be stored"""


class ResultAlreadyExistsException(PersistenceError):
    """A result for this client already exists"""


class ResultNotLoadedException(PersistenceError):
    """Client result could not be loaded"""


class ClientResultStorageObject(pydantic.BaseModel):
    """Client Result persisted in database with corresponding client identifier"""

    key: str
    result: ClientResult


def wrap_pymongo_errors(func: Callable) -> Callable:
    """Decorator to wrap all unhandled pymongo exceptions as persistence errors"""

    # noinspection PyMissingOrEmptyDocstring
    def wrapped_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (PyMongoError, BSONError) as e:
            raise PersistenceError(e) from e

    return wrapped_function


class MongodbConnectionConfig(pydantic.BaseSettings):
    """
    Data class holding info to connection to a MongoDB server.
    Automatically tries to fill in missing values from environment variables
    """

    host: str = Field(...)
    port: int = Field(...)
    username: str = Field(...)
    password: str = Field(...)

    @property
    def url(self) -> str:
        """Return url representation"""
        return f"mongodb://{parse.quote(self.username)}:{parse.quote(self.password)}@{self.host}:{self.port}"

    class Config:
        env_prefix = "fedless_mongodb_"


# noinspection PyMissingOrEmptyDocstring
class ClientResultStorage(abc.ABC):
    """Abstract base class to persist :class:`fedless.client.ClientResult`"""

    @abc.abstractmethod
    def save(self, key: Any, result: ClientResult) -> Any:
        pass

    @abc.abstractmethod
    def load(self, key: Any) -> ClientResult:
        pass


class MongodbClientResultStorage(ClientResultStorage, AbstractContextManager):
    """Store client results in a mongodb database"""

    def __init__(
        self, db: Union[str, MongodbConnectionConfig], database: str, collection: str
    ):
        """
        Connect to mongodb database
        :param db: mongodb url or config object of type :class:`MongodbConnectionConfig`
        :param database: database name
        :param collection: collection name
        """
        self.db = db
        self.database = database
        self.collection = collection

        if isinstance(db, str):
            self._client = pymongo.MongoClient(db)
        else:
            self._client = pymongo.MongoClient(
                host=db.host,
                port=db.port,
                username=db.username,
                password=db.password,
            )
        try:
            self._collection: Collection = self._client[database][collection]
        except InvalidName as e:
            raise StorageConnectionError(e) from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @wrap_pymongo_errors
    def save(self, key: str, result: ClientResult, overwrite: bool = True) -> Any:
        obj = ClientResultStorageObject(key=key, result=result)
        if not overwrite and self._collection.find_one({"key": key}) is not None:
            raise ResultAlreadyExistsException(
                f"Client result with key {key} already exists. Force overwrite with overwrite=True"
            )
        try:
            self._collection.replace_one({"key": key}, obj.dict(), upsert=True)
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e

    def close(self):
        self._client.close()

    @wrap_pymongo_errors
    def load(self, key: str) -> ClientResult:
        try:
            obj_dict = self._collection.find_one(filter={"key": key})
        except ConnectionFailure as e:
            raise StorageConnectionError(e) from e
        if obj_dict is not None and "result" in obj_dict:
            return ClientResult.parse_obj(obj_dict["result"])
        else:
            raise ResultNotLoadedException(f"ClientResult with key {key} not found")
