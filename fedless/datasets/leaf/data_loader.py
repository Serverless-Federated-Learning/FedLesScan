from typing import Iterable, Optional, Dict, List, Iterator

from tensorflow import keras

from fedless.datasets.data import LEAF
from fedless.models import LeafDataset, LEAFConfig




def split_source_by_users(config: LEAFConfig) -> Iterable[LEAFConfig]:
    loader = LEAF(
        dataset=config.dataset,
        location=config.location,
        http_params=config.http_params,
        user_indices=config.user_indices,
    )
    loader.load()

    for i, _ in enumerate(loader.users):
        if not config.user_indices or i in config.user_indices:
            yield LEAFConfig(
                dataset=config.dataset,
                location=config.location,
                http_params=config.http_params,
                user_indices=[i],
            )


def split_sources_by_users(source_urls: List[LEAFConfig]) -> Iterator[LEAFConfig]:
    for source in source_urls:
        for config in split_source_by_users(source):
            yield config
