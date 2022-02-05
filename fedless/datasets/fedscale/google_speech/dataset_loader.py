from pathlib import Path
from typing import Iterable, Optional, Dict, List, Iterator


from fedless.datasets.dataset_loaders import (
    DatasetFormatError,
    DatasetLoader,
    DatasetNotLoadedError,
    merge_datasets,
)

from typing import Union, Dict, Iterator, List, Optional, Tuple

import re
import tensorflow as tf
from pydantic import BaseModel, validate_arguments, AnyHttpUrl

from fedless.cache import cache

from enum import Enum
from pydantic import Field

from fedless.datasets.fedscale.google_speech.data_processing import preprocess_dataset


class FedScaleDataset(str, Enum):
    """
    Officially supported datasets
    """
    SPEECH = "speech"
    
class FedScaleConfig(BaseModel):
    """Configuration parameters for LEAF dataset loader"""

    type: str = Field("speech", const=True)
    dataset: FedScaleDataset
    location: Union[AnyHttpUrl, Path]
    http_params: Dict = None
    user_indices: Optional[List[int]] = None
    
class FedScale(DatasetLoader):
    
    @validate_arguments
    def __init__(
        self,
        dataset: FedScaleDataset,
        location: Union[AnyHttpUrl, Path],
        http_params: Dict = None,
        user_indices: Optional[List[int]] = None,
    ):
        self.dataset = dataset
        self.source = location
        self.http_params = http_params
        self.user_indices = user_indices
           
    
    @cache
    def load(self) -> tf.data.Dataset:
        """
        Load dataset
        :raise DatasetNotLoadedError when an error occurred in the process
        """
        tx_file_path = tf.keras.utils.get_file(cache_subdir='data',origin = self.source,extract=True )
        st_client_idx = re.search(r'client_(\S+).zip', self.source).group(0)
        tx = tf.io.gfile.glob(tx_file_path[:-1*len(st_client_idx)]+"*.wav")
        tx = tf.random.shuffle(tx)
        tx_ds = preprocess_dataset(tx)
        return tx_ds
