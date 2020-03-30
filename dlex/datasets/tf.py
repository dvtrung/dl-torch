import random
from typing import NamedTuple, Any, Dict, List

import tensorflow.compat.v1 as tf
from dlex.datasets import DatasetBuilder


class Dataset:
    def __init__(self, builder: DatasetBuilder, mode: str):
        self.params = builder.params
        self.mode = mode
        self._builder = builder
        self._data = None
        self._sampler = None

    @property
    def builder(self):
        return self._builder

    @property
    def data(self):
        return self._data

    @property
    def processed_data_dir(self) -> str:
        return self.builder.get_processed_data_dir()

    @property
    def raw_data_dir(self) -> str:
        return self.builder.get_raw_data_dir()

    @property
    def configs(self):
        return self.params.dataset

    def __len__(self):
        return len(self.data)

    @property
    def is_training(self):
        return self.mode == "train"

    def populate_feed_dict(
            self,
            feed_dict: Dict[tf.placeholder, Any],
            placeholders: NamedTuple,
            data: List[Any]):
        raise NotImplemented

    def evaluate(self, pred, ref, metric, output_path):
        return self.builder.evaluate(pred, ref, metric, output_path)

    def format_output(self, y_pred, batch_item):
        return self.builder.format_output(y_pred, batch_item)

    def shuffle(self):
        random.shuffle(self._data)

    def write_results_to_file(
            self,
            all_predictions: List[Any],
            output_path: str,
            output_tag: str,
            format: str = None) -> str:
        raise NotImplemented

    def get_sliced_batch(self, placeholders: NamedTuple, start, end) -> NamedTuple:
        return placeholders.__class__(**{
            key: val[start:end] if len(val.get_shape()) > 1 else val
            for key, val in placeholders._asdict().items()
        })