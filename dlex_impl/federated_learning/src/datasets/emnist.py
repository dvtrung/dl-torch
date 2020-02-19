import collections
from typing import List

import tensorflow as tf
import tensorflow_federated as tff
from dlex import MainConfig
from dlex.datasets import DatasetBuilder
from dlex.datasets.tf import Dataset
from dlex.utils import logger


class EMNIST(DatasetBuilder):
    def __init__(self, params: MainConfig):
        super().__init__(params, tensorflow_cls=TensorflowEMNIST)


class TensorflowEMNIST(Dataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
        self._emnist = emnist_train if mode == "train" else emnist_test
        logger.info("No. client ids: %d", len(self._emnist.client_ids))
        self._client_ids = self._emnist.client_ids[:10]
        self._data = [self.preprocess(self._emnist.create_tf_dataset_for_client(x)) for x in self._client_ids]
        logger.info(len(self._data))
        logger.info(self._data[0])

        dataset = self._emnist.create_tf_dataset_for_client(self._client_ids[0])
        self.dummy_batch = tf.nest.map_structure(lambda x: x.numpy(), next(iter(self.preprocess(dataset))))

    def __len__(self):
        return len(self._emnist)

    def preprocess(self, dataset):
        def element_fn(element):
            return collections.OrderedDict([
                ('x', tf.reshape(element['pixels'], [-1])),
                ('y', tf.reshape(element['label'], [1])),
            ])
        dataset = dataset.repeat(self.params.train.num_epochs)
        dataset = dataset.map(element_fn)
        dataset = dataset.shuffle(self.configs.shuffle_buffer)
        dataset = dataset.batch(self.params.train.batch_size)
        return dataset