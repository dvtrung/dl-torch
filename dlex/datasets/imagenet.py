import os

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from dlex.datasets import DatasetBuilder, KerasDataset
from dlex.utils.logging import logger


class KerasImageNet(KerasDataset):
    def __init__(self, dataset, mode, params):
        super().__init__(dataset, mode, params)

        _, self._info = tfds.load("imagenet2012", split=self._mode, with_info=True)
        tfrecord_path = os.path.join(dataset.get_processed_data_dir(), 'imagenet_%s.tfrecord' % mode)

        raw_dataset = tf.data.TFRecordDataset([tfrecord_path])

        parsed_dataset = raw_dataset.map(self._string2feature)
        dataset = parsed_dataset.map(
            lambda item: (
                tf.reshape(item['image'], [params.dataset.size, params.dataset.size, 3]),
                tf.one_hot(item['label'][0], self.num_classes)))

        if mode == "train":
            dataset = dataset.shuffle(1000).repeat().batch(params.batch_size)
        else:
            dataset = dataset.take(len(self) * params.batch_size).repeat().batch(params.batch_size)
        self._dataset = dataset

    def _string2feature(self, example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, {
            'image': tf.io.FixedLenFeature([self._params.dataset.size ** 2 * 3], tf.int64),
            'label': tf.io.FixedLenFeature([1], tf.int64),
        })

    @staticmethod
    def _feature2string(image, label):
        feature = {
            'image': tf.train.Feature(int64_list=tf.train.Int64List(value=image.numpy())),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()])),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    def maybe_prepare_tfrecord(self):
        tfrecord_path = os.path.join(self._dataset.get_processed_data_dir(), 'imagenet_%s.tfrecord' % self._mode)
        if os.path.exists(tfrecord_path):
            return
        data, info = tfds.load("imagenet2012", split=self._mode, with_info=True)
        data = data.map(lambda item: (
            tf.reshape(
                tf.cast(
                    tf.image.resize(item['image'], (self._params.dataset.size, self._params.dataset.size)),
                    tf.int64)
                , [-1]),
            item['label']))
        # tf.one_hot(item['label'], self.num_classes)))

        logger.info("Preparing tfrecord files into %s" % self._dataset.get_processed_data_dir())

        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for image, label in tqdm(iter(data), desc="tfrecord"):
                example = self._feature2string(image, label)
                writer.write(example)

    def __len__(self):
        return self._info.splits[self._mode].num_examples

    @property
    def num_classes(self):
        return 1000


class ImageNet(DatasetBuilder):
    def get_keras_wrapper(self, mode: str) -> KerasDataset:
        return KerasImageNet(self, mode, self._params)