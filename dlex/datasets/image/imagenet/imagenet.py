import os

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.keras import KerasDataset
from dlex.utils.logging import logger


_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
_RESIZE_MIN = 256


def _central_crop(image, crop_size):
    """Performs central crops of the given image list.
    Args:
      image: a 3-D image tensor
      crop_height: the height of the image following the crop.
      crop_width: the width of the image following the crop.
    Returns:
      3-D tensor with cropped image.
    """
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_size[0])
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_size[1])
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(
        image, [crop_top, crop_left, 0], [crop_size[0], crop_size[1], -1])


def _mean_image_subtraction(image, means, num_channels):
    """Subtracts the given means from each image channel.
    For example:
      means = [123.68, 116.779, 103.939]
      image = _mean_image_subtraction(image, means)
    Note that the rank of `image` must be known.
    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.
      num_channels: number of color channels in the image that will be distorted.
    Returns:
      the centered image.
    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    # We have a 1-D tensor of means; convert to 3-D.
    # Note(b/130245863): we explicitly call `broadcast` instead of simply
    # expanding dimensions for better performance.
    means = tf.broadcast_to(means, tf.shape(image))

    return image - means


def _smallest_size_at_least(height, width, resize_min):
    """Computes new shape with the smallest side equal to `smallest_side`.
    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.
    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      resize_min: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.
    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: an int32 scalar tensor indicating the new width.
    """
    resize_min = tf.cast(resize_min, tf.float32)

    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    return new_height, new_width


def _aspect_preserving_resize(image, resize_min):
    """Resize images preserving the original aspect ratio.
    Args:
      image: A 3-D image `Tensor`.
      resize_min: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.
    Returns:
      resized_image: A 3-D tensor containing the resized image.
    """
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    new_height, new_width = _smallest_size_at_least(height, width, resize_min)

    return _resize_image(image, new_height, new_width)


def _resize_image(image, height, width):
    """Simple wrapper around tf.resize_images.
    This is primarily to make sure we use the same `ResizeMethod` and other
    details each time.
    Args:
      image: A 3-D image `Tensor`.
      height: The target height for the resized image.
      width: The target width for the resized image.
    Returns:
      resized_image: A 3-D tensor containing the resized image. The first two
        dimensions have the shape [height, width].
    """
    return tf.compat.v1.image.resize(
        image, [height, width], method=tf.image.ResizeMethod.BILINEAR,
        align_corners=False)


class KerasImageNet(KerasDataset):
    def __init__(self, builder, mode, params):
        super().__init__(builder, mode, params)

        _, self._info = tfds.load("imagenet2012", split=self._mode, with_info=True)
        self.maybe_prepare_tfrecord()
        tfrecord_path = os.path.join(builder.get_processed_data_dir(), 'imagenet_%s.tfrecord' % mode)

        raw_dataset = tf.data.TFRecordDataset([tfrecord_path])

        parsed_dataset = raw_dataset.map(self._string2feature)
        dataset = parsed_dataset.map(
            lambda item: (
                tf.reshape(item['image'] / 128, [params.dataset.size, params.dataset.size, 3]),
                tf.one_hot(item['label'][0], self.num_classes)))

        if mode == "train":
            dataset = dataset.repeat().shuffle(1000).batch(params.train.batch_size)
        else:
            dataset = dataset.take(len(self) // params.train.batch_size * params.train.batch_size).repeat().batch(params.train.batch_size)
        self.dataset = dataset

    def _string2feature(self, example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, {
            'image': tf.io.FixedLenFeature([self.params.dataset.size ** 2 * 3], tf.float32),
            'label': tf.io.FixedLenFeature([1], tf.int64),
        })

    @staticmethod
    def _feature2string(image, label):
        feature = {
            'image': tf.train.Feature(float_list=tf.train.FloatList(value=image.numpy())),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()])),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    @property
    def input_size(self):
        return [self.params.dataset.size, self.params.dataset.size]

    @property
    def num_channels(self):
        return 3

    def _preprocess_input(self, image):
        image = _aspect_preserving_resize(image, _RESIZE_MIN)
        image = _central_crop(image, self.input_size)
        image = _mean_image_subtraction(image, _CHANNEL_MEANS, self.num_channels)
        return image

    def maybe_prepare_tfrecord(self):
        os.makedirs(self.builder.get_processed_data_dir(), exist_ok=True)
        tfrecord_path = os.path.join(self.builder.get_processed_data_dir(), 'imagenet_%s.tfrecord' % self._mode)
        if os.path.exists(tfrecord_path):
            return
        data, info = tfds.load("imagenet2012", split=self._mode, with_info=True)
        data = data.map(lambda item: (
            tf.reshape(self._preprocess_input(item['image']), [-1]),
            item['label']))
        # tf.one_hot(item['label'], self.num_classes)))

        logger.info("Preparing tfrecord files into %s" % self.builder.get_processed_data_dir())

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
        return KerasImageNet(self, mode, self.params)