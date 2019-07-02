from dataclasses import dataclass

import tensorflow as tf


@dataclass
class Batch:
    X: tf.Tensor
    Y: tf.Tensor
    X_len: list = None
    Y_len: list = None
