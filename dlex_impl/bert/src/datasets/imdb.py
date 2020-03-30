import os
import pickle
import re
from typing import List

import tensorflow as tf
from dlex.utils import logger, dump_pkl, load_pkl
from tensorflow import keras
import tensorflow_hub as hub
import pandas as pd
import bert
from bert import run_classifier, optimization, tokenization

from dlex import Params
from dlex.datasets import DatasetBuilder
from dlex.datasets.tf import Dataset

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


def load_directory_data(directory):
    logger.info("Loading data from %s...", directory)
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    logger.info("Creating tokenizer from hub module...")
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.compat.v1.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


class IMDB(DatasetBuilder):
    def __init__(self, params: Params):
        super().__init__(params, [
            'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        ])

    def maybe_preprocess(self, force=False) -> bool:
        if not super().maybe_preprocess(force):
            return True
        for mode in ["train"]:
            pass

    def get_tensorflow_wrapper(self, mode: str):
        return TensorflowIMDB(self, mode)

    def evaluate(self, pred, ref, metric: str, output_path: str):
        if metric == "acc":
            return tf.metrics.accuracy(ref, pred)
        elif metric == "f1":
            return tf.contrib.metrics.f1_score(ref, pred)
        elif metric == "auc":
            return tf.metrics.auc(ref, pred)
        elif metric == "recall":
            return tf.metrics.recall(ref, pred)
        elif metric == "precision":
            return tf.metrics.precision(ref, pred)
        else:
            return super().evaluate(pred, ref, metric, output_path)


class TensorflowIMDB(Dataset):
    def __init__(self, builder, mode):
        super().__init__(builder, mode)

        self._input_fn = bert.run_classifier.input_fn_builder(
            features=self.data,
            seq_length=self.configs.max_len,
            is_training=mode == "train",
            drop_remainder=False)
        logger.info("Finished loading data")

    @property
    def data(self):
        if not self._data:
            pkl_filepath = os.path.join(self.processed_data_dir, self.mode + ".pkl")
            if os.path.exists(pkl_filepath):
                self._data = load_pkl(pkl_filepath)
            else:
                label_list = [0, 1]

                pos_df = load_directory_data(os.path.join(self.raw_data_dir, "aclImdb", self.mode, "pos"))
                neg_df = load_directory_data(os.path.join(self.raw_data_dir, "aclImdb", self.mode, "neg"))
                pos_df["polarity"] = 1
                neg_df["polarity"] = 0

                data = pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)
                # data = data.sample(1000)
                data = data.apply(lambda x: bert.run_classifier.InputExample(
                    guid=None,  # Globally unique ID for bookkeeping, unused in this example
                    text_a=x['sentence'],
                    text_b=None,
                    label=x['polarity']), axis=1)
                self._data = bert.run_classifier.convert_examples_to_features(
                    data, label_list, self.configs.max_len, self.tokenizer)
                logger.info("Saving data to %s...", pkl_filepath)
                dump_pkl(self._data, pkl_filepath)
            logger.info("Dataset %s: %d samples", self.mode, len(self._data))
        return self._data

    @property
    def tokenizer(self):
        if not hasattr(self.builder, "tokenizer"):
            self.builder.tokenizer = create_tokenizer_from_hub_module()
        return self.builder.tokenizer

    @property
    def num_labels(self):
        return 2

    def input_fn(self, params):
        return self._input_fn(params)

    @property
    def num_train_steps(self):
        train_cfg = self.params.train
        return int(len(self) / train_cfg.batch_size * train_cfg.num_epochs)