import io
import os

import tensorflow as tf
from sklearn.model_selection import train_test_split

from dlex.configs import Params
from dlex.datasets.base import NLPDataset, TensorflowDataset, Dataset
from datasets.nlp.nlp import normalize_string
from dlex.tf.models.base import Batch


class Anki(NLPDataset):
    def maybe_preprocess(self, force=False):
        pass

    def maybe_download_and_extract(self, force=False):
        self._download_and_extract(
            'http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
            self.get_raw_data_dir()
        )

    def get_tensorflow_wrapper(self, mode: str) -> TensorflowDataset:
        return AnkiTensorflowDataset(self, mode)

    def get_pytorch_wrapper(self, mode: str) -> Dataset:
        return AnkiDataset(mode)


class AnkiTensorflowDataset(TensorflowDataset):
    def __init__(self, dataset: Anki, mode: str):
        super().__init__(mode, params)
        lines = io.open(
            os.path.join(dataset.get_raw_data_dir(), "spa-eng", "spa.txt"),
            encoding='UTF-8').read().strip().split('\n')

        target_texts, input_texts = zip(*[["<sos> %s <eos>" % normalize_string(w) for w in l.split('\t')] for l in lines[:3000]])
        input_tensor, self._input_lang_tokenizer = self._tokenize(input_texts)
        target_tensor, self._target_lang_tokenizer = self._tokenize(target_texts)

        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
            input_tensor,
            target_tensor,
            test_size=0.2)

        dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))\
            .shuffle(params.shuffle_buffer_size or 10)
        dataset = dataset.batch(params.train.batch_size, drop_remainder=True)
        self.dataset = dataset
        self.steps_per_epoch = len(input_tensor_train) // params.train.batch_size

    def _tokenize(self, texts):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(texts)
        tensor = lang_tokenizer.texts_to_sequences(texts)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
        return tensor, lang_tokenizer

    @property
    def input_size(self):
        return len(self._input_lang_tokenizer.word_index)

    @property
    def output_size(self):
        return len(self._target_lang_tokenizer.word_index)

    @property
    def sos_token_idx(self):
        return self._target_lang_tokenizer.word_index['<sos>']

    @property
    def eos_token_idx(self):
        return self._target_lang_tokenizer.word_index['<eos>']

    def all(self):
        return [Batch(X, Y) for X, Y in self.dataset.take(self.steps_per_epoch)]


class AnkiDataset(Dataset):
    def __init__(self, mode, params):
        super().__init__(mode, params)

