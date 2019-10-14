import random

import numpy as np

from dlex.datasets.nlp.utils import Vocab
from dlex.datasets.voice.builder import VoiceDataset
from dlex.datasets.seq2seq.torch import PytorchSeq2SeqDataset
from dlex.torch import BatchItem

random.seed(1)


class Dummy(VoiceDataset):
    def __init__(self, params):
        super().__init__(params)

    def get_pytorch_wrapper(self, mode: str):
        return PytorchDummy(self, mode)


class PytorchDummy(PytorchSeq2SeqDataset):
    input_size = 50

    def __init__(self, builder, mode):
        self.vocab = Vocab()
        for w in range(1, self.input_size + 1):
            self.vocab.add_token(str(w))
        super().__init__(builder, mode)
        self._output_size = self.input_size + len(self.params.dataset.special_tokens)
        labels = list(range(self.input_size))
        feats = np.eye(self.input_size)
        min_length = 10
        max_length = 20
        inputs = [[random.choice(labels) for _ in range(random.randint(min_length, max_length))] for _ in range(len(self))]
        self._data = [
            BatchItem(X=[feats[label] for label in seq], Y=seq)
            for seq in inputs]

        if self.params.dataset.sort:
            self._data.sort(key=lambda item: len(item.Y))

    def __len__(self):
        return 10000 if self.mode == "train" else 100