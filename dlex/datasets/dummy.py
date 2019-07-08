import random

import numpy as np

from dlex.datasets.voice.builder import VoiceDatasetBuilder
from dlex.datasets.voice.torch import PytorchSeq2SeqDataset
from dlex.torch import BatchItem

random.seed(1)


class Dummy(VoiceDatasetBuilder):
    def __init__(self, params):
        super().__init__(params)

    def get_pytorch_wrapper(self, mode: str):
        return PytorchDummy(self, mode, self.params)


class PytorchDummy(PytorchSeq2SeqDataset):
    input_size = 50

    def __init__(self, builder, mode, params):
        super().__init__(builder, mode, params)
        label_start_from = 1 if 'blank' in params.dataset.special_tokens else 2
        self._output_size = self.input_size + label_start_from
        for w in range(1, self.input_size + 1):
            self._vocab.add_token(str(w))

        labels = list(range(label_start_from, self.output_size))
        feats = np.eye(self.input_size)
        max_length = 20
        inputs = [[random.choice(labels) for _ in range(random.randint(1, max_length))] for _ in range(len(self))]
        self._data = [
            BatchItem(X=[feats[label - label_start_from] for label in seq], Y=seq)
            for seq in inputs]

        if params.dataset.sort:
            self._data.sort(key=lambda item: len(item.Y))

    @property
    def sos_token_idx(self):
        return 0

    @property
    def eos_token_idx(self):
        return 1

    @property
    def blank_token_idx(self):
        return 0

    def __len__(self):
        return 10000 if self._mode == "train" else 100