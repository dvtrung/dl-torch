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
    input_size = 54

    def __init__(self, builder, mode):
        self.vocab = Vocab()
        num_tokens = self.input_size - len(builder.params.dataset.special_tokens)
        for w in builder.params.dataset.special_tokens:
            self.vocab.add_token(f"<{w}>")
        for w in range(1, num_tokens + 1):
            self.vocab.add_token(str(w))
        super().__init__(builder, mode)
        self._output_size = self.input_size + len(self.params.dataset.special_tokens)
        labels = [str(i) for i in range(1, num_tokens + 1)]
        feats = np.eye(num_tokens)
        min_length = 10
        max_length = 20
        random.seed = 42
        inputs = [[random.choice(labels) for _ in range(random.randint(min_length, max_length))] for _ in range(len(self))]
        # inputs = [[random.choice(labels)] * random.randint(min_length, max_length) for _ in range(len(self))]
        self._data = [
            BatchItem(
                X=[feats[int(label) - 1] for label in seq],
                Y=[self.vocab.get_token_id(t) for t in seq])
            for seq in inputs]

        if self.params.dataset.sort:
            self._data.sort(key=lambda item: len(item.Y))

    def __len__(self):
        return 10000 if self.mode == "train" else 100