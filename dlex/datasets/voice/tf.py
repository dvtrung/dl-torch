from datasets.nlp.nlp import Vocab
from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.tf import TensorflowDataset


class TensorflowSpeechRecognitionDataset(TensorflowDataset):
    def __init__(
            self,
            builder: DatasetBuilder,
            mode: str,
            params,
            vocab_path: str):
        super().__init__(builder, mode, params)
        self.vocab = Vocab(vocab_path)
        self.vocab.add_token('<sos>')
        self.vocab.add_token('<eos>')
        self._output_size = len(self.vocab)

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def sos_token_idx(self) -> int:
        return self.vocab.sos_token_idx

    @property
    def eos_token_idx(self) -> int:
        return self.vocab.eos_token_idx