import abc
import os

from torchtext.vocab import GloVe

from dlex.configs import ModuleConfigs
from dlex.datasets.builder import DatasetBuilder


class NLPDataset(DatasetBuilder):
    @abc.abstractmethod
    def evaluate(self, pred, ref, metric: str) -> (int, int):
        if metric == "bleu":
            import nltk
            # reference = self._trim_result(reference)
            # hypothesis = self._trim_result(hypothesis)
            score = nltk.translate.bleu_score.sentence_bleu([ref], pred, weights=(0.5, 0.5))
            total = 1
            return score, total
        else:
            return super().evaluate(pred, ref, metric)

    def get_embedding_vectors(self):
        return GloVe(
            name='6B', dim=self.params.dataset.embedding_dim,
            cache=os.path.join(ModuleConfigs.TMP_PATH, "torchtext"))

    def format_output(self, y_pred, batch_item) -> (str, str, str):
        return super().format_output(y_pred, batch_item)