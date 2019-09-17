import abc
import os

from torchtext.vocab import GloVe, FastText

from dlex.configs import ModuleConfigs
from dlex.datasets.builder import DatasetBuilder


class NLPDataset(DatasetBuilder):
    @abc.abstractmethod
    def evaluate(self, y_pred, y_ref, metric: str) -> (int, int):
        if metric == "bleu":
            import nltk
            score = nltk.translate.bleu_score.corpus_bleu([[ref] for ref in y_ref], y_pred)
            return score
        else:
            return super().evaluate(y_pred, y_ref, metric)

    def get_embedding_vectors(self):
        if self.params.dataset.pretrained_embeddings == 'glove':
            return GloVe(
                name='6B', dim=self.params.dataset.embedding_dim,
                cache=os.path.join(ModuleConfigs.TMP_PATH, "torchtext"))
        elif self.params.dataset.pretrained_embeddings == 'fasttext':
            return FastText()

    def format_output(self, y_pred, batch_item) -> (str, str, str):
        return super().format_output(y_pred, batch_item)

    def get_vocab_path(self, name):
        os.makedirs(os.path.join(self.get_processed_data_dir(), "vocab"), exist_ok=True)
        return os.path.join(self.get_processed_data_dir(), "vocab", f"{name}.txt")