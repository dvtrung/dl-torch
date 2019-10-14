import abc
import os

from dlex.datasets.builder import DatasetBuilder


class NLPDatasetBuilder(DatasetBuilder):
    @abc.abstractmethod
    def evaluate(self, y_pred, y_ref, metric: str) -> (int, int):
        if metric == "bleu":
            import nltk
            score = nltk.translate.bleu_score.corpus_bleu([[ref] for ref in y_ref], y_pred)
            return score
        else:
            return super().evaluate(y_pred, y_ref, metric)

    def format_output(self, y_pred, batch_item) -> (str, str, str):
        return super().format_output(y_pred, batch_item)

    def get_vocab_path(self, tag: str):
        os.makedirs(os.path.join(self.get_processed_data_dir(), "vocab"), exist_ok=True)
        return os.path.join(self.get_processed_data_dir(), "vocab", f"{tag}.txt")