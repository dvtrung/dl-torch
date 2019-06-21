import abc

from dlex.datasets.builder import DatasetBuilder


class NLPDatasetBuilder(DatasetBuilder):
    @abc.abstractmethod
    def evaluate(self, hypothesis, reference, metric: str) -> (int, int):
        if metric == "bleu":
            import nltk
            reference = self._trim_result(reference)
            hypothesis = self._trim_result(hypothesis)
            score = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0.5, 0.5))
            total = 1
            return score, total
        else:
            raise Exception("Unsupported metric.")