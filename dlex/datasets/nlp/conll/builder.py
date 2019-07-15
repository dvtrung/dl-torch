import os

import nltk
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

from dlex.configs import AttrDict
from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.nlp.utils import write_vocab, Vocab
from dlex.datasets.torch import PytorchSeq2SeqDataset
from dlex.torch import BatchItem
from dlex.utils.logging import logger

DOWNLOAD_URLS = [
    "https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz",
    "https://www.comp.nus.edu.sg/~nlp/sw/m2scorer.tar.gz",
]


class CoNLL2014(DatasetBuilder):
    def __init__(self, params: AttrDict):
        super().__init__(params)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        for url in DOWNLOAD_URLS:
            self.download_and_extract(url, self.get_raw_data_dir())

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)
        if os.path.exists(self.get_processed_data_dir()):
            return
        sentences = {'train': [], 'test': []}
        for mode in sentences.keys():
            with open(os.path.join(self.get_processed_data_dir(), "%s.m2" % mode)) as f:
                sentences[mode] = f.read().split("\n\n")
                sentences[mode] = [s.split('\n', 1) for s in sentences[mode]]
        write_vocab(
            self.get_processed_data_dir(), [s[0][2:] for s in sentences['train']], normalize_fn=lambda s: s.lower())
        self.write_dataset(sentences, self.get_vocab_path())

    def write_dataset(self, sentences, vocab_path):
        vocab = Vocab(vocab_path)
        for mode in sentences.keys():
            outputs = []
            output_file_name = os.path.join(self.get_processed_data_dir(), "%s" % mode + '.csv')
            for sentence in tqdm(sentences[mode]):
                if len(sentence) == 1:
                    sentence = sentence[0][2:]
                    corrections = []
                else:
                    corrections = [s[2:] for s in sentence[1].split('\n')]
                    sentence = sentence[0][2:]
                tokens = sentence.lower().split(' ')
                corrected = tokens
                for corr in reversed(corrections):
                    corr = corr.split('|||')
                    start, end = [int(n) for n in corr[0].split(' ')]
                    error_type = corr[1]
                    corrected_str = corr[2].lower()
                    corrected = corrected[:start] + corrected_str.split(' ') + corrected[end:]
                outputs.append(dict(
                    tokens=' '.join([str(vocab[tkn]) for tkn in tokens]),
                    sentence=sentence,
                    corrected=' '.join([str(vocab[tkn]) for tkn in corrected]),
                    corrected_sentence=' '.join(corrected)
                ))

            logger.info("Output to %s" % output_file_name)
            with open(output_file_name, 'w', encoding='utf-8') as f:
                f.write('\t'.join(['tokens', 'sentence', 'corrected', 'corrected_sentence']) + '\n')
                for o in outputs:
                    f.write('\t'.join([
                        o['tokens'],
                        o['sentence'],
                        o['corrected'],
                        o['corrected_sentence']
                    ]) + '\n')

    def get_vocab_path(self):
        return os.path.join(self.get_processed_data_dir(), "vocab", "word.txt")

    def evaluate(self, pred, ref, metric: str):
        if metric == "wer":
            return nltk.edit_distance(pred, ref), len(ref)
        if metric == "acc":
            return accuracy_score(pred, ref) * len(pred), len(pred)
        else:
            return super().evaluate(pred, ref, metric)

    def get_pytorch_wrapper(self, mode: str):
        return PytorchCoNLL2014(self, mode, self.params)


class PytorchCoNLL2014(PytorchSeq2SeqDataset):
    def __init__(self, builder, mode, params):
        super().__init__(
            builder, mode, params,
            vocab_path=os.path.join(builder.get_processed_data_dir(), "vocab", "word.txt"))
        cfg = params.dataset

        is_debug = mode == "debug"
        if mode == "debug":
            mode = "train"

        data = self.load_data(os.path.join(builder.get_processed_data_dir(), mode + '.csv'))
        if is_debug:
            data = data[:cfg.debug_size]
        if self.params.dataset.sort:
            data.sort(key=lambda it: len(it.Y))
        self._data = data

    def load_data(self, csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[1:]
        lines = [l.split('\t') for l in lines if l != ""]
        data = []
        for l in lines:
            tokens = [int(w) for w in l[0].split(' ')]
            corrected = [int(w) for w in l[2].split(' ')]
            data.append(BatchItem(
                X=tokens,
                Y=corrected
            ))
        return data

    @property
    def vocab_size(self):
        return len(self.vocab)

    def format_output(self, y_pred, batch_item: BatchItem) -> (str, str, str):
        if self.configs.output_format == "text":
            pr = np.array(y_pred)
            gt = batch_item.Y[1:-1] if 'sos' in self.params.dataset.special_tokens else batch_item.Y
            return ' '.join([self.vocab.get_token(word_id) for word_id in batch_item.X]), \
                   ' '.join([self.vocab.get_token(word_id) for word_id in gt]), \
                   ' '.join([self.vocab.get_token(word_id) for word_id in pr])
        else:
            return super().format_output(y_pred, batch_item)

    @property
    def input_size(self):
        return len(self.vocab)