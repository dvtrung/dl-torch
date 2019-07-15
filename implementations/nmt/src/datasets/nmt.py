"""Datasets for neural machine translation"""
import abc
from typing import Dict, List, Any

import nltk
import torch

from dlex.configs import AttrDict
from dlex.torch.datasets import NLPDataset, Vocab
from dlex.utils.logging import logger
from dlex.torch.utils.ops_utils import LongTensor


class NMTBaseDataset(NLPDataset):
    """
    :type lang_src: str
    :type lang_tgt: str
    :type lang: (str, str)
    :type vocab: dict[str, Vocab]
    :type input_size: int
    :type output_size: int
    :type data: list
    """
    def __init__(self, mode: str, params: AttrDict, vocab_paths: str = None):
        super().__init__(mode, params)
        cfg = params.dataset

        self.lang_src = cfg.source
        self.lang_tgt = cfg.target
        self.lang = (self.lang_src, self.lang_tgt)

        if vocab_paths is not None:
            # Load vocab
            self.vocab = {}
            for lang in vocab_paths:
                self.vocab[lang] = Vocab(vocab_paths[lang])
                logger.info("%s vocab size: %d", lang, len(self.vocab[lang]))
            self.input_size = len(self.vocab[self.lang_src])
            self.output_size = len(self.vocab[self.lang_tgt])

        is_debug = mode == 'debug'
        if mode == 'debug':
            self.mode = 'test'

        self.data = self._load_data()

        if is_debug:
            self.data = self.data[:cfg.debug_size]

    @abc.abstractmethod
    def _load_data(self):
        """
        :rtype dict
        """
        pass

    def collate_fn(self, batch: List[Dict[str, Any]]):
        batch.sort(key=lambda item: len(item['X']), reverse=True)
        inp = [LongTensor(item['X']).view(-1) for item in batch]
        tgt = [LongTensor(item['Y']).view(-1) for item in batch]
        inp = torch.nn.utils.rnn.pad_sequence(
            inp, batch_first=True,
            padding_value=self.eos_token_idx)
        tgt = torch.nn.utils.rnn.pad_sequence(
            tgt, batch_first=True,
            padding_value=self.eos_token_idx)

        return dict(
            X=inp, X_len=LongTensor([len(item['X']) for item in batch]),
            Y=tgt, Y_len=LongTensor([len(item['Y']) for item in batch]))

    def _trim_result(self, ls):
        start = 0 if len(ls) > 0 and ls[0] != self.sos_token_idx else 1
        end = 0
        while end < len(ls) and ls[end] != self.eos_token_idx:
            end += 1
        return ls[start:end]

    def evaluate(self, y_pred, batch, metric):
        if metric == "bleu":
            target_variables = batch.Y
            score, total = 0, 0
            for k, _y_pred in enumerate(y_pred):
                target = self._trim_result(target_variables[k].cpu().detach().numpy().tolist())
                predicted = self._trim_result(_y_pred)
                score += nltk.translate.bleu_score.sentence_bleu([target], predicted, weights=(0.5, 0.5))
                total += 1
            return score, total

    def format_output(self, y_pred, batch_item):
        src = self._trim_result(batch_item['X'].cpu().numpy())
        tgt = self._trim_result(batch_item['Y'].cpu().numpy())
        y_pred = self._trim_result(y_pred)
        if self.configs.output_format == "text":
            return ' '.join([self.vocab[self.lang_src].get_token(word_id) for word_id in src]), \
                ' '.join([self.vocab[self.lang_tgt].get_token(word_id) for word_id in tgt]), \
                ' '.join([self.vocab[self.lang_tgt].get_token(word_id) for word_id in y_pred])
        else:
            return super().format_output(y_pred, batch_item)

    @property
    def sos_token_idx(self):
        assert self.vocab[self.lang_src].sos_token_idx == self.vocab[self.lang_tgt].sos_token_idx
        return self.vocab[self.lang_src].sos_token_idx

    @property
    def eos_token_idx(self):
        assert self.vocab[self.lang_src].eos_token_idx == self.vocab[self.lang_tgt].eos_token_idx
        return self.vocab[self.lang_src].eos_token_idx