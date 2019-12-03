import csv
import json
import os
import subprocess
import tempfile
from typing import Dict, List

from dlex.torch.utils.ops_utils import maybe_cuda, LongTensor
import torch.nn as nn
from tqdm import tqdm

from dlex.datasets.nlp.utils import Vocab
from dlex.torch.datatypes import VariableLengthTensor
from ..base import QADataset, BatchX, QABatch


class PytorchSQuAD_V1(QADataset):
    vocab_word = None
    vocab_char = None
    word_embedding_layer = None

    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)
        data = []
        with open(os.path.join(builder.get_processed_data_dir(), mode + ".csv")) as f:
            reader = csv.reader(f)
            examples = list(reader)

        examples = [[val.split(' ') for val in ex] for ex in examples]
        if mode == 'train':
            examples = list(filter(
                lambda ex: len(ex[1]) <= self.configs.paragraph_max_length and len(ex[2]) <= self.configs.question_max_length,
                examples))

        # We want these variables to be shared among instances
        PytorchSQuAD_V1.word_embedding_layer, itos = self.load_embeddings(
            tokens=self.builder.vocab_word.tolist(),
            specials=['<sos>', '<eos>', '<oov>', '<pad>'])
        PytorchSQuAD_V1.vocab_word = self.builder.vocab_word if itos is None else Vocab(itos)
        PytorchSQuAD_V1.vocab_char = self.builder.vocab_char

        for id, context, question, answer_span in tqdm(examples, desc="Loading data (%s)" % mode):
            data.append(dict(
                id=id[0],
                context=context,
                cw=self.vocab_word.encode_token_list(context),
                qw=self.vocab_word.encode_token_list(question),
                cc=[self.vocab_char.encode_token_list(list(w)) for w in context],
                qc=[self.vocab_char.encode_token_list(list(w)) for w in question],
                answer_span=[int(pos) for pos in answer_span[0].split('-')]
            ))
        self._data = data

    @property
    def word_dim(self):

        return self.configs.embeddings.dim

    @property
    def char_dim(self):
        return self.configs.char_dim

    def collate_fn(self, batch: List[Dict]):
        # batch.sort(key=lambda item: len(item.X), reverse=True)
        w_contexts = [item['cw'] for item in batch]
        w_questions = [item['qw'] for item in batch]

        # char_max_length = max([max(len(c) for c in item['cc']) for item in batch])
        char_max_length = 16
        c_contexts = [LongTensor([
            char_idx[:char_max_length] + max(char_max_length - len(char_idx), 0) * [self.vocab_char.blank_token_idx]
            for char_idx in item['cc']
        ]) for item in batch]

        char_max_length = max([max(len(c) for c in item['qc']) for item in batch])
        c_questions = [LongTensor([
            char_idx + (char_max_length - len(char_idx)) * [self.vocab_char.blank_token_idx]
            for char_idx in item['qc']
        ]) for item in batch]

        answer_spans = LongTensor([item['answer_span'] for item in batch])

        w_contexts = VariableLengthTensor(w_contexts, padding_value=self.vocab_word.blank_token_idx)
        w_questions = VariableLengthTensor(w_questions, padding_value=self.vocab_word.blank_token_idx)
        c_contexts = nn.utils.rnn.pad_sequence(
            c_contexts, batch_first=True, padding_value=self.vocab_char.blank_token_idx)
        c_questions = nn.utils.rnn.pad_sequence(
            c_questions, batch_first=True, padding_value=self.vocab_char.blank_token_idx)

        return QABatch(
            X=BatchX(maybe_cuda(w_contexts), maybe_cuda(c_contexts), maybe_cuda(w_questions), maybe_cuda(c_questions)),
            Y=answer_spans, Y_len=None)

    def evaluate(self, y_pred, y_ref, metric: str):
        if metric in {"em", "f1"}:
            assert len(y_pred) == len(self.data)
            ret = {}
            for pred, data in zip(y_pred, self.data):
                ret[data['id']] = ' '.join(self.vocab_word.decode_idx_list(data['cw'][pred[0]:pred[1] + 1]))
            try:
                f_name = tempfile.mktemp()
                with open(f_name, 'w') as f:
                    json.dump(ret, f)

                process = subprocess.Popen([
                    'python', os.path.join(*os.path.split(os.path.realpath(__file__))[:-1], 'evaluate-v1.1.py'),
                    os.path.join(self.builder.get_working_dir(), "%s-v1.1.json" % self.mode),
                    f.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, _ = process.communicate()
            finally:
                os.remove(f_name)
            out = json.loads(out.decode())
            return out['exact_match'] if metric == 'em' else out['f1']
        else:
            super().evaluate(y_pred, y_ref, metric)