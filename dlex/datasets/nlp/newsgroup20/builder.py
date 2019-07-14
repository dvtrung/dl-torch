import os
from typing import List

from sklearn.metrics import accuracy_score
import torch.nn as nn
from tqdm import tqdm

from dlex.configs import AttrDict
from dlex.datasets.builder import DatasetBuilder
from dlex.datasets.nlp.utils import write_vocab, normalize_string, space_tokenize, Vocab
from dlex.datasets.torch import PytorchDataset
from dlex.torch import Batch
from dlex.torch import BatchItem
from dlex.torch.utils.ops_utils import LongTensor
from dlex.utils.logging import logger

DOWNLOAD_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"


class WikiText2(DatasetBuilder):
    def __init__(self, params: AttrDict):
        super().__init__(params)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        print(self.get_raw_data_dir())
        self.download_and_extract(DOWNLOAD_URL, self.get_raw_data_dir())

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)
        if os.path.exists(self.get_processed_data_dir()):
            return
        transcripts = {'train': [], 'test': [], 'valid': []}
        for mode in transcripts.keys():
            with open(os.path.join(self.get_raw_data_dir(), "wikitext-2", "wiki.%s.tokens" % mode), encoding="utf-8") as f:
                for s in f.read().split('\n'):
                    if s.strip() == "":
                        continue
                    transcripts[mode].append(s)
        write_vocab(
            self.get_processed_data_dir(), transcripts['train'],
            output_file_name="word.txt",
            normalize_fn=normalize_string,
            tokenize_fn=space_tokenize)

        self.write_dataset(
            "word", transcripts,
            vocab_path=self.get_vocab_path("word"),
            normalize_fn=normalize_string,
            tokenize_fn=space_tokenize)

    def get_vocab_path(self, unit):
        return os.path.join(self.get_processed_data_dir(), "vocab", "%s.txt" % unit)

    def write_dataset(self, output_prefix, transcripts, vocab_path, normalize_fn, tokenize_fn):
        vocab = Vocab(vocab_path)
        for mode in transcripts.keys():
            outputs = []
            output_fn = os.path.join(self.get_processed_data_dir(), "%s_%s" % (output_prefix, mode) + '.csv')
            for transcript in tqdm(transcripts[mode], desc=mode):
                tokens = [str(vocab[tkn]) for tkn in tokenize_fn(normalize_fn(transcript))]
                outputs.append(dict(
                    tokens=' '.join(tokens),
                    sentence=normalize_fn(transcript)
                ))

            # outputs[mode].sort(key=lambda item: len(item['target_word']))
            logger.info("Output to %s" % output_fn)
            with open(output_fn, 'w', encoding='utf-8') as f:
                f.write('\t'.join(['tokens', 'sentence']) + '\n')
                for o in outputs:
                    f.write('\t'.join([
                        o['tokens'],
                        o['sentence']
                    ]) + '\n')

    def evaluate(self, pred, ref, metric: str):
        if metric == "acc":
            return accuracy_score(pred, ref) * len(pred), len(pred)
        else:
            return super().evaluate(pred, ref, metric)

    def get_pytorch_wrapper(self, mode: str):
        return PytorchWikiText2(self, mode, self.params)


class PytorchWikiText2(PytorchDataset):
    def __init__(self, builder, mode, params):
        super().__init__(builder, mode, params)
        cfg = params.dataset
        super().__init__(builder, mode, params)

        self._vocab = Vocab(builder.get_vocab_path(params.dataset.unit))

        is_debug = mode == "debug"
        if mode == "debug":
            mode = "train"

        self._data = self.load_data(os.path.join(builder.get_processed_data_dir(), "%s_%s" % (cfg.unit, mode) + '.csv'))
        if is_debug:
            self._data = self._data[:cfg.debug_size]

    def load_data(self, csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[1:]
        lines = [l.split('\t') for l in lines if l != ""]
        logger.info("Number of texts: %d", len(lines))
        data = []
        tokens = []
        for l in lines:
            tokens += [int(w) for w in l[0].split(' ')] + [self._vocab.eos_token_idx]

        for i in range(len(tokens) - 1):
            seq_len = min(self.params.dataset.input_length, len(tokens) - i - 1)
            data.append(BatchItem(
                X=tokens[i:i + seq_len],
                Y=tokens[i + 1:i + seq_len + 1]
            ))
        return data

    def collate_fn(self, batch: List[BatchItem]):
        batch.sort(key=lambda item: len(item.X), reverse=True)

        inp = [LongTensor(item.X) for item in batch]
        tgt = [LongTensor(item.Y).view(-1) for item in batch]

        tgt_len = [len(t) for t in tgt]
        inp = nn.utils.rnn.pad_sequence(
            inp, batch_first=True)
        tgt = nn.utils.rnn.pad_sequence(
            tgt, batch_first=True,
            padding_value=self._vocab.blank_token_idx)

        return Batch(
            X=inp, X_len=LongTensor([len(item.X) for item in batch]),
            Y=tgt, Y_len=LongTensor(tgt_len))

    @property
    def vocab_size(self):
        return len(self._vocab)

    def format_output(self, y_pred, batch_item: BatchItem) -> (str, str, str):
        if self.cfg.output_format == "text":
            return ' '.join([self._vocab.get_token(word_id) for word_id in batch_item.X]), \
                   ' '.join([self._vocab.get_token(word_id) for word_id in batch_item.Y]), \
                   ' '.join([self._vocab.get_token(word_id) for word_id in y_pred])
        else:
            return super().format_output(y_pred, batch_item)