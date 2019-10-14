import os
import random
import sys
from typing import List

import torch.nn as nn
from tqdm import tqdm

from dlex.configs import MainConfig
from dlex.datasets.nlp.builder import NLPDataset
from dlex.datasets.nlp.utils import write_vocab, Vocab, nltk_tokenize
from dlex.datasets.torch import Dataset
from dlex.torch import Batch
from dlex.torch import BatchItem
from dlex.torch.utils.ops_utils import LongTensor
from dlex.utils.logging import logger

DOWNLOAD_URL = "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz"


class Newsgroup20(NLPDataset):
    def __init__(self, params: MainConfig):
        super().__init__(params)

    @property
    def output_prefix(self):
        return "word_embeddings_%s" % str(self.params.dataset.pretrained_embeddings).lower()

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        self.download_and_extract(DOWNLOAD_URL, self.get_raw_data_dir())

    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)
        if os.path.exists(self.get_processed_data_dir()):
            return

        texts = []  # list of text samples
        labels_index = {}  # dictionary mapping label name to numeric id
        labels = []  # list of label ids
        for name in sorted(os.listdir(os.path.join(self.get_raw_data_dir(), "20_newsgroup"))):
            path = os.path.join(self.get_raw_data_dir(), "20_newsgroup", name)
            if os.path.isdir(path):
                label_id = len(labels_index)
                labels_index[name] = label_id
                for fname in sorted(os.listdir(path)):
                    if fname.isdigit():
                        fpath = os.path.join(path, fname)
                        args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                        with open(fpath, **args) as f:
                            t = f.read()
                            i = t.find('\n\n')  # skip header
                            if 0 < i:
                                t = t[i:]
                        if t.strip() != "":
                            texts.append(t)
                            labels.append(label_id)

        num_validation_samples = int(0.2 * len(texts))
        texts = {'train': texts[num_validation_samples:], 'valid': texts[:num_validation_samples]}
        labels = {'train': labels[num_validation_samples:], 'valid': labels[:num_validation_samples]}

        if self.params.dataset.pretrained_embeddings is None:
            write_vocab(
                self.get_processed_data_dir(), texts['train'],
                output_file_name="word.txt",
                normalize_fn=lambda s: s.lower(),
                tokenize_fn=nltk_tokenize)
            vocab = Vocab(os.path.join(self.get_processed_data_dir(), "vocab", "word.txt"))
        else:
            vocab = Vocab()
            vocab.load_embeddings(self.params.dataset.pretrained_embeddings)

        self.write_dataset(
            self.output_prefix, texts, labels,
            vocab=vocab,
            normalize_fn=lambda s: s.lower(),
            tokenize_fn=nltk_tokenize)

    def get_vocab_path(self, unit):
        return os.path.join(self.get_processed_data_dir(), "vocab", "%s.txt" % unit)

    def write_dataset(self, output_prefix, texts, labels, vocab: Vocab, normalize_fn, tokenize_fn):
        for mode in texts.keys():
            outputs = []
            output_fn = os.path.join(self.get_processed_data_dir(), "%s_%s" % (output_prefix, mode) + '.csv')
            for text, label in tqdm(list(zip(texts[mode], labels[mode])), desc=mode):
                tokens = [str(vocab[tkn]) for tkn in tokenize_fn(normalize_fn(text))]
                outputs.append(dict(
                    tokens=' '.join(tokens),
                    sentence=normalize_fn(text),
                    label=str(label)
                ))

            # outputs[mode].sort(key=lambda item: len(item['target_word']))
            logger.info("Output to %s" % output_fn)
            with open(output_fn, 'w', encoding='utf-8') as f:
                f.write('\t'.join(list(outputs[0].keys())) + '\n')
                for o in outputs:
                    f.write('\t'.join(o.values()) + '\n')

    def evaluate(self, pred, ref, metric: str):
        if metric == "acc":
            return int(pred == ref), 1
        else:
            return super().evaluate(pred, ref, metric)

    def get_pytorch_wrapper(self, mode: str):
        return PytorchNewsgroup20(self, mode)


class PytorchNewsgroup20(Dataset):
    def __init__(self, builder, mode):
        super().__init__(builder, mode)
        cfg = self.params.dataset
        super().__init__(builder, mode)

        if self.params.dataset.pretrained_embeddings is None:
            self.vocab = Vocab(os.path.join(builder.get_processed_data_dir(), "vocab", "word.txt"))
        else:
            self.vocab = Vocab()
            self.vocab.load_embeddings(self.params.dataset.pretrained_embeddings)

        is_debug = mode == "debug"
        if mode == "debug":
            mode = "train"
        if mode == "test":
            mode = "valid"

        self._data = self.load_data(os.path.join(builder.get_processed_data_dir(), "%s_%s.csv" % (builder.output_prefix, mode)))
        if is_debug:
            self._data = self._data[:cfg.debug_size]

    def load_data(self, csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')[1:]
        lines = [l.split('\t') for l in lines if l != ""]
        logger.info("Number of texts: %d", len(lines))
        data = [BatchItem(
            X=[int(w) for w in l[0].split(' ') if w != ''],
            Y=int(l[2])
        ) for l in lines]
        random.shuffle(data)
        return data

    def collate_fn(self, batch: List[BatchItem]):
        batch.sort(key=lambda item: len(item.X), reverse=True)

        inp = [LongTensor(item.X) for item in batch]
        tgt = LongTensor([item.Y for item in batch])

        inp = nn.utils.rnn.pad_sequence(
            inp, batch_first=True)

        return Batch(
            X=inp, X_len=LongTensor([len(item.X) for item in batch]),
            Y=tgt)

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def embedding_dim(self):
        return self.vocab.embedding_dim

    @property
    def num_classes(self):
        return 20

    def format_output(self, y_pred, batch_item: BatchItem) -> (str, str, str):
        if self.configs.output_format == "text":
            return ' '.join([self.vocab.get_token(word_id) for word_id in batch_item.X]), \
                   ' '.join([self.vocab.get_token(word_id) for word_id in batch_item.Y]), \
                   ' '.join([self.vocab.get_token(word_id) for word_id in y_pred])
        else:
            return super().format_output(y_pred, batch_item)