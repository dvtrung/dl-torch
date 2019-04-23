"""Datasets for neural machine translation"""

import os

import nltk
import torch
from dl_torch.datasets.base.nlp import NLPDataset, load_tkn_to_idx, load_idx_to_tkn, \
    prepare_vocab_words, normalize_string, token_to_idx
from tqdm import tqdm
from dl_torch.utils.logging import logger
from dl_torch.utils.ops_utils import LongTensor
from dl_torch.utils.utils import maybe_download, maybe_unzip

DOWNLOAD_URL_FRA_ENG = "https://www.manythings.org/anki/fra-eng.zip"


def readLangs(filepath, reverse=False):
    logger.info("Reading data...")

    # Read the file and split into lines
    lines = open(filepath, encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s).split(' ') for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]

    return pairs


def filter_pair(p, max_length=10):
    return len(p[0]) < max_length and \
           len(p[1]) < max_length


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


class Tatoeba(NLPDataset):
    working_dir = os.path.join("datasets", "tatoeba", "fra-eng")
    raw_data_dir = os.path.join(working_dir, "raw")
    processed_data_dir = os.path.join(working_dir, "data")

    def __init__(self, mode, params, args=None):
        super().__init__(mode, params, args=args)
        self.working_dir = os.path.join("datasets", "tatoeba", "fra-eng", "data")
        self.lang = ('eng', 'fra')

        # Load vocab
        self.word_to_idx = {}
        self.idx_to_word = {}
        for lang in self.lang:
            self.word_to_idx[lang] = load_tkn_to_idx(
                os.path.join(self.working_dir, "vocab", lang + ".txt"))
            self.idx_to_word[lang] = load_idx_to_tkn(
                os.path.join(self.working_dir, "vocab", lang + ".txt"))

        self.sos_id = self.word_to_idx[self.lang[0]]['<sos>']
        self.eos_id = self.word_to_idx[self.lang[0]]['<eos>']
        self.input_size = len(self.word_to_idx[self.lang[0]])
        self.output_size = len(self.word_to_idx[self.lang[1]])

        # Load data
        if self.mode in ["test", "train"]:
            data = []
            fo = open(os.path.join(self.processed_data_dir, self.mode + ".csv"), "r", encoding='utf-8')
            fo.readline()  # header
            for line in tqdm(fo):
                line = line.strip().split(',')
                data.append(dict(
                    X=[int(i) for i in line[0].split(' ')],
                    Y=[int(i) for i in line[1].split(' ')]
                ))
            fo.close()
            self.data = data
        elif self.mode == "infer":
            self.data = []

    @classmethod
    def maybe_download_and_extract(cls, force=False):
        maybe_download(
            "data.zip",
            cls.working_dir,
            DOWNLOAD_URL_FRA_ENG)
        maybe_unzip("data.zip", cls.working_dir, "raw")

    @classmethod
    def maybe_preprocess(cls, force=False):
        super().prepare(force)

        if os.path.exists(cls.processed_data_dir):
            return
        pairs = readLangs(
            os.path.join(cls.working_dir, "raw", "fra.txt"),
            reverse=False)
        logger.info("Read %s sentence pairs", len(pairs))
        pairs = filter_pairs(pairs)
        logger.info("Trimmed to %s sentence pairs", len(pairs))

        os.makedirs(cls.processed_data_dir, exist_ok=True)
        lang = ('eng', 'fra')
        default_tags = ['<pad>', '<sos>', '<eos>', '<oov>']

        word_token_to_idx = {}
        for i in [0, 1]:
            prepare_vocab_words(
                cls.processed_data_dir,
                [_p[i] for _p in pairs],
                lang[i], 0, default_tags)
            word_token_to_idx[lang[i]] = load_tkn_to_idx(
                os.path.join(cls.processed_data_dir, "vocab", lang[i] + ".txt"))

        data = {
            'train': pairs[:10000],
            'test': pairs[10000:]
        }
        for mode in ['train', 'test']:
            with open(os.path.join(cls.processed_data_dir, "%s.csv" % mode), 'w') as fo:
                fo.write('lang1,lang2\n')
                for item in data[mode]:
                    fo.write(','.join([
                        ' '.join([str(token_to_idx(word_token_to_idx[lang[0]], w)) for w in item[0]]),
                        ' '.join([str(token_to_idx(word_token_to_idx[lang[1]], w)) for w in item[1]])
                    ]) + "\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        batch.sort(key=lambda item: len(item['X']), reverse=True)
        inp = [LongTensor(item['X']).view(-1) for item in batch]
        tgt = [LongTensor(item['Y']).view(-1) for item in batch]
        inp = torch.nn.utils.rnn.pad_sequence(
            inp, batch_first=True,
            padding_value=self.word_to_idx[self.lang[0]]["<eos>"])
        tgt = torch.nn.utils.rnn.pad_sequence(
            tgt, batch_first=True,
            padding_value=self.word_to_idx[self.lang[1]]["<eos>"])

        return dict(
            X=inp, X_len=[LongTensor(len(item['X'])) for item in batch],
            Y=tgt, Y_len=[LongTensor(len(item['Y'])) for item in batch])

    def _trim_result(self, ls):
        start = 0 if ls[0] != self.sos_id else 1
        end = 0
        while end < len(ls) and ls[end] != self.eos_id:
            end += 1
        return ls[start:end]

    def evaluate(self, y_pred, batch, metric):
        if metric == "bleu":
            target_variables = batch['Y']
            score, total = 0, 0
            for k, _y_pred in enumerate(y_pred):
                target = self._trim_result(target_variables[k].cpu().detach().numpy().tolist())
                predicted = self._trim_result(_y_pred)
                score += nltk.translate.bleu_score.sentence_bleu([target], predicted)
                total += 1
            return score, total
