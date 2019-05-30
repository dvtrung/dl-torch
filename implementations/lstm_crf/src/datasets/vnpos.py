"""vnPOS: Dataset for word segmentation and POS tagging
Available at http://vnlp.net/wp-content/uploads/2009/06/du-lieu-vnpos1.zip
"""

import os
import glob
import re
import tempfile
import torch
import shutil
from tqdm import tqdm
import numpy as np

from dlex.utils.logging import logger
from dlex.utils.utils import maybe_download, maybe_unzip
from dlex.utils.metrics import ser
from dlex.utils.ops_utils import LongTensor
from dlex.datasets.base.nlp import NLPDataset, load_idx_to_tkn, load_tkn_to_idx, \
    write_vocab, get_token_id, normalize_string, normalize_word

DOWNLOAD_URL = "http://vnlp.net/wp-content/uploads/2009/06/du-lieu-vnpos1.zip"


def normalize_char(sent):
    # x = re.sub("[^ a-zA-Z0-9\uAC00-\uD7A3]+", " ", x)
    # x = re.sub("[\u3040-\u30FF]+", "\u3042", x) # convert Hiragana and Katakana to あ
    # x = re.sub("[\u4E00-\u9FFF]+", "\u6F22", x) # convert CJK unified ideographs to 漢
    sent = re.sub("\s+", " ", sent)
    sent = re.sub("^ | $", "", sent)
    sent = sent.strip().lower()

    chars = list(sent)
    return ''.join(chars)


def tokenize(sent, unit):
    if unit == "char":
        sent = normalize_char(sent)
        return re.sub(" ", "", sent)
    if unit == "word":
        sent = normalize_string(sent)
        return re.split(" |_", sent)


def prepare_vocab_chars(working_dir, sentences):
    chars = set()
    for line in sentences:
        line = ' '.join(line)
        line = line.strip().lower()
        chars |= set(line)

    with open(os.path.join(working_dir, "vocab", "chars.txt"), "w", encoding='utf-8') as fo:
        fo.write('<sos>\n<eos>\n<pad>\n<oov>\n')
        fo.write("\n".join(sorted(list(chars))))


def prepare_tag_list(working_dir):
    with open(os.path.join(working_dir, "vocab", "seg_tags.txt"), "w", encoding='utf-8') as fo:
        fo.write('<w>\n<sow>')


def maybe_preprocess(path, working_dir):
    if os.path.exists(working_dir):
        return

    logger.info("Preprocess data")
    os.makedirs(working_dir)
    os.mkdir(os.path.join(working_dir, "vocab"))

    original_sentences = {'train': [], 'test': []}
    sentences = {'train': [], 'test': []}
    postags = {'train': [], 'test': []}
    for mode in ['train', 'test']:
        for file in glob.glob(os.path.join(path, "%s*" % mode)):
            for sent in open(file, encoding='utf-8'):
                sent = sent.strip()
                if sent == '':
                    continue
                words = [normalize_word(s.split('//')[0]) for s in sent.split(' ')]
                tags = []
                for word in sent.split(' '):
                    word = word.lower()
                    if 'a' <= word[0] <= 'z':
                        if len(word.split('//')[1]) <= 2:
                            tags.append(word.split('//')[1])
                    else:
                        tags.append("<punc>")
                original_sentences[mode].append(sent)
                sentences[mode].append(words)
                postags[mode].append(tags)

    prepare_vocab_chars(working_dir, sentences['train'])
    write_vocab(working_dir, sentences['train'], min_freq=0)
    write_vocab(working_dir, postags['train'], name="pos_tags", min_freq=0, default_tags=['<w>', '<oov>'])
    prepare_tag_list(working_dir)

    word_token_to_idx = load_tkn_to_idx(os.path.join(working_dir, "vocab", "words.txt"))
    pos_tag_to_idx = load_tkn_to_idx(os.path.join(working_dir, "vocab", "pos_tags.txt"))
    char_token_to_idx = load_tkn_to_idx(os.path.join(working_dir, "vocab", "chars.txt"))

    for mode in ['train', 'test']:
        data = []
        for p_tags, compound_words, sent in \
                zip(postags[mode], sentences[mode], original_sentences[mode]):
            chars = tokenize(sent, 'char')
            words = []
            seg_tags = []
            pos_tags = []
            for w, tag in zip(compound_words, p_tags):
                if '_' in w:
                    w = w.split('_')
                    words += w
                    seg_tags += [1] + [0] * (len(w) - 1)
                    pos_tags += [tag] + ['<w>'] * (len(w) - 1)
                else:
                    words.append(w)
                    seg_tags.append(1)
                    pos_tags.append(tag)

            data.append((words, seg_tags, pos_tags, chars))

        with open(os.path.join(working_dir, "%s.csv" % mode), "w", encoding='utf-8') as fo:
            fo.write('word_tokens,word_seg_tags,word_pos_tags,char_tokens\n')
            data.sort(key=lambda d: len(d[0]), reverse=True)
            for words, seg_tags, pos_tags, chars in data:
                fo.write(','.join([
                    ' '.join([str(get_token_id(word_token_to_idx, w)) for w in words]),
                    ' '.join([str(t) for t in seg_tags]),
                    ' '.join([str(get_token_id(pos_tag_to_idx, t)) for t in pos_tags]),
                    ' '.join([str(get_token_id(char_token_to_idx, c)) for c in chars]),
                ]) + '\n')


class VNPos(NLPDataset):
    def __init__(self, mode, params):
        super().__init__(mode, params)

        # Load vocab and tag list
        self.char_to_idx = load_tkn_to_idx(os.path.join(self.get_processed_data_dir(), "vocab", "chars.txt"))
        self.word_to_idx = load_tkn_to_idx(os.path.join(self.get_processed_data_dir(), "vocab", "words.txt"))
        self.idx_to_word = load_idx_to_tkn(os.path.join(self.get_processed_data_dir(), "vocab", "words.txt"))
        self.tag_to_idx = load_tkn_to_idx(os.path.join(
            self.get_processed_data_dir(),
            "vocab",
            "seg_tags.txt" if self.params.dataset.tag_type == "seg" else "pos_tags.txt"))
        self.idx_to_tag = load_idx_to_tkn(os.path.join(
            self.get_processed_data_dir(),
            "vocab",
            "seg_tags.txt" if self.params.dataset.tag_type == "seg" else "pos_tags.txt"))

        def _add_tag(tag):
            self.tag_to_idx[tag] = len(self.tag_to_idx)
            self.idx_to_tag.append(tag)

        if '<pad>' not in self.tag_to_idx:
            _add_tag('<pad>')
        if '<sos>' not in self.tag_to_idx:
            _add_tag('<sos>')
        if '<eos>' not in self.tag_to_idx:
            _add_tag('<eos>')

        self.vocab_char_size = len(self.char_to_idx)
        self.vocab_word_size = len(self.idx_to_word)
        self.num_tags = len(self.tag_to_idx)

        # Load data
        self.data = []
        if self.mode in ["test", "train"]:
            self.data = self.load_data_from_file(os.path.join(self.get_processed_data_dir(), self.mode + ".csv"))
        elif self.mode == "infer":
            self.data = []

    def load_from_input(self, inp):
        sent = inp
        self.data = [dict(
            X=[get_token_id(self.word_to_idx, w) for w in normalize_string(sent).split(' ')],
            Y=[1] * len(normalize_string(sent).split(' '))
        )]

    @classmethod
    def maybe_download_and_extract(cls, force=False):
        super().maybe_download_and_extract(force)
        if not os.path.exists(cls.get_raw_data_dir()):
            cls.download_and_extract(
                DOWNLOAD_URL, cls.get_raw_data_dir())

    @classmethod
    def maybe_preprocess(cls, force=False):
        super().maybe_preprocess(force)
        if os.path.exists(cls.get_processed_data_dir()):
            return
        maybe_preprocess(
            os.path.join(cls.get_raw_data_dir(), "Du lieu vnPOS", "vnPOS"),
            cls.get_processed_data_dir()
        )

    def load_data_from_file(self, path):
        cfg = self.params.dataset

        data = []
        batch_cx = []  # character input
        batch_wx = []  # word input
        batch_y = []
        cx_maxlen = 0  # maximum length of character sequence
        wx_maxlen = 0  # maximum length of word sequence

        fo = open(path, "r", encoding='utf-8')
        fo.readline()  # header
        for line in tqdm(fo):
            line = line.strip().split(',')
            tags = (line[1] if self.params.dataset.tag_type == "seg" else line[2]).split(' ')
            data.append(dict(
                X=[int(i) for i in line[0].split(' ')],
                Y=[self.tag_to_idx["<sos>"]] + [int(t) for t in tags]
            ))

        for line in []:  # fo:
            line = line.strip().split(',')
            word_tokens = [int(i) for i in line[0].split(' ')]
            word_tags = [int(i) for i in line[1].split(' ')]
            wx_len = len(word_tokens)
            wx_maxlen = wx_maxlen if wx_maxlen else wx_len
            wx_pad = [self.params.pad_idx] * (wx_maxlen - wx_len)
            batch_wx.append(word_tokens + wx_pad)
            batch_y.append([self.tag_to_idx["<sos>"]] + word_tags + wx_pad)
            if self.params.embed_unit[:4] == "char":
                cx = [idx_to_word[i] for i in wx]
                cx_maxlen = max(cx_maxlen, len(max(cx, key=len)))
                batch_cx.append(
                    [[self.params.sos_idx] + [char_to_idx[c] for c in w] + [self.params.eos_idx] for w in cx])
            if len(batch_wx) == cfg.batch_size:
                if self.params.embed_unit[:4] == "char":
                    for cx in batch_cx:
                        for w in cx:
                            w += [self.tag_to_idx["<pad>"]] * (cx_maxlen - len(w) + 2)
                        cx += [[self.tag_to_idx["<pad>"]] * (cx_maxlen + 2)] * (wx_maxlen - len(cx))

                data.append((torch.LongTensor(batch_cx).cuda(), torch.LongTensor(batch_wx).cuda(),
                             torch.LongTensor(batch_y).cuda()))  # append a mini-batch
                batch_cx = []
                batch_wx = []
                batch_y = []
                cx_maxlen = 0
                wx_maxlen = 0
        fo.close()
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        X = [LongTensor(item['X']).view(-1) for item in batch]
        X_len = [len(item['X']) for item in batch]
        Y = [LongTensor(item['Y']).view(-1) for item in batch]
        X = torch.nn.utils.rnn.pad_sequence(
            X, batch_first=True,
            padding_value=self.word_to_idx["<pad>"])
        Y = torch.nn.utils.rnn.pad_sequence(
            Y, batch_first=True,
            padding_value=self.tag_to_idx["<pad>"])

        return dict(
            X=X,
            X_len=X_len,
            Y=Y)

    def evaluate(self, y_pred, batch, metric):
        correct_total = 0
        count_total = 0
        for k, predicted in enumerate(y_pred):
            ground_truth = batch['Y'][k].cpu()[1:]
            ground_truth = [i for i in ground_truth if i != self.tag_to_idx['<pad>']]
            predicted = predicted[:len(ground_truth)]
            if metric == 'ser':
                ground_truth = np.array(ground_truth)
                if '<punc>' in self.tag_to_idx:
                    mask = ground_truth != self.tag_to_idx['<punc>']
                    predicted = np.array(predicted)[mask]
                    ground_truth = ground_truth[mask]

                correct, count = ser(predicted, ground_truth, [self.tag_to_idx['<w>']])
                correct_total += correct
                count_total += count
        return correct_total, count_total

    def format_output(self, y_pred, inp):
        if y_pred[0] == self.tag_to_idx['<sos>']:
            y_pred = y_pred[1:]

        if self.cfg.output_format is None:
            return str(y_pred)

        if self.cfg.output_format == "word+delimiter":
            ret = []
            for word_id, tag in zip(inp['X'], y_pred):
                if word_id == self.word_to_idx["<pad>"]:
                    break
                if tag == self.tag_to_idx['<sow>'] and ret:
                    ret.append('/')
                ret.append(self.idx_to_word[word_id])
            return ' '.join(ret)

        if self.cfg.output_format == "word+tag":
            ret = []
            prev_tag = None
            for word_id, tag in zip(inp['X'], y_pred):
                if word_id == self.word_to_idx["<pad>"]:
                    break  # end of reference

                if tag != self.tag_to_idx["<w>"]:  # new tag
                    if prev_tag is not None:
                        ret.append('[%s]' % self.idx_to_tag[prev_tag])
                    prev_tag = tag
                ret.append(self.idx_to_word[word_id])
            if prev_tag:
                ret.append('[%s]' % self.idx_to_tag[prev_tag])
            return ' '.join(ret)
