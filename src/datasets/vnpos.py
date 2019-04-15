"""vnPOS: Dataset for word segmentation and POS tagging
Available at http://vnlp.net/wp-content/uploads/2009/06/du-lieu-vnpos1.zip
"""

import os
import glob
import re
import torch
import shutil
from tqdm import tqdm

from utils.logging import logger
from utils.utils import maybe_download, maybe_unzip
from utils.metrics import ser
from utils.ops_utils import Tensor, LongTensor
from datasets.base.nlp import NLPDataset, load_idx_to_tkn, load_tkn_to_idx, \
    prepare_vocab_words, token_to_idx, normalize_string as normalize_word

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
        sent = normalize_word(sent)
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

    # punctuations = [',', '.', '-', '"', ':', '!', '(', ')', '...']
    # punc = [p + " " for p in punctuations]
    sentences = {'train': [], 'test': []}
    postags = {'train': [], 'test': []}
    for mode in ['train', 'test']:
        for file in glob.glob(os.path.join(path, "%s*" % mode)):
            for sent in open(file, encoding='utf-8'):
                sent = sent.strip()
                if sent == '':
                    continue
                words = [s.split('//')[0] for s in sent.split(' ')]
                tags = []
                for word in sent.split(' '):
                    if 'a' <= word.lower()[0] <= 'z':
                        tags.append(word.split('//')[1])
                    else:
                        tags.append("<punc>")
                sentences[mode].append(words)
                postags[mode].append(tags)

    prepare_vocab_chars(working_dir, sentences['train'])
    prepare_vocab_words(working_dir, sentences['train'], min_freq=1)
    prepare_vocab_words(working_dir, postags['train'], name="pos_tags", min_freq=1, default_tags=['<w>', '<oov>'])
    prepare_tag_list(working_dir)

    word_token_to_idx = load_tkn_to_idx(os.path.join(working_dir, "vocab", "words.txt"))
    pos_tag_to_idx = load_tkn_to_idx(os.path.join(working_dir, "vocab", "pos_tags.txt"))
    char_token_to_idx = load_tkn_to_idx(os.path.join(working_dir, "vocab", "chars.txt"))

    for mode in ['train', 'test']:
        data = []
        for p_tags, words in zip(postags[mode], sentences[mode]):
            line = ' '.join(words)
            chars = tokenize(line, 'char')
            line = normalize_word(line)
            words = []
            seg_tags = []
            pos_tags = []
            for w, tag in zip(line.split(' '), p_tags):
                if '_' in w:
                    w = w.split('_')
                    words += w
                    seg_tags += [1] + [0] * (len(w) - 1)
                    pos_tags += [tag] + [0] * (len(w) - 1)
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
                    ' '.join([str(token_to_idx(word_token_to_idx, w)) for w in words]),
                    ' '.join([str(t) for t in seg_tags]),
                    ' '.join([str(token_to_idx(pos_tag_to_idx, t)) for t in pos_tags]),
                    ' '.join([str(token_to_idx(char_token_to_idx, c)) for c in chars]),
                ]) + '\n')


class Dataset(NLPDataset):
    def __init__(self, mode, params, args=None):
        super().__init__(mode, params, args)
        self.working_dir = os.path.join("datasets", "vnpos", "data")

        # Load vocab and tag list
        self.char_to_idx = load_tkn_to_idx(os.path.join(self.working_dir, "vocab", "chars.txt"))
        self.word_to_idx = load_tkn_to_idx(os.path.join(self.working_dir, "vocab", "words.txt"))
        self.idx_to_word = load_idx_to_tkn(os.path.join(self.working_dir, "vocab", "words.txt"))
        self.tag_to_idx = load_tkn_to_idx(os.path.join(
            self.working_dir,
            "vocab",
            "seg_tags.txt" if self.params.dataset.tag_type == "seg" else "pos_tags.txt"))
        self.idx_to_tag = load_idx_to_tkn(os.path.join(
            self.working_dir,
            "vocab",
            "seg_tags.txt" if self.params.dataset.tag_type == "seg" else "pos_tags.txt"))

        if '<pad>' not in self.tag_to_idx:
            self.tag_to_idx["<pad>"] = len(self.tag_to_idx)
        if '<sos>' not in self.tag_to_idx:
            self.tag_to_idx["<sos>"] = len(self.tag_to_idx)
        if '<eos>' not in self.tag_to_idx:
            self.tag_to_idx["<eos>"] = len(self.tag_to_idx)

        self.vocab_char_size = len(self.char_to_idx)
        self.vocab_word_size = len(self.idx_to_word)
        self.num_tags = len(self.tag_to_idx)

        # Load data
        self.data = []
        if self.mode in ["test", "train"]:
            self.data = self.load_data_from_file(os.path.join(self.working_dir, self.mode + ".csv"))
        elif self.mode == "infer":
            self.data = []

    def load_from_input(self, inp):
        self.data = [dict(
            X=[token_to_idx(self.word_to_idx, w) for w in tokenize(sent[0], "word")],
            Y=[1] * len(tokenize(sent[0], "word"))
        ) for sent in inp]

    @classmethod
    def prepare(cls, force=False):
        super().prepare(force)
        working_dir = os.path.join("datasets", "vnpos")
        maybe_download(
            "data.zip",
            working_dir,
            DOWNLOAD_URL)
        maybe_unzip("data.zip", working_dir, "raw")

        if force:
            shutil.rmtree(os.path.join(working_dir, "data"))
            while os.path.exists(os.path.join(working_dir, "data")):
                pass

        maybe_preprocess(
            os.path.join(working_dir, "raw", "Du lieu vnPOS", "vnPOS"),
            os.path.join(working_dir, "data")
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
        word_tokens = [LongTensor(item['X']).view(-1) for item in batch]
        word_tags = [LongTensor(item['Y']).view(-1) for item in batch]
        word_tokens = torch.nn.utils.rnn.pad_sequence(
            word_tokens, batch_first=True,
            padding_value=self.word_to_idx["<pad>"])
        word_tags = torch.nn.utils.rnn.pad_sequence(
            word_tags, batch_first=True,
            padding_value=self.tag_to_idx["<pad>"])

        return dict(X=word_tokens, Y=word_tags)

    def evaluate(self, y_pred, batch, metric):
        ret = 0
        for k, predicted in enumerate(y_pred):
            ground_truth = batch['Y'][k].cpu()[1:]
            ground_truth = [i for i in ground_truth if i != self.tag_to_idx['<pad>']]
            predicted = predicted[:len(ground_truth)]
            if metric == 'ser':
                ret += ser(predicted, ground_truth, [self.tag_to_idx['<w>']])
        return ret / len(y_pred)

    def format_output(self, y_pred, inp, display=None):
        if display is None:
            return str(y_pred)

        if display == "word+delimiter":
            ret = []
            for word, tag in zip(inp['X'], y_pred):
                if word == self.word_to_idx["<pad>"]:
                    continue
                if tag == 1 and not ret:
                    ret.append('/')
                ret.append(self.idx_to_word[word])
            return ' '.join(ret[-1])

        if display == "word+tag":
            ret = []
            prev_tag = None
            for word, tag in zip(inp['X'], y_pred):
                if word == self.word_to_idx["<pad>"]:
                    continue
                ret.append(self.idx_to_word[word])
                if tag != self.tag_to_idx["<w>"]:
                    if prev_tag is not None:
                        ret.append('//' + self.idx_to_tag[prev_tag])
                    prev_tag = tag
            return ' '.join(ret)
