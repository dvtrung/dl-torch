import os, glob, re
from utils.logging import logger
from utils.utils import maybe_download, maybe_unzip
from utils.metrics import ser
import torch

from datasets.base import BaseDataset
from utils.ops_utils import Tensor, LongTensor

DOWNLOAD_URL = "http://vnlp.net/wp-content/uploads/2009/06/du-lieu-vnpos1.zip"

def normalize_word(x):
    # x = re.sub("[^ a-zA-Z0-9\uAC00-\uD7A3]+", " ", x)
    # x = re.sub("[\u3040-\u30FF]+", "\u3042", x) # convert Hiragana and Katakana to あ
    # x = re.sub("[\u4E00-\u9FFF]+", "\u6F22", x) # convert CJK unified ideographs to 漢
    x = re.sub("\s+", " ", x)
    x = re.sub("^ | $", "", x)
    x = x.lower()

    x = x.split(' ')
    ret = []
    for w in x:
        if all(c <= 'a' or c >='z' for c in w):
            ret.append('<non-word>')
        else:
            ret.append(w)
    return ' '.join(ret)

def normalize_char(x):
    # x = re.sub("[^ a-zA-Z0-9\uAC00-\uD7A3]+", " ", x)
    # x = re.sub("[\u3040-\u30FF]+", "\u3042", x) # convert Hiragana and Katakana to あ
    # x = re.sub("[\u4E00-\u9FFF]+", "\u6F22", x) # convert CJK unified ideographs to 漢
    x = re.sub("\s+", " ", x)
    x = re.sub("^ | $", "", x)
    x = x.strip().lower()

    x = list(x)
    return ''.join(x)

def tokenize(x, unit):
    if unit == "char":
        x = normalize_char(x)
        return re.sub(" ", "", x)
    if unit == "word":
        x = normalize_word(x)
        return re.split(" |_", x)

def load_tkn_to_idx(filename):
    print("loading %s" % filename)
    tkn_to_idx = {}
    fo = open(filename, encoding='utf-8')
    for line in fo:
        line = line.strip()
        if line == "": continue
        tkn_to_idx[line] = len(tkn_to_idx)
    fo.close()
    return tkn_to_idx

def load_idx_to_tkn(filename):
    print("loading %s" % filename)
    idx_to_tkn = []
    fo = open(filename, encoding='utf-8')
    for line in fo:
        line = line.strip()
        if line == "": continue
        idx_to_tkn.append(line)
    fo.close()
    return idx_to_tkn

def prepare_vocab_chars(working_directory):
    f = open(os.path.join(working_directory, "train.txt"), encoding='utf-8')
    chars = set()
    for line in f:
        line = line.strip().lower()
        chars |= set(line)

    with open(os.path.join(working_directory, "vocab", "chars.txt"), "w", encoding='utf-8') as fo:
        fo.write('<sos>\n<eos>\n<pad>\n<oov>\n')
        fo.write("\n".join(list(chars)))


def prepare_vocab_words(working_directory):
    f = open(os.path.join(working_directory, "train.txt"), encoding='utf-8')
    words = {}
    for line in f:
        ls = set(tokenize(line, 'word'))
        for word in ls:
            if word.strip() == '': continue
            if word in words: words[word] += 1
            else: words[word] = 1

    words = list([word for word in words if words[word] > 2])
    with open(os.path.join(working_directory, "vocab", "words.txt"), "w", encoding='utf-8') as fo:
        fo.write('<sos>\n<eos>\n<pad>\n<oov>\n')
        fo.write("\n".join(words))


def prepare_tag_list(working_directory):
    with open(os.path.join(working_directory, "vocab", "tags.txt"), "w", encoding='utf-8') as fo:
        fo.write('<w>\n<sow>')


def prepare_dataset(working_directory):
    word_token_to_idx = load_tkn_to_idx(os.path.join(working_directory, "vocab", "words.txt"))
    char_token_to_idx = load_tkn_to_idx(os.path.join(working_directory, "vocab", "chars.txt"))

    for dataset in ["train", "test"]:
        data = []
        f = open(os.path.join(working_directory, "%s.txt" % dataset), encoding='utf-8')
        for line in f:
            chars = tokenize(line, 'char')
            line = normalize_word(line)
            words = []
            wtags = []
            for w in line.split(' '):
                if '_' in w:
                    w = w.split('_')
                    words += w
                    wtags += [1] + [0] * (len(w) - 1)
                else:
                    words.append(w)
                    wtags.append(1)
            data.append((words, wtags, chars))

        with open(os.path.join(working_directory, "%s.csv" % dataset), "w", encoding='utf-8') as fo:
            fo.write('wtokens,wtags,ctokens\n')
            data.sort(key=lambda d: len(d[0]), reverse=True)
            for words, wtags, chars in data:
                fo.write(','.join([
                    ' '.join([str(word_token_to_idx[w] if w in word_token_to_idx else word_token_to_idx['<oov>']) for w in words]),
                    ' '.join([str(t) for t in wtags]),
                    ' '.join([str(char_token_to_idx[c] if c in char_token_to_idx else char_token_to_idx['<oov>']) for c in chars]),
                ]) + '\n')

def maybe_preprocess(path, working_directory):
    if os.path.exists(working_directory): return
    os.makedirs(working_directory)
    os.mkdir(os.path.join(working_directory, "vocab"))

    punctuations = [',', '.', '-', '"', ':', '!', '(', ')', '...']
    punc = [p + " " for p in punctuations]
    for type in ['train', 'test']:
        sentences = []
        for file in glob.glob(os.path.join(path, "%s*" % (type))):
            for sent in open(file, encoding='utf-8'):
                sent = sent.strip()
                try:
                    if sent == '': continue
                    ls = sent.split('//')[:-1]
                    for i in range(1, len(ls)):
                        s = ls[i]
                        if 'A' <= s[0] <= 'Z' and ' ' in s: s = s[s.index(' ') + 1:]
                        if len(s) > 2 and s[:2] in punc: s = s[2:]
                        ls[i] = s
                    # print(s)
                    sentences.append(' '.join(ls))
                except:
                    continue

        with open(os.path.join(working_directory, "%s.txt" % (type)), "w", encoding='utf-8') as f:
            f.write('\n'.join(sentences))

    prepare_vocab_chars(working_directory)
    prepare_vocab_words(working_directory)
    prepare_tag_list(working_directory)
    prepare_dataset(working_directory)

class Dataset(BaseDataset):
    def __init__(self, mode, params):
        super().__init__(mode, params)
        self.working_directory = os.path.join("datasets", "vnpos", "data")

        self.load_vocab(params)
        self.data = self.load_data(os.path.join(self.working_directory, self.mode + ".csv"))

    @classmethod
    def prepare(cls):
        working_directory = os.path.join("datasets", "vnpos")

        maybe_download(
            "data.zip",
            working_directory,
            DOWNLOAD_URL)
        maybe_unzip("data.zip", working_directory, "raw")
        maybe_preprocess(
            os.path.join(working_directory, "raw", "Du lieu vnPOS", "vnPOS"),
            os.path.join(working_directory, "data")
        )

    def load_vocab(self, params):
        self.char_to_idx = load_tkn_to_idx(os.path.join(self.working_directory, "vocab", "chars.txt"))
        self.word_to_idx = load_tkn_to_idx(os.path.join(self.working_directory, "vocab", "words.txt"))
        self.idx_to_word = load_idx_to_tkn(os.path.join(self.working_directory, "vocab", "words.txt"))
        self.tag_to_idx = load_tkn_to_idx(os.path.join(self.working_directory, "vocab", "tags.txt"))
        self.tag_to_idx["<pad>"] = len(self.tag_to_idx)
        self.tag_to_idx["<sos>"] = len(self.tag_to_idx)
        self.tag_to_idx["<eos>"] = len(self.tag_to_idx)

        self.vocab_char_size = len(self.char_to_idx)
        self.vocab_word_size = len(self.idx_to_word)
        self.num_tags = len(self.tag_to_idx)

    def load_data(self, path):
        cfg = self.params.dataset

        data = []
        batch_cx = [] # character input
        batch_wx = [] # word input
        batch_y = []
        cx_maxlen = 0 # maximum length of character sequence
        wx_maxlen = 0 # maximum length of word sequence

        fo = open(path, "r", encoding='utf-8')
        fo.readline()  # header
        for line in fo:
            line = line.strip().split(',')
            data.append(dict(
                wtokens=[int(i) for i in line[0].split(' ')],
                wtags=[self.tag_to_idx["<sos>"]] + [int(i) for i in line[1].split(' ')]
            ))

        for line in []:# fo:
            line = line.strip().split(',')
            wtokens = [int(i) for i in line[0].split(' ')]
            wtags = [int(i) for i in line[1].split(' ')]
            wx_len = len(wtokens)
            wx_maxlen = wx_maxlen if wx_maxlen else wx_len
            wx_pad = [self.params.pad_idx] * (wx_maxlen - wx_len)
            batch_wx.append(wtokens + wx_pad)
            batch_y.append([self.tag_to_idx["<sos>"]] + wtags + wx_pad)
            if self.params.embed_unit[:4] == "char":
                cx = [idx_to_word[i] for i in wx]
                cx_maxlen = max(cx_maxlen, len(max(cx, key = len)))
                batch_cx.append([[self.params.sos_idx] + [char_to_idx[c] for c in w] + [self.params.eos_idx] for w in cx])
            if len(batch_wx) == cfg.batch_size:
                if self.params.embed_unit[:4] == "char":
                    for cx in batch_cx:
                        for w in cx:
                            w += [self.tag_to_idx["<pad>"]] * (cx_maxlen - len(w) + 2)
                        cx += [[self.tag_to_idx["<pad>"]] * (cx_maxlen + 2)] * (wx_maxlen - len(cx))

                data.append((torch.LongTensor(batch_cx).cuda(), torch.LongTensor(batch_wx).cuda(), torch.LongTensor(batch_y).cuda())) # append a mini-batch
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
        wtokens = [LongTensor(item['wtokens']).view(-1) for item in batch]
        wtags = [LongTensor(item['wtags']).view(-1) for item in batch]
        wtokens = torch.nn.utils.rnn.pad_sequence(wtokens, batch_first=True, padding_value=self.word_to_idx["<pad>"])
        wtags = torch.nn.utils.rnn.pad_sequence(wtags, batch_first=True, padding_value=self.tag_to_idx["<pad>"])

        return dict(wtokens=wtokens, wtags=wtags)

    def eval(self, y_pred, batch, metrics):
        ret = 0
        for k in range(len(y_pred)):
            pr = y_pred[k]
            gt = batch['wtags'][k].cpu()[1:]
            gt = [i for i in gt if i != self.tag_to_idx['<pad>']]
            pr = pr[:len(gt)]
            ret += ser(pr, gt, [self.tag_to_idx['<sow>']])
        return ret
