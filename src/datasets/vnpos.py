import os, glob, re
from utils.logging import logger
from utils.utils import maybe_download, maybe_unzip
from utils.metrics import ser
import torch
import shutil
from tqdm import tqdm

from utils.ops_utils import Tensor, LongTensor
from datasets.base import NLPDataset

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
    tkn_to_idx = {}
    fo = open(filename, encoding='utf-8')
    for line in fo:
        line = line.strip()
        if line == "": continue
        tkn_to_idx[line] = len(tkn_to_idx)
    fo.close()
    return tkn_to_idx

def load_idx_to_tkn(filename):
    idx_to_tkn = []
    fo = open(filename, encoding='utf-8')
    for line in fo:
        line = line.strip()
        if line == "": continue
        idx_to_tkn.append(line)
    fo.close()
    return idx_to_tkn

def prepare_vocab_chars(working_dir, sentences):
    chars = set()
    for line in sentences:
        line = ' '.join(line)
        line = line.strip().lower()
        chars |= set(line)

    with open(os.path.join(working_dir, "vocab", "chars.txt"), "w", encoding='utf-8') as fo:
        fo.write('<sos>\n<eos>\n<pad>\n<oov>\n')
        fo.write("\n".join(sorted(list(chars))))


def prepare_vocab_words(
    working_dir,
    sentences,
    name="words",
    min_freq=0,
    default_tags=['<sos>', '<eos>', '<pad>', '<oov>']):

    words = {}
    for ls in sentences:
        ls = ' '.join(ls).replace('_', ' ').split(' ')
        for word in ls:
            if word.strip() == '': continue
            if word in words: words[word] += 1
            else: words[word] = 1

    words = list([word for word in words if words[word] > min_freq])
    with open(os.path.join(working_dir, "vocab", name + ".txt"), "w", encoding='utf-8') as fo:
        fo.write('\n'.join(default_tags) + '\n')
        fo.write("\n".join(sorted(words)))


def prepare_tag_list(working_dir):
    with open(os.path.join(working_dir, "vocab", "seg_tags.txt"), "w", encoding='utf-8') as fo:
        fo.write('<w>\n<sow>')


def token_to_idx(d, w): return d[w] if w in d else d['<oov>']


def maybe_preprocess(path, working_dir):
    if os.path.exists(working_dir): return

    logger.info("Preprocess data")
    os.makedirs(working_dir)
    os.mkdir(os.path.join(working_dir, "vocab"))

    # punctuations = [',', '.', '-', '"', ':', '!', '(', ')', '...']
    # punc = [p + " " for p in punctuations]
    sentences = {'train': [], 'test': []}
    postags = {'train': [], 'test': []}
    for mode in ['train', 'test']:
        for file in glob.glob(os.path.join(path, "%s*" % (mode))):
            for sent in open(file, encoding='utf-8'):
                sent = sent.strip()
                try:
                    if sent == '': continue
                    words = [s.split('//')[0] for s in sent.split(' ')]
                    tags = []
                    for s in sent.split(' '):
                        if 'a' <= s.lower()[0] <= 'z':
                            tags.append(s.split('//')[1])
                        else:
                            tags.append("<punc>")
                    # print(s)
                    sentences[mode].append(words)
                    postags[mode].append(tags)
                except:
                    continue

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
        self.load_vocab()
        self.load_data()


    def load_data(self):
        if self.mode in ["test", "train"]:
            self.data = self.load_data_from_file(os.path.join(self.working_dir, self.mode + ".csv"))
        elif self.mode == "infer":
            self.data = [dict(
                word_tokens=[token_to_idx(self.word_to_idx, w) for w in tokenize(sent[0], "word")],
                word_tags=[1] * len(tokenize(sent[0], "word"))
            ) for sent in self.args.input]

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

    def load_vocab(self):
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

    def load_data_from_file(self, path):
        cfg = self.params.dataset

        data = []
        batch_cx = [] # character input
        batch_wx = [] # word input
        batch_y = []
        cx_maxlen = 0 # maximum length of character sequence
        wx_maxlen = 0 # maximum length of word sequence

        fo = open(path, "r", encoding='utf-8')
        fo.readline()  # header
        for line in tqdm(fo):
            line = line.strip().split(',')
            tags = (line[1] if self.params.dataset.tag_type == "seg" else line[2]).split(' ')
            data.append(dict(
                word_tokens=[int(i) for i in line[0].split(' ')],
                word_tags=[self.tag_to_idx["<sos>"]] + [int(t) for t in tags]
            ))

        for line in []:# fo:
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
        word_tokens = [LongTensor(item['word_tokens']).view(-1) for item in batch]
        word_tags = [LongTensor(item['word_tags']).view(-1) for item in batch]
        word_tokens = torch.nn.utils.rnn.pad_sequence(word_tokens, batch_first=True, padding_value=self.word_to_idx["<pad>"])
        word_tags = torch.nn.utils.rnn.pad_sequence(word_tags, batch_first=True, padding_value=self.tag_to_idx["<pad>"])

        return dict(word_tokens=word_tokens, word_tags=word_tags)

    def eval(self, y_pred, batch, metric):
        ret = 0
        for k in range(len(y_pred)):
            pr = y_pred[k]
            gt = batch['word_tags'][k].cpu()[1:]
            gt = [i for i in gt if i != self.tag_to_idx['<pad>']]
            pr = pr[:len(gt)]
            # print(pr, gt, ser(pr, gt, [self.tag_to_idx['<sow>']]))
            if metric == 'ser':
                ret += ser(pr, gt, [self.tag_to_idx['<w>']])
        return ret

    def format_output(self, y_pred, inp, display=None):
        if display is None:
            return str(y)
        elif display == "word+delimiter":
            ret = []
            for w, t in zip(inp['word_tokens'], y_pred):
                if w == self.word_to_idx["<pad>"]: continue
                if t == 1 and len(ret) > 0:
                    ret.append('/')
                ret.append(self.idx_to_word[w])
            return ' '.join(ret[-1])
        elif display == "word+tag":
            ret = []
            prev_tag = None
            for w, t in zip(batch['word_tokens'], y_pred):
                if w == self.word_to_idx["<pad>"]: continue
                ret.append(self.idx_to_word[w])
                if t != self.tag_to_idx["<w>"]:
                    if prev_tag is not None:
                        ret.append('//' + self.idx_to_tag[prev_tag])
                    prev_tag = t
            return ' '.join(ret)
