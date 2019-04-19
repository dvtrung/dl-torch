"""NLP Dataset"""

import os
import re
import unicodedata

from datasets.base import BaseDataset


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def load_tkn_to_idx(filename):
    tkn_to_idx = {}
    fo = open(filename, encoding='utf-8')
    for line in fo:
        line = line.strip()
        if line == "":
            continue
        tkn_to_idx[line] = len(tkn_to_idx)
    fo.close()
    return tkn_to_idx


def load_idx_to_tkn(filename):
    idx_to_tkn = []
    fo = open(filename, encoding='utf-8')
    for line in fo:
        line = line.strip()
        if line == "":
            continue
        idx_to_tkn.append(line)
    fo.close()
    return idx_to_tkn


def prepare_vocab_words(
        working_dir,
        sentences,
        name="words",
        min_freq=0,
        default_tags=['<pad>', '<sos>', '<eos>', '<oov>']):
    os.makedirs(os.path.join(working_dir, "vocab"), exist_ok=True)
    word_freqs = {}
    for ls in sentences:
        ls = ' '.join(ls).replace('_', ' ').split(' ')
        for word in ls:
            if word.strip() == '':
                continue
            if word in word_freqs:
                word_freqs[word] += 1
            else:
                word_freqs[word] = 1

    words = list([word for word in word_freqs if word_freqs[word] > min_freq])
    with open(os.path.join(working_dir, "vocab", name + ".txt"), "w", encoding='utf-8') as fo:
        fo.write('\n'.join(default_tags) + '\n')
        fo.write("\n".join(sorted(words)))


def normalize_string(sent):
    # x = re.sub("[^ a-zA-Z0-9\uAC00-\uD7A3]+", " ", x)
    # x = re.sub("[\u3040-\u30FF]+", "\u3042", x) # convert Hiragana and Katakana to あ
    # x = re.sub("[\u4E00-\u9FFF]+", "\u6F22", x) # convert CJK unified ideographs to 漢
    sent = unicodeToAscii(sent.lower().strip())
    sent = re.sub(r"([.!?])", r" \1", sent)
    sent = re.sub(r"[^a-zA-Z.!?]+", r" ", sent)
    sent = re.sub("\s+", " ", sent)
    sent = re.sub("^ | $", "", sent)

    words = sent.split(' ')
    ret = []
    for word in words:
        if all(c <= 'a' or c >= 'z' for c in word):
            ret.append('<non-word>')
        else:
            ret.append(word)
    return ' '.join(ret)


def token_to_idx(d, w): return d[w] if w in d else d['<oov>']


class NLPDataset(BaseDataset):
    def __init__(self, mode, params):
        super().__init__(mode, params)
