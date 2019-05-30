"""NLP Dataset"""

import os
import re
import unicodedata

from .base import BaseDataset


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


def normalize_string(sentence):
    """
    :param str sentence:
    :return: normalized sentence, separated by space
    :rtype str
    """
    # x = re.sub("[^ a-zA-Z0-9\uAC00-\uD7A3]+", " ", x)
    # x = re.sub("[\u3040-\u30FF]+", "\u3042", x) # convert Hiragana and Katakana to あ
    # x = re.sub("[\u4E00-\u9FFF]+", "\u6F22", x) # convert CJK unified ideographs to 漢
    sent = unicodeToAscii(sentence.lower().strip())
    sent = re.sub(r"([.!?,])", r" \1", sent)
    sent = re.sub(r"[^a-zA-Z.!?,]+", r" ", sent)
    sent = re.sub(r"\s+", " ", sent)
    sent = re.sub("^ | $", "", sent)

    words = sent.split(' ')
    ret = []
    for word in words:
        ret.append(normalize_word(word))
    return ' '.join(ret)


def normalize_word(word):
    punctuations = [',', '.', '-', '"', ':', '!', '(', ')', '...', '?']
    if word in ',.!?':
        return word
    elif word in punctuations:
        return '<punc>'
    elif any('0' <= c <= '9' for c in word):
        return '<non-word>'
    else:
        return word.lower()


def normalize_none(s):
    return s


def normalize_char(char):
    return char.lower().replace(' ', '_')


def space_tokenize(s):
    return s.split(' ')


def char_tokenize(s):
    return list(s)


def mecab_tokenize(s):
    import MeCab
    wakati = MeCab.Tagger("-Owakati")
    return wakati.parse(s).split()


def write_vocab(
        working_dir,
        sentences,
        name="words",
        min_freq=0,
        default_tags=['<pad>', '<sos>', '<eos>', '<oov>'],
        normalize_fn=normalize_string,
        tokenize_fn=space_tokenize):
    os.makedirs(os.path.join(working_dir, "vocab"), exist_ok=True)
    word_freqs = {}
    for sent in sentences:
        s = normalize_fn(sent.replace('_', ' '))
        # ls = char_tokenize(s) if token == 'char' else space_tokenize(s)
        ls = tokenize_fn(s)
        for word in ls:
            if word.strip() == '':
                continue
            if word in word_freqs:
                word_freqs[word] += 1
            else:
                word_freqs[word] = 1

    words = list([word for word in word_freqs if word_freqs[word] > min_freq])
    words.sort(key=lambda word: word_freqs[word], reverse=True)
    with open(os.path.join(working_dir, "vocab", name + ".txt"), "w", encoding='utf-8') as fo:
        fo.write('\n'.join(default_tags) + '\n')
        fo.write("\n".join(words))


def get_token_id(vocab, word):
    """
    :type vocab: Vocab
    :type word: str
    :rtype: int
    """
    if word in vocab:
        return vocab[word]
    else:
        if '<oov>' in vocab:
            return vocab['<oov>']
        elif '<unk>' in vocab:
            return vocab['<unk>']
        else:
            raise Exception("No out-of-vocabulary token found.")


class Vocab:
    def __init__(self, file_name):
        """
        :type file_name: str
        """
        self._token2index = load_tkn_to_idx(file_name)
        self._index2token = load_idx_to_tkn(file_name)

    def __getitem__(self, token):
        """
        :type token: str
        :rtype: int
        """
        return self._token2index[token] if token in self._token2index else None

    def get_token_id(self, token):
        return self[token] or self.oov_token_idx

    def add_token(self, token):
        """
        :type token: str
        """
        self._token2index[token] = len(self._token2index)
        self._index2token.append(token)

    def __len__(self):
        return len(self._token2index)

    def get_token(self, idx):
        """
        :type idx: int
        :rtype: str
        """
        return self._index2token[idx]

    def decode_idx_list(self, ls):
        """
        :type ls: list[int]
        :rtype: list[str]
        """
        return [self.get_token(idx) for idx in ls]

    @property
    def sos_token_idx(self):
        idx = self['<sos>'] or self['<s>']
        assert idx is not None
        return idx

    @property
    def eos_token_idx(self):
        idx = self['<eos>'] or self['</s>']
        assert idx is not None
        return idx

    @property
    def oov_token_idx(self):
        idx = self['<oov>'] or self['<unk>']
        assert idx is not None
        return idx


class NLPDataset(BaseDataset):
    def __init__(self, mode, params):
        super().__init__(mode, params)
