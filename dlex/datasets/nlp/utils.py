"""NLP Dataset"""
import os
import re
import unicodedata
from typing import List

import nltk
import numpy as np
import spacy as spacy
from torchtext import data

from dlex.configs import ModuleConfigs
from dlex.utils.logging import logger

# nltk.download('punkt')


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
from dlex.utils import maybe_download, maybe_unzip


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


def normalize_lower(sentence: str):
    return sentence.strip().lower()


def normalize_lower_alphanumeric(sentence: str):
    s = sentence.strip().lower()
    s = re.sub("[^a-z0-9\uAC00-\uD7A3]+", " ", s)
    return s


def normalize_string_ascii(sentence):
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


def normalize_string(sentence):
    """
    :param str sentence:
    :return: normalized sentence, separated by space
    :rtype str
    """
    # x = re.sub("[^ a-zA-Z0-9\uAC00-\uD7A3]+", " ", x)
    # x = re.sub("[\u3040-\u30FF]+", "\u3042", x) # convert Hiragana and Katakana to あ
    # x = re.sub("[\u4E00-\u9FFF]+", "\u6F22", x) # convert CJK unified ideographs to 漢
    sentence = re.sub(r"([\.!?,\";\(\)])\'", r" \1", sentence)
    # sent = re.sub(r"[^a-zA-Z.!?,]+", r" ", sent)
    sentence = re.sub(r"\s+", " ", sentence)
    sentence = re.sub("^ | $", "", sentence)

    words = sentence.split(' ')
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


def nltk_tokenize(s):
    return nltk.word_tokenize(s)


class Tokenizer:
    def __init__(self, normalize_fn=None, tokenize_fn=None):
        self.normalize_fn = normalize_fn
        self.tokenize_fn = tokenize_fn

    def process(self, s):
        s = self.normalize_fn(s)
        s = self.tokenize_fn(s)
        return s


spacy_nlp = None


def spacy_tokenize(s):
    import spacy
    from spacy.symbols import ORTH
    global spacy_nlp
    if spacy_nlp is None:
        # sputnik.install('spacy', spacy.about.__version__, 'en_default', data_path=ModuleConfigs.TMP_PATH)
        spacy_nlp = spacy.load('en_core_web_sm', via=ModuleConfigs.TMP_PATH)
        spacy_nlp.tokenizer.add_special_case('<eos>', [{ORTH: '<eos>'}])
        spacy_nlp.tokenizer.add_special_case('<bos>', [{ORTH: '<bos>'}])
        spacy_nlp.tokenizer.add_special_case('<unk>', [{ORTH: '<unk>'}])
    return [_s.text for _s in spacy_nlp.tokenizer(s)]


def normalize_char(char):
    return char.lower().replace(' ', '_')


def space_tokenize(s):
    return s.split(' ')


def char_tokenize(s: str):
    s = s.replace(" ", "_")
    return list(s)


def mecab_tokenize(s):
    import MeCab
    wakati = MeCab.Tagger("-Owakati")
    return wakati.parse(s).split()


def write_vocab(
        working_dir,
        sentences,
        output_file_name: str,
        tokenizer: Tokenizer = None,
        min_freq=0,
        specials=None):
    if tokenizer is None:
        tokenizer = Tokenizer(normalize_none, space_tokenize)
    if specials is None:
        specials = ['<pad>', '<sos>', '<eos>', '<oov>']
    os.makedirs(os.path.join(working_dir, "vocab"), exist_ok=True)
    word_freqs = {}
    for sent in sentences:
        # if normalize_fn is not None:
        #     s = normalize_fn(sent.replace('_', ' '))
        # else:
        #     s = sent
        # ls = char_tokenize(s) if token == 'char' else space_tokenize(s)
        ls = tokenizer.process(sent)
        for word in ls:
            if word.strip() == '':
                continue
            if word in word_freqs:
                word_freqs[word] += 1
            else:
                word_freqs[word] = 1

    words = list([word for word in word_freqs if word_freqs[word] > min_freq])
    words.sort(key=lambda word: word_freqs[word], reverse=True)
    file_path = os.path.join(working_dir, "vocab", output_file_name)
    with open(file_path, "w", encoding='utf-8') as fo:
        fo.write('\n'.join(specials) + '\n')
        fo.write("\n".join(words))

    logger.info("Vocab written to %s (%d tokens)", file_path, len(specials) + len(words))


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
    def __init__(self, token_list: List[str] = None):
        if token_list is None:
            self._token2index = {}
            self._index2token = []
        else:
            self._index2token = token_list
            self._token2index = {token: idx for idx, token in enumerate(token_list)}
        self.embeddings = None
        self.embedding_dim = None

    @classmethod
    def from_file(cls, file_name):
        index2token = []
        fo = open(file_name, encoding='utf-8')
        for line in fo:
            line = line.strip()
            if line == "":
                continue
            index2token.append(line)
        fo.close()
        return cls(index2token)

    def __getitem__(self, token: str) -> int:
        return self._token2index[token] if token in self._token2index else self.oov_token_idx

    def tolist(self) -> List[str]:
        return self._index2token

    def get_token_id(self, token):
        return self[token] or self.oov_token_idx

    def add_token(self, token: str):
        if token not in self._token2index:
            self._token2index[token] = len(self._token2index)
            self._index2token.append(token)

    def __len__(self):
        return len(self._token2index)

    def get_token(self, idx: int) -> str:
        return self._index2token[idx]

    def decode_idx_list(self, ls: List[int]) -> List[str]:
        return [self.get_token(idx) for idx in ls]

    def encode_token_list(self, ls: List[str]) -> List[int]:
        return [self.get_token_id(token) for token in ls]

    @property
    def sos_token_idx(self) -> int:
        idx = self['<sos>'] or self['<s>']
        assert idx is not None
        return idx

    @property
    def eos_token_idx(self) -> int:
        idx = self['<eos>'] or self['</s>']
        assert idx is not None
        return idx

    @property
    def blank_token_idx(self):
        idx = self['<blank>'] or self['<pad>']
        assert idx is not None
        return idx

    @property
    def oov_token_idx(self) -> int:
        if '<oov>' in self._token2index:
            return self._token2index['<oov>']
        elif '<unk>' in self._token2index:
            return self._token2index['<unk>']
        raise Exception("<oov> token not found.")