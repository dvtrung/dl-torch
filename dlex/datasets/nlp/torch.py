import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torchtext

from dlex.configs import ModuleConfigs
from dlex.datasets.nlp.utils import Tokenizer, Vocab
from dlex.datasets.torch import Dataset
from dlex.utils import logger


class NLPDataset(Dataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)

    @property
    def vocab_size(self):
        raise NotImplementedError()

    def load_embeddings(self, tokens: List[str] = None, specials: List[str] = None) -> Tuple[nn.Embedding, Vocab]:
        """
        Load pretrained embedding defined in dataset.embeddings
        :param tokens: if specified, only load embeddings of these tokens
        :param specials: special tokens
        :return:
        """
        emb = self.configs.embeddings
        if not emb.pretrained:
            assert emb.dim is not None
            return nn.Embedding(self.vocab_size, emb.dim), None
        elif emb.pretrained.lower() in ["glove", "fasttext"]:
            if emb.pretrained.lower() == 'glove':
                from torchtext.vocab import GloVe
                vocab = GloVe(
                    name=emb.name, dim=emb.dim,
                    cache=os.path.join(ModuleConfigs.get_tmp_path(), "torchtext"))
            elif emb.pretrained.lower() == 'fasttext':
                from torchtext.vocab import FastText
                return FastText()

            vectors = vocab.vectors
            index2token = vocab.itos
            token2index = None
            if tokens:  # limit vocabulary to list of tokens
                num_oovs = 0
                keep = []
                index2token = []
                token2index = {}
                for t in tokens:
                    _t = t.lower()
                    if _t in token2index:
                        if t not in token2index:
                            token2index[t] = token2index[_t]
                    elif _t in vocab.stoi:
                        keep.append(vocab.stoi[_t.lower()])
                        token2index[_t] = len(index2token)
                        token2index[t] = len(index2token)
                        index2token.append(_t)
                    else:
                        num_oovs += 1
                vectors = vectors[keep]
                if num_oovs:
                    logger.warning(f"{num_oovs} tokens not found in pre-trained embeddings")

            logger.debug(f"Load embeddings: {emb.pretrained} (no. embeddings: {len(index2token):,})")

            if specials is not None:
                for s in specials:
                    token2index[s] = len(index2token)
                    index2token.append(s)
                index2token += specials
                vectors = torch.cat([vectors, torch.rand(len(specials), len(vectors[0]))])

            return nn.Embedding.from_pretrained(vectors, freeze=emb.freeze or True), Vocab(index2token, token2index)
        else:
            raise ValueError("%s is not supported." % emb.name)

    def tokenize(
            self,
            sentences: List[str],
            delimiter=' ',
            min_freq=0,
            specials=None,
            vectors=None):
        if specials is None:
            specials = ['<pad>', '<sos>', '<eos>', '<oov>']
        field = torchtext.data.Field(sequential=True, tokenize=lambda s: s.split(delimiter))
        field.build_vocab(sentences, specials=specials, min_freq=min_freq, vectors=vectors)
        return field