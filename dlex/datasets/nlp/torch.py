import os
from typing import List

import torch
import torchtext

from dlex.configs import ModuleConfigs
from dlex.datasets.nlp.utils import Tokenizer
from dlex.datasets.torch import Dataset


class NLPDataset(Dataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)

    @property
    def vocab_size(self):
        raise NotImplementedError()

    def load_embeddings(self, specials=None):
        emb = self.configs.embeddings
        if emb.pretrained is None:
            assert emb.dim is not None
            return torch.nn.Embedding(self.vocab_size, emb.dim), None
        if emb.pretrained.lower() in ["glove", "fasttext"]:
            if emb.pretrained.lower() == 'glove':
                from torchtext.vocab import GloVe
                vocab = GloVe(
                    name=emb.name, dim=emb.dim,
                    cache=os.path.join(ModuleConfigs.TMP_PATH, "torchtext"))
            elif emb.pretrained.lower() == 'fasttext':
                from torchtext.vocab import FastText
                return FastText()

            vectors = vocab.vectors
            itos = vocab.itos
            if specials is not None:
                itos += specials
                vectors = torch.cat([vectors, torch.rand(len(specials), len(vectors[0]))])
            return torch.nn.Embedding.from_pretrained(vectors, freeze=emb.freeze or True), itos
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