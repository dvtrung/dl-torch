import os

from torchtext import data, datasets

from dlex.configs import MainConfig, ModuleConfigs
from dlex.datasets.nlp.builder import NLPDataset
from dlex.datasets.torch import Dataset
from dlex.torch import Batch
from dlex.torch import BatchItem


class IMDB(NLPDataset):
    def __init__(self, params: MainConfig):
        super().__init__(params)

    def maybe_preprocess(self, force=False):
        TEXT = data.Field(
            sequential=True,
            tokenize=lambda x: x.split(),
            lower=True,
            include_lengths=True,
            batch_first=True)
        LABEL = data.LabelField()
        self.train_data, self.test_data = datasets.IMDB.splits(TEXT, LABEL, root=self.get_raw_data_dir())
        TEXT.build_vocab(self.train_data, vectors=self.get_embedding_vectors())
        LABEL.build_vocab(self.train_data)

        # self.data = {}
        # self.data['train'], self.data["valid"] = train_data.split()
        # train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        #    (train_data, valid_data, test_data),
        #    batch_size=32, sort_key=lambda x: len(x.text),
        #    repeat=False, shuffle=True)

        self.TEXT, self.LABEL = TEXT, LABEL

    def get_pytorch_wrapper(self, mode: str):
        return PytorchIMDB(self, mode)


class PytorchIMDB(Dataset):
    def __init__(self, builder, mode):
        super().__init__(builder, mode)

    def get_iter(self, batch_size, start=0, end=-1):
        iter = data.BucketIterator(
            self.builder.train_data if self.mode == "train" else self.builder.test_data,
            batch_size=batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False, shuffle=True)
        print(len(iter), len(self.builder.train_data[0]))
        return map(lambda item: Batch(
            X=item.text[0].cuda(),
            X_len=item.text[1].cuda(),
            Y=item.label.cuda()
        ), iter)

    def __len__(self):
        return 25000

    @property
    def vocab_size(self):
        return len(self.builder.TEXT.vocab)

    @property
    def embedding_dim(self):
        return self.params.dataset.embedding_dim

    @property
    def embedding_weights(self):
        return self.builder.TEXT.vocab.vectors

    @property
    def num_classes(self):
        return len(self.builder.LABEL.vocab)

    def format_output(self, y_pred, batch_item: BatchItem) -> (str, str, str):
        if self.configs.output_format == "text":
            return ' '.join([self.builder.TEXT.vocab.itos(word_id) for word_id in batch_item.X]), \
                   ' '.join([self.builder.TEXT.vocab.itos(word_id) for word_id in batch_item.Y]), \
                   ' '.join([self.builder.TEXT.vocab.itos(word_id) for word_id in y_pred])
        else:
            return super().format_output(y_pred, batch_item)