from sklearn.metrics import accuracy_score
from torchtext import data, datasets

from dlex.configs import MainConfig
from dlex.datasets.nlp.builder import NLPDataset
from dlex.datasets.nlp.utils import nltk_tokenize, spacy_tokenize
from dlex.datasets.torch import Dataset
from dlex.torch import Batch
from dlex.torch import BatchItem


class WikiText2(NLPDataset):
    def __init__(self, params: MainConfig):
        super().__init__(params)

    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)

    def maybe_preprocess(self, force=False):
        TEXT = data.Field(lower=True, tokenize=spacy_tokenize)
        self.train_data, self.valid_data, self.test_data = datasets.WikiText2.splits(TEXT, root=self.get_raw_data_dir())
        TEXT.build_vocab(self.train_data, vectors=self.get_embedding_vectors())
        self.TEXT = TEXT

    def evaluate(self, pred, ref, metric: str, output_path: str):
        if metric == "acc":
            return accuracy_score(pred, ref) * len(pred), len(pred)
        else:
            return super().evaluate(pred, ref, metric, output_path)

    def get_pytorch_wrapper(self, mode: str):
        return PytorchWikiText2(self, mode)

    def get_tensorflow_wrapper(self, mode: str):
        raise Exception("No tensorflow interface.")

    def decode(self, tokens):
        return [self.TEXT.vocab.itos[idx] for idx in tokens]


class PytorchWikiText2(Dataset):
    def __init__(self, builder, mode):
        super().__init__(builder, mode)

    @property
    def data(self):
        return self.builder.train_data if self.mode == "train" else self.builder.test_data

    def get_iter(self, batch_size, start=0, end=-1):
        iter = data.BPTTIterator(
            self.data,
            batch_size=batch_size,
            bptt_len=self.params.dataset.bptt_len)
        return map(lambda item: Batch(
            X=item.text.transpose(1, 0).cuda(),
            Y=item.target.transpose(1, 0).cuda()
        ), iter)

    def __len__(self):
        return 10000

    @property
    def vocab_size(self):
        return len(self.builder.TEXT.vocab)

    @property
    def embedding_dim(self):
        return self.params.dataset.embedding_dim

    @property
    def embedding_weights(self):
        return self.builder.TEXT.vocab.vectors

    def format_output(self, y_pred, batch_item: BatchItem) -> (str, str, str):
        if self.configs.output_format == "text":
            return ' '.join(self.builder.decode(batch_item.X)), \
                   ' '.join(self.builder.decode(batch_item.Y)), \
                   ' '.join(self.builder.decode(y_pred))
        else:
            return super().format_output(y_pred, batch_item)