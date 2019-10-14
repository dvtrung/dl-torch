import torch.nn as nn

import torch
from dlex.configs import MainConfig
from dlex.torch.models.base import BaseModel
from dlex.torch import Batch


class SequenceClassifier(BaseModel):
    def __init__(self, params: MainConfig, dataset):
        super().__init__(params, dataset)
        embedding_dim = self.params.model.embedding_dim or self.params.dataset.embedding_dim
        if params.dataset.pretrained_embeddings is not None:
            self.embed = nn.Embedding(dataset.vocab_size, embedding_dim)
            self.embed.weight = nn.Parameter(dataset.embedding_weights, requires_grad=False)
        else:
            self.embed = nn.Embedding(dataset.vocab_size, embedding_dim)

        cfg = params.model
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.rnn_type in ['lstm', 'gru']:
            self.rnn = getattr(nn, cfg.rnn_type.upper())(
                embedding_dim, cfg.hidden_size,
                cfg.num_layers, dropout=cfg.dropout, batch_first=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(
                cfg.embedding_dim, cfg.hidden_size,
                cfg.num_layers, nonlinearity=nonlinearity, dropout=cfg.dropout)
        self.linear = nn.Linear(cfg.hidden_size, dataset.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch: Batch):
        embed = self.embed(batch.X)
        if batch.X_len is not None:
            embed = nn.utils.rnn.pack_padded_sequence(embed, batch.X_len, batch_first=True)
        _, hidden = self.rnn(embed)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        output = self.linear(hidden[-1])
        return output

    def get_loss(self, batch, output):
        return self.criterion(output, batch.Y)

    def infer(self, batch: Batch):
        output = self(batch)
        _, labels = torch.max(output, dim=-1)
        return labels.cpu().detach().numpy().tolist(), output, None

