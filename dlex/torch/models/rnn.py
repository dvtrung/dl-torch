import torch.nn as nn

import torch
from dlex.configs import Params
from dlex.torch.models.base import BaseModel
from dlex.torch import Batch


class SequenceClassifier(BaseModel):
    def __init__(self, params: Params, dataset):
        super().__init__(params, dataset)

        embedding_dim = self.params.model.embedding_dim or self.params.dataset.embedding_dim
        if embedding_dim:
            if params.dataset.pretrained_embeddings is not None:
                self.embed = nn.Embedding(dataset.vocab_size, embedding_dim)
                self.embed.weight = nn.Parameter(dataset.embedding_weights, requires_grad=False)
            else:
                self.embed = nn.Embedding(dataset.vocab_size, embedding_dim)
        else:
            embedding_dim = self.configs.input_size or dataset.input_size
            self.embed = None

        cfg = self.configs
        self.drop = nn.Dropout(cfg.dropout)
        if cfg.rnn_type in ['lstm', 'gru']:
            self.rnn = getattr(nn, cfg.rnn_type.upper())(
                embedding_dim, cfg.hidden_size,
                cfg.num_layers, dropout=cfg.dropout, batch_first=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[cfg.rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(
                cfg.embedding_dim, cfg.hidden_size,
                cfg.num_layers, nonlinearity=nonlinearity, dropout=cfg.dropout)
        self.linear = nn.Linear(cfg.hidden_size, dataset.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch: Batch):
        X = batch.X['h0_essential'].data
        lengths = batch.X['h0_essential'].lengths
        embed = self.embed(X) if self.embed else X
        if lengths is not None:
            embed = nn.utils.rnn.pack_padded_sequence(embed, lengths, batch_first=True)
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
        return labels.cpu().detach().numpy().tolist(), batch.Y.detach().numpy().tolist(), output, None

