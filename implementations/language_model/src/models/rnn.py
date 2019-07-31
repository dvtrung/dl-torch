import torch.nn as nn
import torch

from dlex.configs import AttrDict
from dlex.torch.models.base import BaseModel
from dlex.torch import Batch


class RNNLM(BaseModel):
    def __init__(self, params: AttrDict, dataset):
        super().__init__(params, dataset)
        cfg = params.model
        self.drop = nn.Dropout(cfg.dropout)

        embedding_dim = self.params.model.embedding_dim or self.params.dataset.embedding_dim
        if params.dataset.pretrained_embeddings is not None:
            self.embed = nn.Embedding(dataset.vocab_size, embedding_dim)
            self.embed.weight = nn.Parameter(dataset.embedding_weights, requires_grad=False)
        else:
            self.embed = nn.Embedding(dataset.vocab_size, embedding_dim)

        assert cfg.rnn_type in ['lstm', 'gru']
        self.rnn = getattr(nn, cfg.rnn_type.upper())(
            embedding_dim, cfg.hidden_size,
            cfg.num_layers, dropout=cfg.dropout, batch_first=True)
        self.decoder = nn.Linear(cfg.hidden_size, dataset.vocab_size)

        if cfg.tie_weights:
            if cfg.hidden_size != cfg.embedding_dim:
                raise ValueError('When using the tied flag, hidden_size must be equal to embedding_size')
            self.decoder.weight = self.embed.weight

        self.init_weights()
        self.rnn_type = cfg.rnn_type
        self.criterion = nn.CrossEntropyLoss()

        assert isinstance(params.train.batch_size, int)  # dynamic batch size not allowed
        self._hidden = self.init_hidden(params.train.batch_size)

    def init_weights(self):
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def repackage_hidden(self, h, device):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if isinstance(h, tuple):
            return tuple(self.repackage_hidden(v, device) for v in h)
        else:
            return torch.tensor(h.data, device=h.device).to(device)

    def forward(self, batch: Batch):
        self._hidden = self.repackage_hidden(self._hidden, batch.X.device)
        emb = self.drop(self.embed(batch.X))
        # output, self._hidden = self.rnn(emb, self._hidden)
        output, self._hidden = self.rnn(emb)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded

    def infer(self, batch: Batch):
        output = self(batch)
        _, sequence = torch.max(output, dim=-1)
        return list(sequence.cpu().detach().numpy()), output, None

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        cfg = self.params.model
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(cfg.num_layers, batch_size, cfg.hidden_size),
                    weight.new_zeros(cfg.num_layers, batch_size, cfg.hidden_size))
        else:
            return weight.new_zeros(cfg.num_layers, batch_size, cfg.hidden_size)

    def get_loss(self, batch, output):
        return self.criterion(output.view(output.shape[0] * output.shape[1], -1), batch.Y.view(-1))
