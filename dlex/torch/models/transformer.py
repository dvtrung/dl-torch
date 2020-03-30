import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dlex.configs import Params
from dlex.datasets.seq2seq.torch import PytorchSeq2SeqDataset
from dlex.torch import Batch
from dlex.torch.models.base import BaseModel
from dlex.torch.utils.variable_length_tensor import get_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Embedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, positional_encoding: bool = True):
        super().__init__()
        self.emb = nn.Embedding(input_dim, hidden_dim)
        if positional_encoding:
            self.pos_enc = PositionalEncoding(hidden_dim)
        else:
            self.pos_enc = nn.Sequential()

    def forward(self, x):
        return self.pos_enc(self.emb(x))


class Transformer(BaseModel):
    def __init__(self, params: Params, dataset: PytorchSeq2SeqDataset):
        super().__init__(params, dataset)
        self.src_mask = None
        cfg = params.model
        self.transformer = nn.Transformer(
            d_model=cfg.dim_model,
            nhead=cfg.num_heads,
            num_encoder_layers=cfg.encoder.num_layers or cfg.num_layers,
            num_decoder_layers=cfg.decoder.num_layers or cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout
        )

        self.decoder_emb = Embedding(dataset.output_size, cfg.dim_model)
        self.decoder_projection = nn.Linear(cfg.decoder.hidden_size, dataset.output_size)

    def forward(self, batch):
        batch_y = self.decoder_emb(batch.Y[:, :-1].contiguous())
        batch_y_len = [l - 1 for l in batch.Y_len]

        output = self.transformer(
            batch.X.transpose(0, 1), batch_y.transpose(0, 1),
            tgt_mask=self.transformer.generate_square_subsequent_mask(batch_y.shape[1]).cuda(self.decoder_projection.weight.device),
            src_key_padding_mask=~get_mask(batch.X_len, max_len=batch.X.shape[1]),
            tgt_key_padding_mask=~get_mask(batch_y_len, max_len=batch_y.shape[1])
        ).transpose(0, 1)

        output = self.decoder_projection(output)
        return output

    def infer(self, batch):
        batch_y = self.decoder_emb(batch.Y[:, :-1].contiguous())
        # batch_y.rename('N', 'T', 'E')
        batch_y_len = [l - 1 for l in batch.Y_len]
        # batch_X.rename('N', 'S', 'E')

        output = self.transformer(
            batch.X.transpose(0, 1), batch_y.transpose(0, 1),
            tgt_mask=self.transformer.generate_square_subsequent_mask(batch_y.shape[1]).cuda(self.decoder_projection.weight.device),
            src_key_padding_mask=~get_mask(batch.X_len, max_len=batch.X.shape[1]),
            tgt_key_padding_mask=~get_mask(batch_y_len, max_len=batch_y.shape[1]),
        ).transpose(0, 1)
        output = self.decoder_projection(output)
        output = F.softmax(output, dim=-1)
        output = output.max(-1)[1]
        y_ref = [y[1:y_len - 1].tolist() for y, y_len in zip(batch.Y, batch.Y_len)]
        return [pred[:len_pred - 2].tolist() for pred, len_pred in zip(output, batch.Y_len)], y_ref, None, None

    def get_loss(self, batch, output):
        y = batch.Y[:, 1:].contiguous()
        loss = F.cross_entropy(
            output.view(-1, self.dataset.output_size),
            y.view(-1),
            ignore_index=self.dataset.pad_token_idx)
        return loss


class NMT(Transformer):
    def __init__(self, params: Params, dataset: PytorchSeq2SeqDataset):
        super().__init__(params, dataset)
        cfg = params.model
        self.embedding = Embedding(self.dataset.input_size, cfg.dim_model)

    def forward(self, batch):
        return super().forward(Batch(
            X=self.embedding(batch.X),
            X_len=batch.X_len,
            Y=batch.Y,
            Y_len=batch.Y_len
        ))

    def infer(self, batch: Batch):
        return super().infer(Batch(
            X=self.embedding(batch.X),
            X_len=batch.X_len,
            Y=batch.Y,
            Y_len=batch.Y_len
        ))