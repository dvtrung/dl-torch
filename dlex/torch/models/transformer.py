import torch
import torch.nn as nn
import torch.nn.functional as F
from dlex.configs import MainConfig
from dlex.datasets.seq2seq.torch import PytorchSeq2SeqDataset
from dlex.torch import Batch
from dlex.torch.models.base import BaseModel
from dlex.torch.utils.ops_utils import maybe_cuda


def generate_key_padding_mask(seq_len, max_len):
    ret = [[not i < l for i in range(max_len)] for l in seq_len]
    return maybe_cuda(torch.BoolTensor(ret))


class Transformer(BaseModel):
    def __init__(self, params: MainConfig, dataset: PytorchSeq2SeqDataset):
        super().__init__(params, dataset)
        self.src_mask = None
        cfg = params.model
        self.transformer = nn.Transformer(
            d_model=cfg.dim_model,
            nhead=cfg.num_heads,
            num_encoder_layers=cfg.encoder.num_layers or cfg.num_layers,
            num_decoder_layers=cfg.decoder.num_layers or cfg.num_layers,
            dim_feedforward=cfg.dim_model,
            dropout=cfg.dropout
        )

        if dataset.input_size != cfg.encoder.hidden_size:
            self.encoder_linear = nn.Linear(cfg.encoder.input_size, cfg.encoder.hidden_size)
        else:
            self.encoder_linear = None

        self.decoder_emb = nn.Embedding(dataset.output_size, cfg.decoder.hidden_size)
        self.decoder_projection = nn.Linear(cfg.decoder.hidden_size, dataset.output_size)

    def forward(self, batch: Batch):
        batch_y = self.decoder_emb(batch.Y[:, :-1].contiguous())
        batch_y_len = [l - 1 for l in batch.Y_len]

        if self.encoder_linear:
            batch_X = self.encoder_linear(batch.X)

        output = self.transformer(
            batch_X.transpose(0, 1), batch_y.transpose(0, 1),
            src_key_padding_mask=generate_key_padding_mask(batch.X_len, batch.X.shape[1]),
            tgt_key_padding_mask=generate_key_padding_mask(batch_y_len, batch_y.shape[1])
        ).transpose(0, 1)
        output = self.decoder_projection(output)
        return output

    def infer(self, batch: Batch):
        batch_y = self.decoder_emb(batch.Y[:, :-1].contiguous())
        batch_y_len = [l - 1 for l in batch.Y_len]

        if self.encoder_linear:
            batch_X = self.encoder_linear(batch.X)

        output = self.transformer(
            batch_X.transpose(0, 1), batch_y.transpose(0, 1),
            src_key_padding_mask=generate_key_padding_mask(batch.X_len, batch.X.shape[1]),
            tgt_key_padding_mask=generate_key_padding_mask(batch_y_len, batch_y.shape[1])
        ).transpose(0, 1)
        output = self.decoder_projection(output)
        output = output.max(1)[1]
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
    def __init__(self, params: MainConfig, dataset: PytorchSeq2SeqDataset):
        super().__init__(params, dataset)
        cfg = params.model
        self.embedding = nn.Embedding(
            num_embeddings=self.dataset.input_size,
            embedding_dim=cfg.encoder.input_size)

    def forward(self, batch: Batch):
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