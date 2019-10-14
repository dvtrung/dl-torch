import torch
import torch.nn as nn
import torch.nn.functional as F

from dlex.torch import Batch
from dlex.torch.models.base import BaseModel
from dlex.torch.utils.ops_utils import LongTensor
from .encoder import Encoder
from .decoder import Decoder


class Transformer(BaseModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        cfg = self.params.model
        return Encoder(
            len_max_seq=cfg.encoder.max_length or cfg.max_length,
            input_size=cfg.encoder.input_size, dim_model=cfg.dim_model, dim_inner=cfg.dim_inner,
            num_layers=cfg.encoder.num_layers or cfg.num_layers,
            num_heads=cfg.num_heads,
            dim_key=cfg.key_size, dim_value=cfg.value_size,
            dropout=cfg.dropout)

    def _build_decoder(self):
        cfg = self.params.model
        return Decoder(
            vocab_size=self.dataset.output_size,
            len_max_seq=cfg.decoder.max_length or cfg.max_length,
            output_size=cfg.decoder.output_size, dim_model=cfg.dim_model, dim_inner=cfg.dim_inner,
            num_layers=cfg.encoder.num_layers or cfg.num_layers,
            num_heads=cfg.num_heads,
            dim_key=cfg.key_size, dim_value=cfg.value_size,
            dropout=cfg.dropout, share_embeddings=cfg.decoder.share_embeddings,
            pad_idx=self.dataset.pad_token_idx)

    def forward(self, batch: Batch):
        # src_seq, src_pos, tgt_seq, tgt_pos
        # tgt_seq = batch.Y[:, :-1]
        batch_y = batch.Y[:, :-1].contiguous()
        X_pos = LongTensor([[i + 1 if i < x_len else 0 for i in range(len(x))] for x, x_len in zip(batch.X, batch.X_len)])
        Y_pos = LongTensor([[i + 1 if i < y_len - 1 else 0 for i in range(len(y))] for y, y_len in zip(batch_y, batch.Y_len)])
        encoder_outputs = self.encoder(batch.X, X_pos)
        decoder_outputs = self.decoder(batch_y, Y_pos, X_pos, encoder_outputs)

        return decoder_outputs

    def get_loss(self, batch, output):
        y = batch.Y[:, 1:].contiguous()
        loss = F.cross_entropy(
            output.view(-1, self.dataset.output_size),
            y.view(-1),
            ignore_index=self.dataset.pad_token_idx)
        return loss

    def infer(self, batch: Batch):
        X_pos = LongTensor(
            [[i + 1 if i < x_len else 0 for i in range(len(x))] for x, x_len in zip(batch.X, batch.X_len)])
        Y_pos = LongTensor(
            [[i + 1 if i < y_len else 0 for i in range(len(y))] for y, y_len in zip(batch.Y, batch.Y_len)])
        encoder_outputs = self.encoder(batch.X, X_pos)
        output = self.decoder(batch.Y, Y_pos, X_pos, encoder_outputs)
        output = output.max(1)[1]
        y_ref = [y[1:y_len - 1].tolist() for y, y_len in zip(batch.Y, batch.Y_len)]
        return [pred[:len_pred - 2].tolist() for pred, len_pred in zip(output, batch.Y_len)], y_ref, None, None


class NMT(Transformer):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)

    def _build_encoder(self) -> Encoder:
        cfg = self.params.model

        self.embedding = nn.Embedding(
            num_embeddings=self.dataset.input_size,
            embedding_dim=cfg.encoder.input_size)
        if cfg.encoder.embedding is not None:
            # TODO: implement custom embedding
            pass
        self.embedding.requires_grad = cfg.encoder.update_embedding

        return Encoder(
            len_max_seq=cfg.encoder.max_length or cfg.max_length,
            input_size=cfg.encoder.input_size, dim_model=cfg.dim_model, dim_inner=cfg.dim_inner,
            num_layers=cfg.encoder.num_layers or cfg.num_layers,
            num_heads=cfg.num_heads,
            dim_key=cfg.key_size, dim_value=cfg.value_size,
            dropout=cfg.dropout)

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