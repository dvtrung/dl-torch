import torch
import torch.nn as nn

from dlex.torch import Batch
from dlex.torch.models.base import BaseModel
from .encoder import Encoder
from .decoder import Decoder


class Transformer(BaseModel):
    def __init__(
            self, params, dataset,
            n_src_vocab, n_tgt_vocab):

        super().__init__(params, dataset)

        cfg = params.model

        self.encoder = Encoder(
            len_max_seq=cfg.encoder.max_length or cfg.max_length,
            input_size=cfg.encoder.input_size, dim_model=cfg.dim_model, dim_inner=cfg.dim_inner,
            num_layers=cfg.num_layers, num_heads=cfg.num_heads,
            dim_key=cfg.dim_key, dim_value=cfg.dim_value,
            dropout=cfg.dropout)

        self.decoder = Decoder(
            vocab_size=self.dataset.output_size,
            len_max_seq=cfg.decoder.max_length or cfg.max_length,
            output_size=cfg.decoder.output_size, dim_model=cfg.dim_model, dim_inner=cfg.dim_inner,
            num_layers=cfg.num_layers, num_heads=cfg.num_heads,
            dim_key=cfg.dim_key, dim_value=cfg.dim_value,
            dropout=cfg.dropout, share_embeddings=cfg.decoder.share_embeddings,
            pad_idx=dataset.pad_token_idx)

        # To facilitate the residual connections, the dimensions of all module outputs shall be the same.
        assert cfg.dim_model == cfg.encoder.input_size

    def forward(self, batch, src_seq, src_pos, tgt_seq, tgt_pos):

        # tgt_seq = batch.Y[:, :-1]

        # TODO:
        X_pos = None
        Y_pos = None
        encoder_outputs, *_ = self.encoder(batch.X, X_pos)
        decoder_outputs, *_ = self.decoder(batch.Y, Y_pos, batch.X, encoder_outputs)

        return decoder_outputs.view(-1, decoder_outputs.size(2))


class NMT(Transformer):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)

    def _build_encoder(self) -> Encoder:
        """
        :rtype: EncoderRNN
        """
        cfg = self.params.model

        self.embedding = nn.Embedding(
            num_embeddings=self.dataset.input_size,
            embedding_dim=cfg.encoder.input_size)
        if cfg.encoder.embedding is not None:
            # TODO: implement custom embedding
            pass
        self.embedding.requires_grad = cfg.encoder.update_embedding

        return Encoder(
            input_size=cfg.encoder.input_size,
            rnn_type=cfg.encoder.rnn_type,
            num_layers=cfg.encoder.num_layers,
            subsample=cfg.encoder.subsample,
            hidden_size=cfg.encoder.hidden_size,
            output_size=cfg.encoder.output_size,
            bidirectional=cfg.encoder.bidirectional,
            dropout=cfg.dropout)

    def forward(self, batch: Batch):
        return super().forward(Batch(
            X=self.embedding(batch.X.to(self.embedding.weight.device)),
            X_len=batch.X_len,
            Y=batch.Y.to(self.embedding.weight.device),
            Y_len=batch.Y_len
        ))

    def infer(self, batch: Batch):
        return super().infer(Batch(
            X=self.embedding(batch.X),
            X_len=batch.X_len,
            Y=batch.Y,
            Y_len=batch.Y_len
        ))