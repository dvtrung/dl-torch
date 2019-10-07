import torch
import torch.nn as nn

from .utils import get_sinusoid_encoding_table, get_attn_key_pad_mask, get_non_pad_mask
from .sublayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_inner, num_heads, dim_key, dim_value, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            num_heads,
            dim_model,
            dim_key, dim_value,
            dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(dim_model, dim_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.self_attention(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    def __init__(
            self,
            len_max_seq, input_size,
            num_layers, num_heads, dim_key, dim_value,
            dim_model, dim_inner, dropout=0.1):

        super().__init__()

        num_positions = len_max_seq + 1

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(num_positions, dim_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(dim_model, dim_inner, num_heads, dim_key, dim_value, dropout=dropout)
            for _ in range(num_layers)])

        if input_size != dim_model:
            self.emb = nn.Linear(input_size, dim_model)
        else:
            self.emb = None

    def forward(self, seq, seq_pos, return_attns=False):
        if self.emb:
            seq = self.emb(seq)
        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_pos, seq)
        non_pad_mask = (seq_pos > 0).type(torch.float).unsqueeze(-1)

        # -- Forward
        enc_output = seq + self.position_enc(seq_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output