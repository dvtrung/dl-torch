import torch
import torch.nn as nn

from .utils import get_sinusoid_encoding_table, get_non_pad_mask, get_subsequent_mask, \
    get_attn_key_pad_mask
from .sublayers import MultiHeadAttention, PositionwiseFeedForward


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, dim_model, dim_inner, num_heads, dim_key, dim_value, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(num_heads, dim_model, dim_key, dim_value, dropout=dropout)
        self.encoder_attention = MultiHeadAttention(num_heads, dim_model, dim_key, dim_value, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(dim_model, dim_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.self_attention(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.encoder_attention(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(
            self,
            vocab_size, len_max_seq, output_size,
            num_layers, num_heads, dim_key, dim_value,
            dim_model, dim_inner, dropout, share_embeddings,
            pad_idx):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            vocab_size, output_size, padding_idx=pad_idx)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, output_size, padding_idx=pad_idx),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(dim_model, dim_inner, num_heads, dim_key, dim_value, dropout=dropout)
            for _ in range(num_layers)])

        self.tgt_word_prj = nn.Linear(dim_model, self.dataset.output_size, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        if share_embeddings:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.tgt_word_emb.weight
            self.x_logit_scale = (dim_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        dec_output = self.tgt_word_prj(dec_output) * self.x_logit_scale

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output