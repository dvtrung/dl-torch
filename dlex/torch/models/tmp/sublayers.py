import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, num_heads, dim_model, dim_key, dim_value, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.dim_key = dim_key
        self.dim_value = dim_value

        self.w_qs = nn.Linear(dim_model, num_heads * dim_key)
        self.w_ks = nn.Linear(dim_model, num_heads * dim_key)
        self.w_vs = nn.Linear(dim_model, num_heads * dim_value)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (dim_model + dim_key)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (dim_model + dim_key)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (dim_model + dim_value)))

        self.attention = ScaledDotProductAttention(temperature=np.power(dim_key, 0.5))
        self.layer_norm = nn.LayerNorm(dim_model)

        self.fc = nn.Linear(num_heads * dim_value, dim_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None):
        sz_b, len_q, _ = queries.size()
        sz_b, len_k, _ = keys.size()
        sz_b, len_v, _ = values.size()

        residual = queries

        q = self.w_qs(queries).view(sz_b, len_q, self.num_heads, self.dim_key)
        k = self.w_ks(keys).view(sz_b, len_k, self.num_heads, self.dim_key)
        v = self.w_vs(values).view(sz_b, len_v, self.num_heads, self.dim_value)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.dim_key) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.dim_key) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.dim_value) # (n*b) x lv x dv

        mask = mask.repeat(self.num_heads, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(self.num_heads, sz_b, len_q, self.dim_value)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output