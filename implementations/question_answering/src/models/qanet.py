from typing import List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dlex.configs import AttrDict
from dlex.torch import Batch
from dlex.torch.models.base import BaseModel
from dlex.torch.utils.ops_utils import maybe_cuda, LongTensor
from ..datasets.squad import QABatch


def mask_logits(inputs, mask):
    mask = mask.type(torch.float32)
    return inputs + (-1e30) * (1 - mask)


def length_to_mask(lengths: List, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    if isinstance(lengths, list):
        lengths = LongTensor(lengths)
    assert len(lengths.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or lengths.max().item()
    mask = torch.arange(
        max_len, device=lengths.device,
        dtype=lengths.dtype).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=lengths.device)
    return mask


class InitializedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, relu=False, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.out = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups,
            bias=bias)
        self.relu = relu
        if relu:
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu:
            return F.relu(self.out(x))
        else:
            return self.out(x)


def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + maybe_cuda(signal)).transpose(1, 2)


def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                        padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


class Highway(nn.Module):
    def __init__(self, layer_num: int, size, dropout):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([InitializedConv1d(size, size, relu=False, bias=True) for _ in range(self.n)])
        self.gate = nn.ModuleList([InitializedConv1d(size, size, bias=True) for _ in range(self.n)])
        self.dropout = dropout

    def forward(self, x):
        # x: shape [batch_size, hidden_size, length]
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=self.dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
            # x = F.relu(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, connector_dim, num_heads, dropout):
        super().__init__()
        self.mem_conv = InitializedConv1d(connector_dim, connector_dim * 2, kernel_size=1, relu=False, bias=False)
        self.query_conv = InitializedConv1d(connector_dim, connector_dim, kernel_size=1, relu=False, bias=False)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.connector_dim = connector_dim
        self.num_heads = num_heads
        self.bias = nn.Parameter(bias)
        self.dropout = dropout

    def forward(self, queries, mask):
        memory = queries

        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        Q = self.split_last_dim(query, self.num_heads)
        K, V = [self.split_last_dim(tensor, self.num_heads) for tensor in
                torch.split(memory, self.connector_dim, dim=2)]

        key_depth_per_head = self.connector_dim // self.num_heads
        Q *= key_depth_per_head ** -0.5
        x = self.dot_product_attention(Q, K, V, mask=mask)
        return self.combine_last_two_dim(x.permute(0, 2, 1, 3)).transpose(1, 2)

    def dot_product_attention(self, q, k, v, bias=False, mask=None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q, k.permute(0, 1, 3, 2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret


class Embedding(nn.Module):
    def __init__(self, connector_dim, word_dim, char_dim, dropout, dropout_char):
        super().__init__()
        self.conv2d = nn.Conv2d(char_dim, connector_dim, kernel_size=(1, 5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = InitializedConv1d(word_dim + connector_dim, connector_dim, bias=False)
        self.high = Highway(2, connector_dim, dropout)
        self.dropout_word = nn.Dropout(dropout)
        self.dropout_char = nn.Dropout(dropout_char)

    def forward(self, ch_emb, wd_emb, length):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = self.dropout_char(ch_emb)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)
        ch_emb = ch_emb.squeeze()

        wd_emb = self.dropout_word(wd_emb)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.conv1d(emb)
        emb = self.high(emb)
        return emb


class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int, connector_dim: int, num_heads: int, dropout):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(connector_dim, num_heads, dropout)
        self.FFN_1 = InitializedConv1d(ch_num, ch_num, relu=True, bias=True)
        self.FFN_2 = InitializedConv1d(ch_num, ch_num, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(connector_dim) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(connector_dim)
        self.norm_2 = nn.LayerNorm(connector_dim)
        self.conv_num = conv_num
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num + 1) * blks
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1, 2)).transpose(1, 2)
            if i % 2 == 0:
                out = self.dropout(out)
            out = conv(out)
            out = self.layer_dropout(out, res, self.dropout_p * float(l) / total_layers)
            l += 1
        res = out
        out = self.norm_1(out.transpose(1, 2)).transpose(1, 2)
        out = self.dropout(out)
        out = self.self_att(out, mask)
        out = self.layer_dropout(out, res, self.dropout_p * float(l) / total_layers)
        l += 1
        res = out

        out = self.norm_2(out.transpose(1, 2)).transpose(1, 2)
        out = self.dropout(out)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, self.dropout_p * float(l) / total_layers)
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return self.dropout(inputs) + residual
        else:
            return inputs + residual


class ContextQueryAttention(nn.Module):
    def __init__(self, connector_dim, dropout):
        super().__init__()
        w4C = torch.empty(connector_dim, 1)
        w4Q = torch.empty(connector_dim, 1)
        w4mlu = torch.empty(1, 1, connector_dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2)  # B * context_max_length * emb_dim
        Q = Q.transpose(1, 2)
        batch_size = C.shape[0]
        S = self.trilinear(C, Q)
        Cmask = Cmask.view(batch_size, C.shape[1], 1)
        Qmask = Qmask.view(batch_size, 1, Q.shape[1])
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    def trilinear(self, C, Q):
        C = self.dropout(C)
        Q = self.dropout(Q)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Q.shape[1]])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, C.shape[1], -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res


class Pointer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = InitializedConv1d(dim * 2, 1)
        self.w2 = InitializedConv1d(dim * 2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)
        return Y1, Y2


class QANet(BaseModel):
    def __init__(self, params: AttrDict, dataset):
        super().__init__(params, dataset)
        cfg = params.model
        self.word_emb = nn.Embedding(dataset.vocab_size_word, cfg.word_dim)
        self.char_emb = nn.Embedding(dataset.vocab_size_char, cfg.char_dim)
        self.emb = Embedding(cfg.connector_dim, cfg.word_dim, cfg.char_dim, cfg.dropout, cfg.dropout_char)
        self.emb_enc = EncoderBlock(
            conv_num=4,
            ch_num=cfg.connector_dim, k=7,
            connector_dim=cfg.connector_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout)
        self.cq_att = ContextQueryAttention(cfg.connector_dim, cfg.dropout)
        self.cq_resizer = InitializedConv1d(cfg.connector_dim * 4, cfg.connector_dim)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(
            conv_num=2, ch_num=cfg.connector_dim, k=5, connector_dim=cfg.connector_dim, num_heads=cfg.num_heads,
            dropout=cfg.dropout) for _ in range(7)])
        self.out = Pointer(cfg.connector_dim)

    def forward(self, batch: QABatch):
        cfg = self.params.model
        maskC = length_to_mask(batch.X_len.context_word).float()
        maskQ = length_to_mask(batch.X_len.question_word).float()
        Cw = self.word_emb(batch.X.context_word)
        Cc = self.char_emb(batch.X.context_char)
        Qw = self.word_emb(batch.X.question_word)
        Qc = self.char_emb(batch.X.question_char)
        C = self.emb(Cc, Cw, Cw.shape[1])
        Q = self.emb(Qc, Qw, Qw.shape[1])
        Ce = self.emb_enc(C, maskC, 1, 1)
        Qe = self.emb_enc(Q, maskQ, 1, 1)
        X = self.cq_att(Ce, Qe, maskC, maskQ)
        M0 = self.cq_resizer(X)
        M0 = F.dropout(M0, p=cfg.dropout, training=self.training)
        for i, block in enumerate(self.encoder_blocks):
            M0 = block(M0, maskC, i * (2 + 2) + 1, 7)
        M1 = M0
        for i, block in enumerate(self.encoder_blocks):
            M0 = block(M0, maskC, i * (2 + 2) + 1, 7)
        M2 = M0
        M0 = F.dropout(M0, p=cfg.dropout, training=self.training)
        for i, block in enumerate(self.encoder_blocks):
            M0 = block(M0, maskC, i * (2 + 2) + 1, 7)
        M3 = M0
        p1, p2 = self.out(M1, M2, M3, maskC)
        return p1, p2

    def get_loss(self, batch: QABatch, output):
        p1, p2 = output
        y1, y2 = batch.Y[:, 0], batch.Y[:, 1]
        return F.cross_entropy(p1, y1) + F.cross_entropy(p2, y2)

    def infer(self, batch: Batch):
        p1, p2 = self.forward(batch)
        p1 = torch.argmax(p1, -1).tolist()
        p2 = torch.argmax(p2, -1).tolist()
        y1, y2 = batch.Y[:, 0].tolist(), batch.Y[:, 1].tolist()
        pred = list(map(list, zip(*[p1, p2])))
        ref = list(map(list, zip(*[y1, y2])))
        return pred, ref, None, None
