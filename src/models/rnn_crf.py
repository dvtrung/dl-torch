import torch
from torch import nn

from models.base import BaseModel
from utils.ops_utils import Tensor, LongTensor, maybe_cuda

CUDA = torch.cuda.is_available()
#CUDA = False


class Model(BaseModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self.params = params
        self.rnn = RNN(params, dataset.vocab_char_size, dataset.vocab_word_size, dataset.num_tags)
        self.crf = crf(params, dataset.num_tags, dataset.tag_to_idx["<sos>"], dataset.tag_to_idx["<eos>"], dataset.tag_to_idx["<pad>"])
        self = self.cuda() if CUDA else self

    def forward(self, cx, wx, y): # for training
        mask = wx.data.gt(0).float()
        h = self.rnn(cx, wx, mask)
        Z = self.crf.forward(h, mask)
        score = self.crf.score(h, y, mask)
        return Z - score # NLL loss

    def decode(self, cx, wx): # for prediction
        mask = wx.data.gt(0).float()
        h = self.rnn(cx, wx, mask)
        return LongTensor(self.crf.decode(h, mask)).view(mask.shape[0], -1)


class embed(nn.Module):
    def __init__(self, params, char_vocab_size, word_vocab_size, embed_size):
        super().__init__()
        self.params = params
        num_embeds = self.params.embed_unit.count("+") + 1 # number of embeddings (1, 2)
        dim = embed_size // num_embeds # dimension of each embedding vector

        # architecture
        if self.params.embed_unit[:4] == "char":
            self.char_embed = self.cnn(char_vocab_size, dim)
        if self.params.embed_unit[-4:] == "word":
            self.word_embed = nn.Embedding(word_vocab_size, dim, padding_idx = self.params.pad_idx)

    class cnn(nn.Module):
        def __init__(self, dim_in, dim_out):
            super().__init__()
            self.embed_size = 50
            self.num_featmaps = 30 # feature maps generated by each kernel
            self.kernel_sizes = [3]

            # architecture
            self.embed = nn.Embedding(dim_in, self.embed_size, padding_idx = self.params.pad_idx)
            self.conv = nn.ModuleList([nn.Conv2d(
                in_channels = 1, # Ci
                out_channels = self.num_featmaps, # Co
                kernel_size = (i, self.embed_size) # (height, width)
            ) for i in self.kernel_sizes]) # num_kernels (K)
            self.dropout = nn.Dropout(self.params.dropout)
            self.fc = nn.Linear(len(self.kernel_sizes) * self.num_featmaps, dim_out)

        def forward(self, x):
            x = x.view(-1, x.size(2)) # [batch_size (B) * word_seq_len (L), char_seq_len (H)]
            x = self.embed(x) # [B * L, H, embed_size (W)]
            x = x.unsqueeze(1) # [B * L, Ci, H, W]
            h = [conv(x) for conv in self.conv] # [B * L, Co, H, 1] * K
            h = [F.relu(k).squeeze(3) for k in h] # [B * L, Co, H] * K
            h = [F.max_pool1d(k, k.size(2)).squeeze(2) for k in h] # [B * L, Co] * K
            h = torch.cat(h, 1) # [B * L, Co * K]
            h = self.dropout(h)
            h = self.fc(h) # [B * L, dim_out]
            h = h.view(self.params.batch_size, -1, h.size(1)) # [B, L, dim_out]
            return h

    def forward(self, cx, wx):
        ch = self.char_embed(cx) if self.params.embed_unit[:4] == "char" else Tensor()
        wh = self.word_embed(wx) if self.params.embed_unit[-4:] == "word" else Tensor()
        h = torch.cat([ch, wh], 2)
        return h


class RNN(nn.Module):
    def __init__(self, params, char_vocab_size, word_vocab_size, num_tags):
        super().__init__()

        self.params = params
        # architecture
        self.embed = embed(params, char_vocab_size, word_vocab_size, params.embed_size)
        self.num_dirs = 2 if params.rnn_type == 'bilstm' else 1
        if params.rnn_type in ['lstm', 'bilstm']:
            self.rnn = nn.LSTM(
                input_size=params.embed_size,
                hidden_size=params.hidden_size // self.num_dirs,
                num_layers=params.num_layers,
                bias=True,
                batch_first=True,
                dropout=params.dropout,
                bidirectional=(params.rnn_type == 'bilstm')
            )
        self.out = nn.Linear(params.hidden_size, num_tags) # RNN output to tag

    def init_hidden(self, batch_size): # initialize hidden states
        params = self.params
        h = zeros(params.num_layers * self.num_dirs, batch_size, params.hidden_size // self.num_dirs) # hidden state
        if params.rnn_type in ["lstm", "bilstm"]:
            c = zeros(params.num_layers * self.num_dirs, batch_size, params.hidden_size // self.num_dirs) # cell state
            return (h, c)
        return h

    def forward(self, cx, wx, mask):
        self.hidden = self.init_hidden(wx.shape[0])
        x = self.embed(cx, wx)
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int(), batch_first=True)
        h, _ = self.rnn(x, self.hidden)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        h = self.out(h)
        h *= mask.unsqueeze(2)
        return h


class crf(nn.Module):
    def __init__(self, params, num_tags, sos_idx, eos_idx, pad_idx):
        super().__init__()
        self.num_tags = num_tags
        self.params = params
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        # matrix of transition scores from j to i
        self.trans = nn.Parameter(randn(num_tags, num_tags))
        self.trans.data[self.sos_idx, :] = -10000. # no transition to SOS
        self.trans.data[:, self.eos_idx] = -10000. # no transition from EOS except to PAD
        self.trans.data[:, self.pad_idx] = -10000. # no transition from PAD except to PAD
        self.trans.data[self.pad_idx, :] = -10000. # no transition to PAD except from EOS
        self.trans.data[self.pad_idx, self.eos_idx] = 0.
        self.trans.data[self.pad_idx, self.pad_idx] = 0.


    def forward(self, h, mask): # forward algorithm
        # initialize forward variables in log space
        batch_size = h.shape[0]
        score = maybe_cuda(torch.full((batch_size, self.num_tags), -10000.)) # [B, C]
        score[:, self.params.sos_idx] = 0.
        trans = self.trans.unsqueeze(0) # [1, C, C]
        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            emit_t = h[:, t].unsqueeze(2) # [B, C, 1]
            score_t = score.unsqueeze(1) + emit_t + trans # [B, 1, C] -> [B, C, C]
            score_t = log_sum_exp(score_t) # [B, C, C] -> [B, C]
            score = score_t * mask_t + score * (1 - mask_t)
        score = log_sum_exp(score + self.trans[self.params.eos_idx])
        return score # partition function

    def score(self, h, y, mask): # calculate the score of a given sequence
        batch_size = h.shape[0]
        score = maybe_cuda(torch.zeros(batch_size))
        h = h.unsqueeze(3)
        trans = self.trans.unsqueeze(2)
        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t]
            emit_t = torch.cat([h[t, y[t + 1]] for h, y in zip(h, y)])
            trans_t = torch.cat([trans[y[t + 1], y[t]] for y in y])
            score += (emit_t + trans_t) * mask_t
        last_tag = y.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.trans[self.params.eos_idx, last_tag]
        return score

    def decode(self, h, mask): # Viterbi decoding
        # initialize backpointers and viterbi variables in log space
        batch_size = h.shape[0]
        bptr = LongTensor()
        score = maybe_cuda(torch.full((batch_size, self.num_tags), -10000.))
        score[:, self.sos_idx] = 0.

        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            score_t = score.unsqueeze(1) + self.trans # [B, 1, C] -> [B, C, C]
            score_t, bptr_t = score_t.max(2) # best previous scores and tags
            score_t += h[:, t] # plus emission scores
            bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = score_t * mask_t + score * (1 - mask_t)
        score += self.trans[self.eos_idx]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(batch_size):
            x = best_tag[b] # best tag
            y = int(mask[b].sum().item())
            for bptr_t in reversed(bptr[b][:y]):
                x = bptr_t[x]
                best_path[b].append(x)
            best_path[b].pop()
            best_path[b].reverse()

        return best_path


def randn(*args):
    x = torch.randn(*args)
    return x.cuda() if CUDA else x


def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if CUDA else x


def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))
