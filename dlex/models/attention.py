import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dlex.models.base import BaseModel, default_params
from dlex.utils.model_utils import rnn_cell
from dlex.utils.ops_utils import maybe_cuda


class EncoderRNN(nn.Module):
    def __init__(self, params, input_size):
        super(EncoderRNN, self).__init__()
        cfg = params.model.encoder
        self.hidden_size = cfg.hidden_size

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.rnn_type == "bilstm",
            dropout=params.model.dropout_p
        )

    def forward(self, input_var, input_lengths):
        input_var = nn.utils.rnn.pack_padded_sequence(input_var, input_lengths, batch_first=True)
        output, hidden = self.rnn(input_var)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


class AttentionLayer(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.
    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}
    Args:
        dim(int): The number of expected features in the output
    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.
    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
    Examples::
         >>> attention = seq2seq.models.AttentionLayer(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)
    """
    def __init__(self, params):
        super(AttentionLayer, self).__init__()
        self.hidden_size = params.model.decoder.hidden_size
        self.linear_out = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, encoder_outputs):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = encoder_outputs.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, encoder_outputs.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        context = torch.bmm(attn, encoder_outputs)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((context, output), dim=2)
        # output -> (batch, out_len, dim)
        output = torch.tanh(self.linear_out(combined.view(-1, hidden_size + self.hidden_size)))
        output = output.view(batch_size, -1, hidden_size)

        return output, attn


class BasicDecoder(nn.Module):
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id
        super().__init__()

    def forward(self, encoder_outputs, decoder_inputs, decoder_outputs):
        batch_size = encoder_outputs.size(0)
        max_length = decoder_outputs.size(1)
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)
        for step in range(decoder_outputs.size(1)):
            symbols = decoder_outputs[:, step, :].topk(1)[1]
            sequence_symbols.append(symbols)
            eos_batches = decoder_inputs[:, step].eq(self.eos_token_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = step
        decoder_outputs = decoder_outputs.permute(1, 0, 2)
        #print("basic")
        #for i in range(batch_size):
        #    for j in range(max_length):
        #        print(sequence_symbols[j][i].item(), end=' ')
        #    print()
        return sequence_symbols, lengths, decoder_outputs


class GreedyDecoder(nn.Module):
    def __init__(self, sos_token_id, eos_token_id):
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        super().__init__()

    def forward(self, encoder_outputs, forward_step_fn, initial_decoder_hidden, max_length):
        batch_size = encoder_outputs.size(0)
        decoder_input = maybe_cuda(torch.full((batch_size, 1), self.sos_token_id, dtype=torch.int64))
        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)
        attentions = []
        for step in range(max_length):
            decoder_output, step_attention = forward_step_fn(
                input_var=decoder_input,
                encoder_hidden=initial_decoder_hidden,
                encoder_outputs=encoder_outputs)
            step_output = decoder_output.squeeze(1)
            decoder_outputs.append(step_output)
            if step_attention is not None:
                attentions.append(step_attention)
            symbols = step_output.topk(1)[1]
            decoder_input = symbols
            eos_batches = symbols.data.eq(self.eos_token_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            sequence_symbols.append(symbols)

        #print("greedy")
        #for i in range(batch_size):
        #    for j in range(max_length):
        #        print(sequence_symbols[j][i].item(), end=' ')
        #    print()
        return sequence_symbols, lengths, decoder_outputs, attentions


class DecoderRNN(nn.Module):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, params, output_size, sos_token_id, eos_token_id):
        super(DecoderRNN, self).__init__()
        self.params = params
        cfg = params.model.decoder
        self.hidden_size = cfg.hidden_size

        self.bidirectional_encoder = cfg.bidirectional
        self.rnn = rnn_cell(cfg.rnn_type)(
            self.hidden_size,
            self.hidden_size,
            cfg.num_layers,
            batch_first=True,
            dropout=params.model.dropout_p
        )

        self.output_size = output_size
        self.use_attention = cfg.use_attention
        self.eos_token_id = eos_token_id
        self.sos_token_id = sos_token_id

        self.init_input = None

        self.embedding = nn.Embedding(output_size, self.hidden_size)
        if cfg.use_attention:
            self.attention = AttentionLayer(params)

        self.out = nn.Linear(self.hidden_size, output_size)
        self.basic_decoder = BasicDecoder(eos_token_id)
        self.greedy_decoder = GreedyDecoder(sos_token_id, eos_token_id)

    def forward_step(self, input_var, encoder_hidden, encoder_outputs):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        # embedded = self.input_dropout(embedded)

        output, _ = self.rnn(embedded, encoder_hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = F.log_softmax(
            self.out(output.contiguous().view(-1, self.hidden_size)),
            dim=1
        ).view(batch_size, output_size, -1)
        return predicted_softmax, attn

    def forward(self, decoder_inputs=None, encoder_hidden=None, encoder_outputs=None, use_teacher_forcing=True):
        ret_dict = dict()
        # assert not (use_teacher_forcing and decoder_inputs is None)  # no teacher forcing during inference

        decoder_inputs, batch_size, max_length = self._validate_args(
            decoder_inputs, encoder_hidden, encoder_outputs)
        initial_decoder_hidden = self._init_state(encoder_hidden)

        if use_teacher_forcing:
            decoder_outputs, attentions = self.forward_step(
                decoder_inputs[:, :-1], initial_decoder_hidden, encoder_outputs)
            sequence_symbols, lengths, decoder_outputs = self.basic_decoder(
                encoder_outputs=encoder_outputs,
                decoder_inputs=decoder_inputs[:, :-1],
                decoder_outputs=decoder_outputs,
            )
        else:
            sequence_symbols, lengths, decoder_outputs, attentions = self.greedy_decoder(
                encoder_outputs=encoder_outputs,
                forward_step_fn=self.forward_step,
                initial_decoder_hidden=initial_decoder_hidden,
                max_length=max_length
            )

        ret_dict[DecoderRNN.KEY_LENGTH] = lengths
        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_ATTN_SCORE] = attentions

        return decoder_outputs, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None or not self.params.model.feed_encoder_state:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if isinstance(self.rnn, nn.LSTM):
                    batch_size = encoder_hidden[0].size(0)
                elif isinstance(self.rnn, nn.GRU):
                    batch_size = encoder_hidden.size(0)

        # set default input and max decoding length
        if inputs is None:
            inputs = torch.LongTensor([self.sos_token_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.params.model.decoder.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length


@default_params(dict(
    teacher_forcing_ratio=0.2,
    feed_encoder_state=False
))
class Attention(BaseModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)

        self.encoder = EncoderRNN(
            params,
            dataset.input_size)
        self.decoder = DecoderRNN(
            params,
            dataset.output_size,
            dataset.sos_token_id, dataset.eos_token_id)
        self.criterion = nn.NLLLoss()

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, batch):
        encoder_outputs, encoder_hidden = self.encoder(batch['X'], batch['X_len'])
        return self.decoder(
            decoder_inputs=batch['Y'],
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            use_teacher_forcing=random.random() < self.params.model.teacher_forcing_ratio)

    def get_loss(self, batch, output):
        y = batch['Y']
        batch_size = y.size(0)
        decoder_outputs, other = output
        loss = 0
        for step, step_output in enumerate(decoder_outputs):
            target = y[:, step + 1]
            loss += self.criterion(step_output.view(batch_size, -1), target)

        return loss / len(decoder_outputs)

    def infer(self, batch):
        encoder_outputs, encoder_hidden = self.encoder(batch['X'], batch['X_len'])
        _, other = self.decoder(
            decoder_inputs=None,
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            use_teacher_forcing=False)
        ret = []
        ret_len = []
        batch_size = len(batch['X'])
        for i in range(batch_size):
            seq = []
            for pos in range(other[DecoderRNN.KEY_LENGTH][i]):
                seq.append(other[DecoderRNN.KEY_SEQUENCE][pos][i].item())
            ret_len.append(len(seq))
            ret.append(seq)
            # print(seq, len(seq))
        return ret, ret_len


class NMT(Attention):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)

        cfg = params.model

        self.embedding = nn.Embedding(
            num_embeddings=dataset.input_size,
            embedding_dim=cfg.encoder.hidden_size)
        if cfg.embedding is not None:
            # TODO: implement custom embedding
            pass
        self.embedding.requires_grad = cfg.update_embedding

        self.encoder = EncoderRNN(
            params,
            cfg.encoder.hidden_size)
        self.decoder = DecoderRNN(
            params,
            dataset.output_size,
            dataset.sos_token_id, dataset.eos_token_id)

    def forward(self, batch):
        return super().forward(dict(
            X=self.embedding(batch['X']),
            X_len=batch['X_len'],
            Y=batch['Y'],
            Y_len=batch['Y_len']
        ))

    def infer(self, batch):
        return super().infer(dict(
            X=self.embedding(batch['X']),
            X_len=batch['X_len'],
            Y=batch['Y'],
            Y_len=batch['Y_len']
        ))