from typing import Tuple, List

import torch.nn as nn

from .decoder import DecodingStates


class EncoderRNN(nn.Module):
    def __init__(
            self,
            input_size: int,
            rnn_type: str,
            bidirectional: bool,
            num_layers: int,
            subsample: List[int],
            hidden_size: int,
            output_size: int,
            dropout: float):
        super(EncoderRNN, self).__init__()
        self._hidden_size = hidden_size

        if subsample is None:
            self._rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size // 2 if bidirectional else hidden_size,
                batch_first=True,
                bidirectional=bidirectional,
                num_layers=num_layers,
                dropout=dropout)
        else:
            self._rnns = nn.ModuleList([nn.LSTM(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size // 2 if bidirectional else hidden_size,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout) for i in range(num_layers)])

        if output_size != hidden_size:
            self._linear = nn.Linear(hidden_size, output_size)
        else:
            self._linear = nn.Sequential()  # do nothing

        self._subsample = subsample

    def forward(self, inputs, input_lengths):
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True, enforce_sorted=False)

        if self._subsample is None:
            inputs, hidden = self._rnn(inputs)
        else:
            for rnn, subsample in zip(self._rnns, self._subsample):
                inputs, hidden = rnn(inputs)
                if subsample > 1:
                    inputs, input_lengths = nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True)
                    inputs = nn.utils.rnn.pack_padded_sequence(
                        inputs[:, ::subsample, :],
                        [l // subsample for l in input_lengths],
                        batch_first=True,
                        enforce_sorted=False)

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True)
        output = self._linear(output)
        return DecodingStates(
            encoder_outputs=output,
            encoder_output_lens=output_lengths,
            encoder_states=hidden)
