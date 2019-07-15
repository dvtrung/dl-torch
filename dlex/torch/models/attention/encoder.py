import torch.nn as nn

from .decoder import DecodingStates


class EncoderRNN(nn.Module):
    def __init__(
            self,
            input_size: int,
            rnn_type: str,
            bidirectional: bool,
            num_layers: int,
            hidden_size: int,
            output_size: int,
            dropout: float):
        super(EncoderRNN, self).__init__()
        self._hidden_size = hidden_size

        self._rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2 if bidirectional else hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout)

        if output_size != hidden_size:
            self._linear = nn.Linear(hidden_size, output_size)
        else:
            self._linear = nn.Sequential()  # do nothing

    def forward(self, inputs, input_lengths):
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self._rnn(inputs)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self._linear(output)
        return DecodingStates(
            encoder_outputs=output,
            encoder_output_lens=input_lengths,
            encoder_states=hidden)
