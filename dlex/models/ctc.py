import torch.nn as nn
import torch.nn.functional as F

from dlex.models.base import BaseModel


class CTC(BaseModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        # dataset.insert_blank_token()
        cfg = params.model
        self.rnn = nn.LSTM(
            input_size=dataset.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout_p
        )
        self.fc = nn.Linear(cfg.hidden_size * 2 if cfg.bidirectional else cfg.hidden_size, dataset.output_size)
        self.ctc_loss = nn.CTCLoss(blank=dataset.blank_token_id)

    def forward(self, batch):
        inp = nn.utils.rnn.pack_padded_sequence(batch['X'], batch['X_len'], batch_first=True)
        output, hidden = self.rnn(inp)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        logits = F.log_softmax(self.fc(output), -1)
        return logits

    def get_loss(self, batch, output):
        batch_size = batch['X'].shape[0]
        return self.ctc_loss(
            log_probs=output.permute(1, 0, 2),
            targets=batch['Y'],
            input_lengths=batch['X_len'],
            target_lengths=batch['Y_len']) / batch_size

    def infer(self, batch):
        output = self.forward(batch)
        output = output.argmax(-1)
        sequences = []
        lengths = []
        for i, output_item in enumerate(output):
            seq = []
            for t in range(batch['X_len'][i]):
                token = output_item[t].item()
                if token != self.dataset.blank_token_id:
                    seq.append(token)
            sequences.append(seq)
            lengths.append(len(seq))
        return sequences, lengths
