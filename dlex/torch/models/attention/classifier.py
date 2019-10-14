from dataclasses import dataclass
from typing import List

import math
import numpy as np
import torch.nn as nn

import torch
from dlex.datasets.seq2seq.torch import PytorchSeq2SeqDataset
from dlex.torch import Batch
from dlex.torch.models.attention.attention import BahdanauAttention, NoAttention
from dlex.torch.models.base import ClassificationModel
from dlex.utils.logging import logger
from .decoder import DecoderRNN
from .encoder import EncoderRNN


@dataclass
class EncoderConfigDict:
    num_layers: int
    hidden_size: int
    output_size: int
    rnn_type: str = "lstm"
    bidirectional: bool = False
    input_size: int = None
    update_embedding: bool = True


@dataclass
class DecoderConfigDict:
    hidden_size: int
    output_size: int = None


@dataclass
class AttentionConfigDict:
    type: str
    size: int = None
    # location-based attention
    num_channels: int = None
    filter_size: int = None


@dataclass
class AttentionModelConfigDict:
    name: str
    encoder: EncoderConfigDict
    decoder: DecoderConfigDict
    attention: AttentionConfigDict
    dropout: float

    def __post_init__(self):
        self.encoder = EncoderConfigDict(**self.encoder)
        self.decoder = DecoderConfigDict(**self.decoder)
        self.attention = AttentionConfigDict(**self.attention)


class SequenceClassifier(ClassificationModel):
    def __init__(self, params, dataset: PytorchSeq2SeqDataset):
        super().__init__(params, dataset)
        self.configs = AttentionModelConfigDict(**self.params.model)

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(self.configs.encoder.num_layers + 1, dtype=np.int)
        logger.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # encoder
        self._encoder = self._build_encoder()
        # attention
        self._attention = self._build_attention()
        # decoder
        self._decoder = self._build_decoder()

        # weight initialization
        self.init_like_chainer()

    def _build_encoder(self) -> EncoderRNN:
        cfg = self.configs
        return EncoderRNN(
            input_size=self.dataset.input_size,
            rnn_type=cfg.encoder.rnn_type,
            bidirectional=cfg.encoder.bidirectional,
            num_layers=cfg.encoder.num_layers,
            hidden_size=cfg.encoder.hidden_size,
            output_size=cfg.encoder.output_size,
            dropout=cfg.dropout)

    def _build_decoder(self) -> DecoderRNN:
        cfg = self.configs
        return DecoderRNN(
            input_size=0,
            rnn_type="rnn",
            num_layers=1,
            hidden_size=cfg.decoder.hidden_size,
            output_size=cfg.decoder.output_size,
            vocab_size=self.dataset.num_classes,
            attention=self._attention,
            sos_idx=0,  # any label since input_size = 0
            eos_idx=None,
            max_length=None,
            beam_search_configs=None,
            dropout=cfg.dropout)

    def _build_attention(self) -> List[torch.nn.Module]:
        cfg = self.configs
        logger.info("Attention type: %s", cfg.attention.type)
        if cfg.attention.type is None:
            attention = [NoAttention()]
        elif cfg.attention.type == 'bahdanau':
            self._test = BahdanauAttention(
                encoder_output_size=cfg.encoder.output_size,
                decoder_hidden_size=cfg.decoder.hidden_size,
                attention_dim=cfg.attention.size,
            )
            attention = torch.nn.ModuleList([self._test])
        else:
            raise Exception(f"Attention type is not defined: ${cfg.attention.type}")
        return attention

    def init_like_chainer(self):
        """Initialize weight like chainer

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """

        def lecun_normal_init_parameters(module):
            for p in module.parameters():
                data = p.data
                if data.dim() > 1 and data.size(1) == 0:
                    continue
                if data.dim() == 1:
                    # bias
                    data.zero_()
                elif data.dim() == 2:
                    # linear weight
                    n = data.size(1)
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                elif data.dim() == 4:
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                else:
                    raise NotImplementedError

        def set_forget_bias_to_one(bias):
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.)

        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        self._decoder.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in range(len(self._decoder._decoder)):
            set_forget_bias_to_one(self._decoder._decoder[l].bias_ih)

    def forward(self, batch: Batch):
        states = self._encoder(batch.X, batch.X_len)
        y = batch.Y.view((-1, 1))
        states.decoder_inputs = torch.cat([torch.full(y.shape, 0, dtype=y.dtype, device=y.device), y], dim=-1)
        states = self._decoder(states, use_teacher_forcing=True)
        return states.decoder_outputs[:, 0, :]

    def infer(self, batch: Batch):
        states = self._encoder(batch.X, batch.X_len)
        y = batch.Y.view((-1, 1))
        states.decoder_inputs = torch.cat([torch.full(y.shape, 0, dtype=y.dtype, device=y.device), y], dim=-1)
        states = self._decoder(states, use_teacher_forcing=True)
        _, sequences = torch.max(states.decoder_outputs[:, 0, :], dim=-1)
        return sequences, None, states

    def calculate_all_attentions(self, batch):
        with torch.no_grad():
            encoder_outputs, encoder_output_lens, _ = self._encoder(batch.X, batch.X_len)
            return self._decoder.calculate_all_attentions(encoder_outputs, encoder_output_lens, batch.Y)

    def subsample_frames(self, x):
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_device(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))
        h.contiguous()
        return h, ilen

    def write_summary(self, summary_writer, batch, output):
        y_pred, others = output
        str_input, str_ground_truth, str_predicted = self.dataset.format_output(y_pred[0], batch.item(i))
        summary_writer.add_text(
            "inference_result",
            str_predicted,
            self.current_epoch)
        # img = [attentions[0] for attentions in others.attentions]
        # logger.debug(len(others.attentions))

    def train_log(self, batch: Batch, output, verbose):
        d = super().train_log(batch, output, verbose)
        if verbose:
            d["input_length"] = batch.X.shape[1]
            d["output_length"] = batch.Y.shape[1]
        return d

    def infer_log(self, batch: Batch, output, verbose):
        d = super().infer_log(batch, output, verbose)
        if verbose:
            d["input_length"] = batch.X.shape[1]
            d["output_length"] = max([len(seq) for seq in output])
        return d


class LabelSequenceClassifier(SequenceClassifier):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)

    def _build_encoder(self) -> EncoderRNN:
        """
        :rtype: EncoderRNN
        """
        cfg = self.params.model

        self.embedding = nn.Embedding(
            num_embeddings=self.dataset.vocab_size,
            embedding_dim=cfg.encoder.input_size)
        if cfg.encoder.embedding is not None:
            # TODO: implement custom embedding
            pass
        self.embedding.requires_grad = cfg.encoder.update_embedding

        return EncoderRNN(
            input_size=cfg.encoder.input_size,
            rnn_type=cfg.encoder.rnn_type,
            num_layers=cfg.encoder.num_layers,
            hidden_size=cfg.encoder.hidden_size,
            output_size=cfg.encoder.output_size,
            bidirectional=cfg.encoder.bidirectional,
            dropout=cfg.dropout
        )

    def forward(self, batch: Batch):
        return super().forward(Batch(
            X=self.embedding(batch.X.to(self.embedding.weight.device)),
            X_len=batch.X_len,
            Y=batch.Y.to(self.embedding.weight.device),
            Y_len=batch.Y_len
        ))

    def infer(self, batch: Batch):
        return super().infer(Batch(
            X=self.embedding(batch.X.to(self.embedding.weight.device)),
            X_len=batch.X_len,
            Y=batch.Y,
            Y_len=batch.Y_len
        ))