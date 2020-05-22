"""Model utils"""
from typing import List, Union
import importlib

import torch
import torch.nn as nn


def get_model(params):
    """Return the model class by its name."""
    module_name, class_name = params.model.name.rsplit('.', 1)
    i = importlib.import_module(module_name)
    try:
        return getattr(i, class_name)
    except AttributeError:
        import inspect
        class_names = [m[0] for m in inspect.getmembers(i, inspect.isclass) if m[1].__module__ == module_name]
        raise AttributeError(
            "%s not found in %s. Available class: %s" % (class_name, module_name, ', '.join(class_names)))


def get_loss_fn(params):
    """Return the loss class by its name."""
    i = importlib.import_module("dlex.utils.losses")
    return getattr(i, params.loss)


def get_optimizer(cfg, model_parameters):
    """Return the optimizer object by its type."""
    op_params = cfg.to_dict()
    del op_params['name']

    optimizer_cls = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adagrad': torch.optim.Adagrad,
        'adadelta': torch.optim.Adadelta
    }
    if cfg.name in optimizer_cls:
        optimizer = optimizer_cls[cfg.name]
    else:
        module_name, class_name = cfg.name.rsplit('.', 1)
        i = importlib.import_module(module_name)
        optimizer = getattr(i, class_name)
    return optimizer(model_parameters, **op_params)


def get_lr_scheduler(cfg, optimizer):
    scheduler_params = cfg.to_dict()
    # del scheduler_params['name']
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        **scheduler_params)
    return scheduler


def rnn_cell(cell):
    if cell == 'lstm':
        return torch.nn.LSTM
    elif cell == 'gru':
        return torch.nn.GRU


def linear_layers(
        dims: List[int],
        norm: nn.Module = nn.LayerNorm,
        dropout: int = 0.0,
        activation_fn="relu",
        ignore_last_layer=True):
    linear_layers = []
    for i, in_dim, out_dim in zip(range(len(dims) - 1), dims[:-1], dims[1:]):
        linear_layers.append(nn.Linear(in_dim, out_dim))
        if norm:
            linear_layers.append(norm(out_dim))
        if dropout > 0:
            linear_layers.append(nn.Dropout(dropout))
        if not (ignore_last_layer and i == len(dims) - 2):
            if activation_fn:
                linear_layers.append(get_activation_fn(activation_fn)())
    return nn.Sequential(*linear_layers)


def get_activation_fn(fn):
    if fn == 'relu':
        return nn.ReLU
    elif fn == 'elu':
        return nn.ELU
    else:
        raise ValueError("%s is not a valid activation function" % fn)


class MultiLinear(nn.Module):
    def __init__(
            self,
            dims,
            embed_dim: int = 0,
            norm_layer: Union[nn.Module, None] = nn.LayerNorm,
            dropout: float = 0.0,
            activation_fn: str = 'relu',
            last_layer_activation_fn: str = None,
            residual_connection=False):
        """

        :param dims:
        :param embed_dim:
        :param norm_layer:
        :param dropout:
        :param activation_fn:
        :param last_layer_activation_fn:
        :param residual_connection:
        """
        super().__init__()

        layers = []
        for i, in_dim, out_dim in zip(range(len(dims) - 1), dims[:-1], dims[1:]):
            sub_layers = [nn.Linear(in_dim + embed_dim, out_dim)]
            if norm_layer:
                sub_layers.append(norm_layer(out_dim))

            if dropout > 0.:
                sub_layers.append(nn.Dropout(dropout))

            is_last_layer = (i == len(dims) - 2)
            if not is_last_layer and activation_fn:
                sub_layers.append(get_activation_fn(activation_fn)())
            elif is_last_layer and last_layer_activation_fn:
                sub_layers.append(get_activation_fn(last_layer_activation_fn)())

            layers.append(nn.Sequential(*sub_layers))

        self.layers = nn.ModuleList(layers)
        self.residual_connection = residual_connection

    def __getitem__(self, item):
        return self.layers[item]

    def __len__(self):
        return len(self.layers)

    def forward(self, X: torch.FloatTensor, append_emb=None, output_all=False):
        outputs = []
        for layer in self.layers:
            residual = X
            if append_emb is not None and type(layer) == nn.Linear:
                X = torch.cat([X, append_emb], -1)
            X = layer(X)
            if self.residual_connection and X.shape[-1] == residual.shape[-1]:
                X += residual
            outputs.append(X)

        return outputs if output_all else outputs[-1]


class RNN(nn.Module):
    def __init__(
            self,
            rnn_type: str,
            input_dim: int,
            hidden_dim: int,
            bidirectional: bool,
            dropout: float):
        super().__init__()

        if rnn_type in ["rnn", "lstm", "gru"]:
            rnn_cls = getattr(nn, rnn_type.upper())

        self.bidirectional = bidirectional
        if not bidirectional:
            self.rnn = rnn_cls(input_dim, hidden_dim, batch_first=True, dropout=dropout)
        else:
            self.rnn = rnn_cls(input_dim, hidden_dim // 2, batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, inputs, input_lengths):
        self.rnn.flatten_parameters()
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)
        if self.bidirectional:
            c, (h, _) = self.rnn(inputs)
            c, _ = nn.utils.rnn.pad_packed_sequence(c, batch_first=True)
            h = h.permute(1, 0, 2).contiguous().view(c.shape[0], -1)
            return c, h
        else:
            c, h = self.rnn(inputs)
            c, _ = nn.utils.rnn.pad_packed_sequence(c, batch_first=True)
            return c, h


class ExponentialMovingAverage(nn.Module):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def forward(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average