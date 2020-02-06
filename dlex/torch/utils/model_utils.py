"""Model utils"""
from typing import List
import importlib

import torch
import torch.nn as nn


def get_model(params):
    """Return the model class by its name."""
    module_name, class_name = params.model.name.rsplit('.', 1)
    i = importlib.import_module(module_name)
    return getattr(i, class_name)


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
                linear_layers.append(dict(
                    relu=nn.ReLU
                )[activation_fn]())
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
            norm_layer: nn.Module = nn.LayerNorm,
            dropout=0.0,
            activation_fn='relu',
            last_layer_activation_fn=None):
        super().__init__()

        layers = []
        for i, in_dim, out_dim in zip(range(len(dims) - 1), dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim + embed_dim, out_dim))
            if norm_layer:
                layers.append(norm_layer(out_dim))

            if dropout > 0.:
                layers.append(nn.Dropout(dropout))

            is_last_layer = (i == len(dims) - 2)
            if not is_last_layer and activation_fn:
                layers.append(get_activation_fn(activation_fn)())
            elif is_last_layer and last_layer_activation_fn:
                layers.append(get_activation_fn(last_layer_activation_fn)())

        self.layers = nn.ModuleList(layers)

    def __getitem__(self, item):
        return self.layers[item]

    def __len__(self):
        return len(self.layers)

    def forward(self, X: torch.FloatTensor, append_emb=None):
        for layer in self.layers:
            if append_emb is not None and type(layer) == nn.Linear:
                X = torch.cat([X, append_emb], -1)
            X = layer(X)
        return X