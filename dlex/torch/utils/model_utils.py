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
        activation_fn="relu"):
    linear_layers = []
    for i, in_dim, out_dim in zip(range(len(dims) - 1), dims[:-1], dims[1:]):
        linear_layers.append(nn.Linear(in_dim, out_dim))
        if norm:
            linear_layers.append(norm(out_dim))
        if dropout > 0:
            linear_layers.append(nn.Dropout(dropout))
        if activation_fn and i != len(dims) - 1:
            linear_layers.append(dict(
                relu=nn.ReLU
            )[activation_fn]())
    return nn.Sequential(*linear_layers)