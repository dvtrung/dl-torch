"""Model utils"""

import importlib
import os

import torch

from dlex.configs import ModuleConfigs
from dlex.utils.logging import logger


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
    op_params = cfg.copy()
    del op_params['name']

    optimizer = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adagrad': torch.optim.Adagrad,
        'adadelta': torch.optim.Adadelta
    }[cfg.name]
    return optimizer(model_parameters, **op_params)


def get_lr_scheduler(cfg, optimizer):
    scheduler_params = cfg.copy()
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
