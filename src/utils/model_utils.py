import importlib, os
import torch

def get_model(params):
    i = importlib.import_module("models." + params.model)
    return i.Model

def get_dataset(params):
    i = importlib.import_module("datasets." + params.dataset.name)
    return i.Dataset

def get_optimizer(params, model):
    op_params = params.optimizer
    if params.optimizer.name == 'sgd':
        return torch.optim.SGD(
            model.parameters(), 
            lr=params.optimizer.learning_rate,
            momentum=params.optimizer.momentum)
    elif params.optimizer.name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=params.optimizer.learning_rate,
            weight_decay=params.optimizer.weight_decay)