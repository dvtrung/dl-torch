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

def save_checkpoint(tag, params, model, optim):
    os.makedirs(os.path.join("saved_models", params.path), exist_ok=True)
    state = {
        'global_step': model.global_step,
        'model': model.state_dict(),
        'optim': optim.state_dict()
    }
    fn = os.path.join("saved_models", params.path, tag + ".pt")
    torch.save(state, fn)

def load_checkpoint(tag, params, model, optim):
    fn = os.path.join("saved_models", params.path, tag + ".pt")
    if os.path.isfile(fn):
        checkpoint = torch.load(fn)
        model.global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optim'])
