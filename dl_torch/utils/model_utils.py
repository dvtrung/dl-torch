"""Model utils"""

import importlib
import os
import json
import torch

from utils.logging import logger


def get_model(params):
    """Return the model class by its name."""
    module_name, class_name = params.model.name.rsplit('.', 1)
    i = importlib.import_module("models." + module_name)
    return getattr(i, class_name)


def get_dataset(params):
    """Return the dataset class by its name."""
    module_name, class_name = params.dataset.name.rsplit('.', 1)
    i = importlib.import_module("datasets." + module_name)
    return getattr(i, class_name)


def get_loss_fn(params):
    """Return the loss class by its name."""
    i = importlib.import_module("utils.losses")
    return getattr(i, params.loss)


def get_optimizer(cfg, model_parameters):
    """Return the optimizer object by its type."""
    op_params = cfg.copy()
    del op_params['name']

    optimizer = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam
    }[cfg.name]
    return optimizer(model_parameters, **op_params)


def save_checkpoint(tag, params, model):
    """Save current training state"""
    os.makedirs(os.path.join("saved_models", params.path), exist_ok=True)
    state = {
        'training_id': params.training_id,
        'global_step': model.global_step,
        'model': model.state_dict(),
        'optimizers': [optimizer.state_dict() for optimizer in model.optimizers]
    }
    fn = os.path.join("saved_models", params.path, tag + ".pt")
    torch.save(state, fn)


def load_checkpoint(tag, params, model):
    """Load from saved state"""
    file_name = os.path.join("saved_models", params.path, tag + ".pt")
    logger.info("Load checkpoint from %s" % file_name)
    if os.path.exists(file_name):
        checkpoint = torch.load(file_name, map_location='cpu')
        params.training_id = checkpoint['training_id']
        logger.info(checkpoint['training_id'])
        model.global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint['model'])
        for i, optimizer in enumerate(model.optimizers):
            optimizer.load_state_dict(checkpoint['optimizers'][i])
    else:
        raise Exception("Checkpoint not found.")


def load_results(params):
    """Load all saved results at each checkpoint."""
    path = os.path.join(params.log_dir, "results.json")
    if os.path.exists(path):
        with open(os.path.join(params.log_dir, "results.json")) as f:
            return json.load(f)
    else:
        return {
            "best_results": {},
            "evaluations": []
        }


def add_result(params, new_result):
    """Add a checkpoint for evaluation result."""
    ret = load_results(params)
    ret["evaluations"].append(new_result)
    for m in params.metrics:
        if m not in ret["best_results"] or \
                new_result['result'][m] > ret['best_results'][m]['result'][m]:
            ret["best_results"][m] = new_result
    with open(os.path.join(params.log_dir, "results.json"), "w") as f:
        f.write(json.dumps(ret, indent=4))
    return ret["best_results"]


def rnn_cell(cell):
    if cell == 'lstm':
        return torch.nn.LSTM
    elif cell == 'gru':
        return torch.nn.GRU
