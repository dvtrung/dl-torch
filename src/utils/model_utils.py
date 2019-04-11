"""Model utils"""

import importlib
import os
import json
import torch

from utils.logging import logger

def get_model(params):
    """Return the model class by its name."""
    module_name, class_name = params.model.rsplit('.', 1)
    i = importlib.import_module("models." + module_name)
    return getattr(i, class_name)

def get_dataset(params):
    """Return the dataset class by its name."""
    i = importlib.import_module("datasets." + params.dataset.name)
    return i.Dataset

def get_optimizer(params, model):
    """Return the optimizer object by its type."""
    op_params = params.optimizer.copy()
    del op_params['name']

    if params.optimizer.name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            **op_params)
    elif params.optimizer.name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            **op_params)

def save_checkpoint(tag, params, model, optim):
    os.makedirs(os.path.join("saved_models", params.path), exist_ok=True)
    state = {
        'training_id': params.training_id,
        'global_step': model.global_step,
        'model': model.state_dict(),
        'optim': optim.state_dict()
    }
    fn = os.path.join("saved_models", params.path, tag + ".pt")
    torch.save(state, fn)

def load_checkpoint(tag, params, model, optim):
    fn = os.path.join("saved_models", params.path, tag + ".pt")
    logger.info("Load checkpoint from %s" % fn)
    if os.path.exists(fn):
        checkpoint = torch.load(fn, map_location='cpu')
        params.training_id = checkpoint['training_id']
        logger.info(checkpoint['training_id'])
        model.global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint['model'])
        if optim is not None:
            optim.load_state_dict(checkpoint['optim'])
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
