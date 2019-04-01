import importlib, os
import torch

def get_model(params):
    i = importlib.import_module("models." + params.model)
    return i.Model

def get_dataset(params):
    i = importlib.import_module("datasets." + params.dataset.name)
    return i.Dataset
