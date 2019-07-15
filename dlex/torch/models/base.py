import abc
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from dlex.configs import AttrDict, ModuleConfigs
from dlex.torch import Batch
from dlex.torch.utils.model_utils import get_optimizer
from dlex.utils.logging import logger


@dataclass
class InferenceOutput:
    output = None
    result = None
    loss: float


class BaseModel(nn.Module):
    config_class = AttrDict

    def __init__(self, params: AttrDict, dataset):
        super().__init__()
        self.params = params
        self.dataset = dataset

    @abc.abstractmethod
    def infer(self, batch: Batch):
        """Infer"""
        raise Exception("Inference method is not implemented")

    def train_log(self, batch: Batch, output, verbose):
        d = dict()
        if verbose:
            d["loss"] = self.get_loss(batch, output).item()
        return d

    def infer_log(self, batch: Batch, output, verbose):
        return dict()

    @abc.abstractmethod
    def get_loss(self, batch, output):
        raise Exception("Loss is not implemented")


class DataParellelModel(nn.DataParallel):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.global_step = 0
        self.current_epoch = 0
        self.params = self.module.params
        self.dataset = self.module.dataset
        self._optimizers = None
        self._loss_fn = None

    def training_step(self, batch: Batch):
        self.module.train(True)
        self.zero_grad()
        output = self.forward(batch)
        loss = self.get_loss(batch, output)
        if np.isnan(loss.item()):
            raise Exception("NaN loss.")
        loss.backward()
        for optimizer in self.optimizers:
            if self.params.train.max_grad_norm is not None and self.params.train.max_grad_norm > 0:
                # params = itertools.chain.from_iterable([group['params'] for group in optimizer.param_groups])
                nn.utils.clip_grad_norm_(self.parameters(), self.params.train.max_grad_norm)
            optimizer.step()
        log_dict = self.module.train_log(batch, output, verbose=self.params.verbose)
        if len(log_dict) > 0:
            logger.info(log_dict)
        return loss.detach().item()

    @property
    def optimizers(self):
        if self._optimizers is None:
            self._optimizers = [get_optimizer(self.params.train.optimizer, self.parameters())]
        return self._optimizers

    @property
    def loss_fn(self):
        if self._loss_fn is None:
            raise Exception("Loss function must be assigned")
        return self._loss_fn

    @property
    def configs(self):
        # Model configs
        return self.params.model

    def load(self, tag):
        path = os.path.join(ModuleConfigs.SAVED_MODELS_PATH, self.params.path, tag + ".pt")
        self.load_state_dict(torch.load(path))

    @property
    def epoch(self):
        return self.global_step / len(self.dataset)

    @abc.abstractmethod
    def infer(self, batch: Batch):
        """Infer"""
        self.module.train(False)
        return self.module.infer(batch)

    def write_summary(self, summary_writer, batch, output):
        pass

    def get_loss(self, batch, output):
        return self.module.get_loss(batch, output)


class ClassificationBaseModel(BaseModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self._criterion = nn.CrossEntropyLoss()

    def infer(self, batch):
        logits = self.forward(batch)
        return torch.max(logits, 1)[1], logits, None

    def get_loss(self, batch: Batch, output):
        return self._criterion(output, batch.Y)


def default_params(default):
    def wrap_fn(cls):
        class wrap_cls(cls):
            def __init__(self, params, dataset):
                params.model.extend_default_keys(default)
                super().__init__(params, dataset)
        return wrap_cls
    return wrap_fn
