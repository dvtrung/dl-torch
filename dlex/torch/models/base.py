import abc
import itertools
import os

import torch
import torch.nn as nn

from dlex.torch import Batch
from dlex.torch.utils.losses import nll_loss
from dlex.torch.utils.model_utils import get_optimizer


class BaseModel(nn.Module):
    def __init__(self, params, dataset):
        super().__init__()
        self.params = params
        self.dataset = dataset

    @abc.abstractmethod
    def infer(self, batch: Batch):
        """Infer"""
        return None


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
        self.zero_grad()
        output = self.forward(batch)
        loss = self.get_loss(batch, output)
        loss.backward()
        for optimizer in self.optimizers:
            if self.params.train.max_grad_norm is not None and self.params.train.max_grad_norm > 0:
                params = itertools.chain.from_iterable([group['params'] for group in optimizer.param_groups])
                nn.utils.clip_grad_norm_(params, self.params.train.max_grad_norm)
            optimizer.step()
        return loss

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
    def cfg(self):
        # Model configs
        return self.params.model

    def load(self, tag):
        path = os.path.join("saved_models", self.params.path, tag + ".pt")
        self.load_state_dict(torch.load(path))

    @property
    def epoch(self):
        return self.global_step / len(self.dataset)

    @abc.abstractmethod
    def infer(self, batch: Batch):
        """Infer"""
        return self.module.infer(batch)

    def write_summary(self, summary_writer, batch, output):
        pass

    def get_loss(self, batch, output):
        return self.module.get_loss(batch, output)


class ImageClassificationBaseModel(BaseModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)

    def infer(self, batch):
        logits = self.forward(batch)
        return torch.max(logits, 1)[1], None

    @staticmethod
    def get_loss(batch: Batch, output):
        return nll_loss(batch, output)


def default_params(default):
    def wrap_fn(cls):
        class wrap_cls(cls):
            def __init__(self, params, dataset):
                params.model.extend_default_keys(default)
                super().__init__(params, dataset)
        return wrap_cls
    return wrap_fn
