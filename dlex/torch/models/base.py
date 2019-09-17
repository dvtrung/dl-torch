import abc
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from dlex.configs import AttrDict, ModuleConfigs
from dlex.torch import Batch
from dlex.torch.utils.model_utils import get_optimizer, get_lr_scheduler
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
    epoch_loss_total = 0.
    epoch_loss_count = 0

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.global_step = 0
        self.current_epoch = 0
        self.params = self.module.params
        self.dataset = self.module.dataset
        self._optimizers = None
        self._lr_schedulers = None
        self._loss_fn = None

    def training_step(self, batch: Batch):
        self.module.train(True)
        self.zero_grad()
        if batch is None or len(batch.Y) == 0:
            raise Exception("Empty batch.")
        output = self.forward(batch)
        loss = self.get_loss(batch, output)

        if np.isnan(loss.item()):
            raise Exception("NaN loss.")

        loss.backward()
        # clip grad norm
        if self.params.train.max_grad_norm is not None and self.params.train.max_grad_norm > 0:
            # params = itertools.chain.from_iterable([group['params'] for group in optimizer.param_groups])
            nn.utils.clip_grad_norm_(self.parameters(), self.params.train.max_grad_norm)

        for optimizer in self.optimizers:
            optimizer.step()

        log_dict = self.module.train_log(batch, output, verbose=self.params.verbose)
        if len(log_dict) > 0:
            logger.info(log_dict)

        # update accumulative loss
        self.epoch_loss_total += loss.detach().item()
        self.epoch_loss_count += 1

        return loss.detach().item()

    def end_training_epoch(self):
        if self.lr_schedulers:
            for lr_scheduler in self.lr_schedulers:
                lr_scheduler.step(self.epoch)

    @property
    def optimizers(self):
        if self._optimizers is None:
            self._optimizers = [get_optimizer(self.params.train.optimizer, self.parameters())]
            if self.params.train.lr_scheduler:
                self._lr_schedulers = [get_lr_scheduler(
                    self.params.train.lr_scheduler,
                    self.optimizers[0])]
        return self._optimizers

    @property
    def lr_schedulers(self):
        return self._lr_schedulers

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

    def start_calculating_loss(self):
        self.epoch_loss_total = 0.
        self.epoch_loss_count = 0

    @property
    def epoch_loss(self):
        return self.epoch_loss_total / self.epoch_loss_count if self.epoch_loss_count > 0 else None

    def save_checkpoint(self, tag):
        """Save current training state"""
        os.makedirs(os.path.join(ModuleConfigs.SAVED_MODELS_PATH, self.params.path), exist_ok=True)
        state = {
            'training_id': self.params.training_id,
            'global_step': self.global_step,
            'epoch_loss_total': self.epoch_loss_total,
            'epoch_loss_count': self.epoch_loss_count,
            'model': self.state_dict(),
            'optimizers': [optimizer.state_dict() for optimizer in self.optimizers]
        }
        fn = os.path.join(ModuleConfigs.SAVED_MODELS_PATH, self.params.path, tag + ".pt")
        torch.save(state, fn)

    def load_checkpoint(self, tag, load_optimizers=True):
        """Load from saved state"""
        file_name = os.path.join(ModuleConfigs.SAVED_MODELS_PATH, self.params.path, tag + ".pt")
        logger.info("Load checkpoint from %s" % file_name)
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name, map_location='cpu')
            self.params.training_id = checkpoint['training_id']
            logger.info(checkpoint['training_id'])
            self.global_step = checkpoint['global_step']
            self.epoch_loss_count = checkpoint['epoch_loss_count']
            self.epoch_loss_total = checkpoint['epoch_loss_total']
            self.load_state_dict(checkpoint['model'])
            if load_optimizers:
                for i, optimizer in enumerate(self.optimizers):
                    optimizer.load_state_dict(checkpoint['optimizers'][i])
        else:
            raise Exception("Checkpoint not found: %s" % file_name)


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
