import abc
import os
from dataclasses import dataclass
from typing import List, NamedTuple, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dlex.configs import ModuleConfigs, AttrDict, Params
from dlex.datasets.torch import Dataset
from dlex.torch import Batch
from dlex.torch.utils.model_utils import get_optimizer, get_lr_scheduler, ExponentialMovingAverage
from dlex.utils.logging import logger


@dataclass
class InferenceOutput:
    output = None
    result = None
    loss: float


class BaseModel(nn.Module):
    config_class = AttrDict
    """
    :param params:
    :type params: Params
    :param dataset:
    :type dataset: PytorchDataset
    """

    def __init__(self, params: Params, dataset: Dataset):
        super().__init__()
        self.params = params
        self.dataset = dataset
        self._num_samples = 0

    @property
    def configs(self):
        return self.params.model

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplemented

    @abc.abstractmethod
    def infer(self, batch):
        """Infer from batch

        :param batch:
        :return: tuple containing:
            pred: prediction
            ref: reference
            model_outputs
            others
        :rtype: tuple
        """
        raise NotImplemented

    def train_log(self, batch, output, verbose):
        d = dict()
        if verbose:
            d["loss"] = self.model.get_loss(batch, output).item()
        return d

    def infer_log(self, batch, output, verbose):
        return dict()

    @abc.abstractmethod
    def get_loss(self, batch, output):
        """Return model loss to optimize

        :param batch:
        :param output: Output of model forward
        :type output:
        :return: A `torch.FloatTensor` with the loss value.
        """
        raise NotImplementedError

    def get_metrics(self, batch: Batch, output) -> Dict[str, Tuple[Union[int, float], int]]:
        return {}

    def to_cuda_tensors(self, tensors):
        if isinstance(tensors, torch.Tensor):
            return tensors.cuda()
        elif isinstance(tensors, list) or isinstance(tensors, tuple):
            return [t.cuda() if t is not None else None for t in tensors]
        else:
            raise ValueError


class ModelWrapper:
    epoch_loss_total = 0.
    epoch_loss_count = 0

    def __init__(self, model, gpus = None):
        self.gpus = gpus
        self.model = model
        if gpus:
            model.to(gpus[0])
            self.module = nn.DataParallel(model, gpus)
        else:
            self.module = model

        self.global_step = 0
        self.current_epoch = 0
        self.params = self.model.params
        self.dataset = self.model.dataset
        self._optimizers = None
        self._lr_schedulers = None
        self._loss_fn = None
        self._num_samples = 0
        self._metrics = {}

        if self.params.train.ema_decay_rate:
            self.ema = ExponentialMovingAverage(self.params.train.ema_decay_rate)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.ema.register(name, param.data)
        else:
            self.ema = None

    def reset_counter(self):
        self._num_samples = 0
        self.epoch_loss_total = 0.
        self.epoch_loss_count = 0
        self._metrics = {}

    def training_step(self, batch):
        self.module.train(True)
        self.module.zero_grad()
        if batch is None or (isinstance(batch, Batch) and len(batch.Y) == 0):
            raise Exception("Empty batch.")

        output = self.module.forward(batch)
        loss = self.model.get_loss(batch, output)
        metrics = self.model.get_metrics(batch, output)
        for metric, (total, num) in metrics.items():
            if metric not in self._metrics:
                self._metrics[metric] = (total, num)
            else:
                _total, _num = self._metrics[metric]
                self._metrics[metric] = (total + _total, num + _num)

        if np.isnan(loss.item()):
            raise Exception("NaN loss.")

        loss.backward()
        # clip grad norm
        if self.params.train.max_grad_norm is not None and self.params.train.max_grad_norm > 0:
            # params = itertools.chain.from_iterable([group['params'] for group in optimizer.param_groups])
            nn.utils.clip_grad_norm_(self.model.parameters(), self.params.train.max_grad_norm)

        for optimizer in self.optimizers:
            optimizer.step()

        if self.ema is not None:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = self.ema(name, param.data)

        # log_dict = self.train_log(batch, output, verbose=self.params.verbose)
        # if len(log_dict) > 0:
        #     logger.info(log_dict)

        # update accumulative loss
        self.epoch_loss_total += loss.detach().item()
        self.epoch_loss_count += 1

        return loss.detach().item()

    def get_metrics(self) -> Dict[str, float]:
        return {metric: total / num for metric, (total, num) in self._metrics.items()}

    def end_training_epoch(self):
        if self.lr_schedulers:
            for lr_scheduler in self.lr_schedulers:
                lr_scheduler.step(self.epoch)

    @property
    def optimizers(self) -> List[torch.optim.Optimizer]:
        if self._optimizers is None:
            self._optimizers = [get_optimizer(self.params.train.optimizer, self.model.parameters())]
            if self.params.train.lr_scheduler:
                self._lr_schedulers = [get_lr_scheduler(
                    self.params.train.lr_scheduler,
                    self.optimizers[0])]
        return self._optimizers

    @property
    def lr_schedulers(self):
        return self._lr_schedulers

    def learning_rates(self) -> List[int]:
        return [opt.param_groups[0]['lr'] for opt in self.optimizers]

    @property
    def loss_fn(self):
        if self._loss_fn is None:
            raise Exception("Loss function must be assigned")
        return self._loss_fn

    @property
    def configs(self):
        return self.params.model

    def load(self, tag):
        path = os.path.join(self.params.checkpoint_dir, tag + ".pt")
        self.load_state_dict(torch.load(path))

    @property
    def epoch(self):
        return self.global_step / len(self.dataset)

    @abc.abstractmethod
    def infer(self, batch):
        """Infer"""
        self.module.train(False)
        return self.model.infer(batch)

    def write_summary(self, summary_writer, batch, output):
        pass

    @property
    def epoch_loss(self):
        return self.epoch_loss_total / self.epoch_loss_count if self.epoch_loss_count > 0 else None

    def save_checkpoint(self, tag):
        """Save current training state"""
        os.makedirs(self.params.checkpoint_dir, exist_ok=True)
        state = {
            'training_id': self.params.training_id,
            'global_step': self.global_step,
            'epoch_loss_total': self.epoch_loss_total,
            'epoch_loss_count': self.epoch_loss_count,
            'model': self.model.state_dict(),
            'optimizers': [optimizer.state_dict() for optimizer in self.optimizers]
        }
        fn = os.path.join(self.params.checkpoint_dir, tag + ".pt")
        torch.save(state, fn)
        logger.debug("Checkpoint saved to %s", fn)

    def load_checkpoint(self, tag, load_optimizers=True):
        """Load from saved state"""
        file_name = os.path.join(self.params.checkpoint_dir, tag + ".pt")
        logger.info("Load checkpoint from %s" % file_name)
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name, map_location='cpu')
            self.params.training_id = checkpoint['training_id']
            logger.info(checkpoint['training_id'])
            self.global_step = checkpoint['global_step']
            self.epoch_loss_count = checkpoint['epoch_loss_count']
            self.epoch_loss_total = checkpoint['epoch_loss_total']
            self.model.load_state_dict(checkpoint['model'])
            if load_optimizers:
                for i, optimizer in enumerate(self.optimizers):
                    optimizer.load_state_dict(checkpoint['optimizers'][i])

            return self.params.training_id
        else:
            raise Exception("Checkpoint not found: %s" % file_name)


class ClassificationModel(BaseModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self._criterion = nn.CrossEntropyLoss()

    def infer(self, batch):
        logits = self.forward(batch)
        return self.get_predictions(logits).tolist(), batch.Y.tolist()

    def get_predictions(self, output):
        if isinstance(output, list):  # ensemble
            logits = sum(F.softmax(l, -1) for l in output)
        else:
            logits = F.softmax(output, -1)
        return torch.max(logits, 1)[1]

    def get_loss(self, batch, output):
        if isinstance(output, list):  # ensemble
            y = self.to_cuda_tensors(batch.Y)
            return sum(self._criterion(o, y) for o in output)
        else:
            return self._criterion(output, self.to_cuda_tensors(batch.Y))

    def get_metrics(self, batch: Batch, output) -> Dict[str, Tuple[Union[int, float], int]]:
        preds = self.get_predictions(output)
        accuracy = torch.sum(preds.cpu() == batch.Y)
        return dict(
            acc=(accuracy.detach().numpy() * 100, len(batch))
        )


def default_params(default):
    def wrap_fn(cls):
        class wrap_cls(cls):
            def __init__(self, params, dataset):
                params.model.extend_default_keys(default)
                super().__init__(params, dataset)
        return wrap_cls
    return wrap_fn
