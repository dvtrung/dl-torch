import os
import abc
import torch

from ..utils.model_utils import get_optimizer


class BaseModel(torch.nn.Module):
    @abc.abstractmethod
    def __init__(self, params, dataset):
        super().__init__()
        self.params = params
        self.dataset = dataset

        self.global_step = 0

        if torch.cuda.is_available():
            # logger.info("Cuda available: %s", torch.cuda.get_device_name(0))
            self.cuda()

        self._optimizers = None
        self._loss_fn = None

    @property
    def optimizers(self):
        if self._optimizers is None:
            self._optimizers = [get_optimizer(self.params.optimizer, self.parameters())]
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
    def infer(self, batch):
        """Infer"""
        return None

    def training_step(self, batch):
        self.zero_grad()
        output = self.forward(batch)
        loss = self.get_loss(batch, output)
        loss.backward()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss


def default_params(default):
    def wrap_fn(cls):
        class wrap_cls(cls):
            def __init__(self, params, dataset):
                params.model.extend_default_keys(default)
                super().__init__(params, dataset)
        return wrap_cls
    return wrap_fn
