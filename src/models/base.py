import os
import abc
import torch

from utils.model_utils import get_optimizer


class BaseModel(torch.nn.Module):
    @abc.abstractmethod
    def __init__(self, params, dataset):
        super().__init__()
        self.params = params
        self.cfg = params.model
        self.dataset = dataset

        self.global_step = 0

        if torch.cuda.is_available():
            # logger.info("Cuda available: %s", torch.cuda.get_device_name(0))
            self.cuda()

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

    @abc.abstractmethod
    def _get_loss_fn(self):
        def loss_fn(batch, output):
            raise Exception("Loss must be implemented")
        return loss_fn

    def _get_optimizer(self):
        return get_optimizer(self.params.optimizer, self.parameters())

    def training_step(self, batch):
        self.zero_grad()
        output = self.forward(batch)
        loss = self.loss_fn(batch, output)
        loss.backward()
        self.optimizer.step()


def default_params(default):
    def wrap_fn(cls):
        class wrap_cls(cls):
            def __init__(self, params, dataset):
                params.model.extend_default_keys(default)
                super().__init__(params, dataset)
        return wrap_cls
    return wrap_fn