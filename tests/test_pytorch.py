import os

import numpy as np
import torch
import torch.nn.functional as F
from dlex import MainConfig
from dlex.datasets import DatasetBuilder
from dlex.datasets.torch import Dataset as _Dataset
from dlex.torch import BaseModel
from dlex.torch import PytorchBackend, Batch
from dlex.torch.models import ClassificationModel
from dlex.torch.utils.model_utils import linear_layers
from dlex.torch.utils.ops_utils import maybe_cuda

from utils import model_configs


class Dataset(DatasetBuilder):
    def __init__(self, params: MainConfig):
        super().__init__(params, pytorch_cls=PytorchDataset)


class PytorchDataset(_Dataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)

    def load_data(self):
        return [dict(X=val, Y=val) for val in np.random.choice(
            range(self.configs.num_classes),
            self.configs.num_train if self.mode == "train" else self.configs.num_test, replace=True)]

    def collate_fn(self, batch):
        batch = super().collate_fn(batch)
        return Batch(X=maybe_cuda(batch['X'].reshape(-1, 1)), Y=maybe_cuda(batch['Y']))


class RegressionModel(BaseModel):
    def __init__(self, params: MainConfig, dataset: Dataset):
        super().__init__(params, dataset)
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, batch):
        assert 'X', 'Y' in batch
        return self.linear(batch['X'].float()).reshape(-1)

    def get_loss(self, batch, output):
        return F.mse_loss(output, batch['Y'].float()) / len(batch)

    def infer(self, batch):
        return self.forward(batch).tolist(), batch['Y'].tolist()


class ClassificationModel(ClassificationModel):
    def __init__(self, params, dataset):
        super().__init__(params, dataset)
        self.dense = linear_layers([1, 10, params.dataset.num_classes])

    def forward(self, batch):
        return self.dense(batch['X'].float())


@model_configs(yaml="""backend: pytorch
model:
    name: test_pytorch.RegressionModel
dataset:
    name: test_pytorch.Dataset
    num_train: 100
    num_test: 10
    num_classes: 10
train:
    num_epochs: 1
    batch_size: 10
    optimizer:
        name: adam
        lr: 0.1
test:
    metrics: [mse]""")
def test_logs(configs):
    params = configs.environments[0].configs_list[0]
    params.gpu = [0]
    be = PytorchBackend(None, params, configs, training_idx=0, report_queue=None)
    report = be.run_train()
    assert report.results is not None
    assert report.current_results is not None
    assert report.status == "finished"
    assert os.path.getsize(os.path.join(params.log_dir, "error.log")) == 0
    assert os.path.getsize(os.path.join(params.log_dir, "info.log")) > 0
    assert os.path.getsize(os.path.join(params.log_dir, "debug.log")) > 0


@model_configs(yaml="""backend: pytorch
model:
    name: test_pytorch.RegressionModel
dataset:
    name: test_pytorch.Dataset
    num_train: 100
    num_test: 10
    num_classes: 10
train:
    num_epochs: 10
    batch_size: 10
    optimizer:
        name: adam
        lr: 0.1
test:
    metrics: [mse]""")
def test_regression_model(configs):
    params = configs.environments[0].configs_list[0]
    params.gpu = [0]
    be = PytorchBackend(None, params, configs, training_idx=0, report_queue=None)
    report = be.run_train()
    assert report.training_idx == 0
    assert 'mse' in report.results
    assert report.results['mse'] < 0.01


@model_configs(yaml="""backend: pytorch
model:
    name: test_pytorch.ClassificationModel
dataset:
    name: test_pytorch.Dataset
    num_train: 100
    num_test: 10
    num_classes: 5
train:
    num_epochs: 5
    batch_size: 10
    optimizer:
        name: adam
        lr: 0.01
test:
    metrics: [acc]""")
def test_classification_model(configs):
    params = configs.environments[0].configs_list[0]
    params.gpu = [0]
    be = PytorchBackend(None, params, configs, training_idx=0, report_queue=None)
    report = be.run_train()
    assert report.training_idx == 0
    assert 'acc' in report.results
    assert report.results['acc'] == 100.