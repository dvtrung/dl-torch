import os

import numpy as np
import torch
import torch.nn.functional as F
from dlex import Params, yaml_configs, Configs
from dlex.configs import YamlConfigs
from dlex.datasets import DatasetBuilder
from dlex.datasets.torch import Dataset as _Dataset
from dlex.torch import BaseModel
from dlex.torch import PytorchBackend, Batch
from dlex.torch.models import ClassificationModel
from dlex.torch.utils.model_utils import linear_layers


class Dataset(DatasetBuilder):
    def __init__(self, params: Params):
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
        return Batch(X=self.maybe_cuda(batch['X'].reshape(-1, 1)), Y=self.maybe_cuda(batch['Y']))


class RegressionModel(BaseModel):
    def __init__(self, params: Params, dataset: Dataset):
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


class TestTraining:
    yaml_content = """backend: pytorch
model:
    name: test_pytorch.RegressionModel
env:
    test_1:
        variables:
            num_classes: [5, 10]
    test_2:
        variables:
            num_classes: 15
    test_3:
        variables:
            num_classes: 20
dataset:
    name: test_pytorch.Dataset
    num_train: 100
    num_test: 10
    num_classes: ~num_classes
train:
    num_epochs: 2
    batch_size: 10
    optimizer:
        name: adam
        lr: 0.1
test:
    metrics: [mse]"""

    def test_training(self):
        with YamlConfigs(self.yaml_content, ["--env", "test_1 test_2"]) as configs:
            assert len(configs.environments) == 2

            idx = 0
            for env in configs.environments:
                assert env.name in ["test_1", "test_2"]
                for variable_values, params in zip(env.variables_list, env.configs_list):
                    idx += 1
                    be = PytorchBackend(params, training_idx=idx)
                    report = be.run_train()

                    assert report.training_idx == idx
                    assert report.results
                    assert report.current_results
                    assert report.status == "finished"
                    assert os.path.getsize(os.path.join(params.log_dir, "error.log")) == 0
                    assert os.path.getsize(os.path.join(params.log_dir, "info.log")) > 0
                    assert os.path.getsize(os.path.join(params.log_dir, "debug.log")) > 0
                    assert os.path.exists(os.path.join(params.checkpoint_dir, "latest.pt"))

    def test_gpu(self):
        with YamlConfigs(self.yaml_content, ["--env", "test_2"]) as configs:
            params = configs.get_default_params()
            params.gpu = [0]
            be = PytorchBackend(params)
            model, datasets = be.load_model("train")
            assert next(model.model.parameters()).is_cuda
            report = be.run_train()
            assert report.results


@yaml_configs("""backend: pytorch
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
def test_regression_model(configs: Configs):
    params = configs.get_default_params()
    be = PytorchBackend(params)
    report = be.run_train()
    assert report.training_idx == 0
    assert 'mse' in report.results
    assert report.results['mse'] < 0.01


@yaml_configs("""backend: pytorch
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
def test_classification_model(configs: Configs):
    params = configs.get_default_params()
    be = PytorchBackend(params)
    report = be.run_train()
    assert report.training_idx == 0
    assert 'acc' in report.results
    assert report.results['acc'] == 100.