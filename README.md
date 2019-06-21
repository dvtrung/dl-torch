# Features

This project provides a codebase for deep learning experiments with Pytorch.

- [ ] Writing minimal code to set up an experiment
- [ ] Pytorch or Tensorflow 2.0 as backend with same training flow
- [ ] Efficient way of logging and analyzing training models
- [ ] Terminal GUI for tracking experiments
- [ ] Integration with Google Colab or remote SSH server

# Set up an experiment

# Folder structure

```
Experiment/
|-- model_configs
|-- model_outputs
|-- logs
|-- saved_models
|-- src
|   |-- datasets
|   |   |-- <dataset>.py
|   |-- models
|   |   |-- <model>.py
|-- README.md
```

## Define dataset

```python
from dlex.datasets.base import BaseDataset

class Dataset(BaseDataset):
    def __init__(self, mode, params):
        super().__init__(mode, params)
```

## Construct model

```python
from dlex.models.base import BaseModel

class Model(BaseModel):
     def __init__(self, params, dataset):
        super().__init__(params, dataset)

    def infer(self, batch):
        ...

    def forward(self, batch):
        ...

    def loss(self, batch):
        ...
```

## Configuration

```yaml
model:
  name: {model import path}
  ...{model configs}
dataset:
  name: {dataset import path}
  ...{dataset configs}
batch_size: 256
num_epochs: 30
optimizer:
  name: adam
  learning_rate: 0.01
  weight_decay: 1e-5
```

## Train

```bash
dlex train <config_path>
dlex evaluate <config_path>
dlex infer <config_path>
```