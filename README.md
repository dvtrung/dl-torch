[!] This code is under development and mainly for my personal use. This project is for fast prototyping of deep learning and machine learning model with minimal code. Some parts of the code may not be well-commented or lack of citation.

# Features

- [ ] Writing minimal code to set up a new experiment
- [ ] Pytorch or Tensorflow 2.0 or scikit-learn as backend with similar training flow
- [ ] Efficiently log and analyze model results
- [ ] Cross-platform GUI for monitoring experiments, either local or remote sever (Electron-based)

![screenshot](./screenshot.png)

See [here](implementations/README.md) for a list of implemented models

# Installation

```
pip install dlex
```

# Set up an experiment

## Step 1:  Folder structure

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

Model parameters and outputs are saved to `./saved_models` and `./model_outputs` unless `DLEX_SAVED_MODELS_PATH` and `DLEX_MODEL_OUTPUTS_PATH` is specified

## Step 2: Define dataset

- `Dataset Builder`: handle downloading and preprocessing data. `DatasetBuilder` should be framework and config independent.
- `PytorchDataset`, `TensorflowDataset`: handle loading dataset from the storage, shuffle, sort, batchify, etc. using concepts from each framework

```python
from dlex.configs import AttrDict
from dlex.datasets.torch import PytorchDataset
from dlex.datasets.builder import DatasetBuilder

class SampleDatasetBuilder(DatasetBuilder):
    def __init__(self, params: AttrDict):
        super().__init__(params)
        
    def maybe_download_and_extract(self, force=False):
        super().maybe_download_and_extract(force)
        # Download dataset...
        # self.download_and_extract([some url], self.get_raw_data_dir())
            
    def maybe_preprocess(self, force=False):
        super().maybe_preprocess(force)
        # Preprocess data...
        
    def get_pytorch_wrapper(self, mode: str):
        return PytorchSampleDataset(self, mode)

class PytorchSampleDataset(PytorchDataset):
    def __init__(self, builder, mode):
        super().__init__(builder, mode)
        # Load data from preprocessed files...
```

## Step 3: Construct model

Model supports loss calculation, training, predicting and outputting prediction to specified format.

```python
from dlex.torch.models.base import BaseModel

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

## Step 4: Configuration

#### Model

#### Dataset

#### Train

- `batch_size`: `int` or `dict` of `{ [progress]: [batch_size] }` (`0` as key must always be included)

- `num_epochs`

- `optimizer`: `dict` of name and optimizer's arguments. Support `sgd`, `adam`, `adadelta`.

#### Test

- `batch_size`: `int`. Training batch size value is used if not specified.

```yaml
model:
  name: {model import path}
  ...{model configs}
dataset:
  name: {dataset import path}
  ...{dataset configs}
train:
  batch_size: 256
  num_epochs: 30
  optimizer:
    name: adam
    learning_rate: 0.01
    weight_decay: 1e-5
```
## Step 5: Train & evaluate

```bash
dlex train <config_path>
dlex evaluate <config_path>
dlex infer <config_path>
```

## Environment Variables

- `DLEX_TMP_PATH` (default: `~/tmp`)
- `DLEX_DATASETS_PATH` (default: `~/tmp/dlex/datasets`)
- `DLEX_SAVED_MODELS_PATH` (default: `./saved_models`)

## Using dlex

[https://github.com/trungd/ml-graph](https://github.com/trungd/ml-graph/): Implementations of machine learning algorithms for graph