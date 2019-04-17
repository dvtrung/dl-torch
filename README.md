# Features

This project provides a codebase for deep learning experiments with Pytorch.

- [ ] Writing minimal code to set up an experiment
- [ ] Efficient way of logging and analyzing training models
- [ ] Terminal GUI for tracking experiments
- [ ] Integration with Google Colab or remote SSH server

# Set up an experiment

## Define dataset

```python
from datasets.base import BaseDataset

class Dataset(BaseDataset):
    def __init__(self, mode, params):
        super().__init__(mode, params)
```

## Construct model

```python
from models.base import BaseModel

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
  name: autoencoder
dataset:
  name: mnist
batch_size: 256
num_epochs: 30
optimizer:
  name: adam
  learning_rate: 0.01
  weight_decay: 1e-5
```

## Train

```
python src/train.py -c <config_path>
```

# Logs & Outputs

# Examples

## Models

|Model|Usage|
|------|-----|
| [rnn-crf](docs/models/rnn-crf.md) | Sequence labelling |
| cnn | Image recognition |
| autoencoder |  |
| [attention](docs/models/attention.md) | Sequence to sequence |


## Datasets

|Dataset|Description|
|------|-----|
| mnist | |
| vnpos | |
