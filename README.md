# Features

This project provides a codebase for deep learning experiments with Pytorch.

- [ ] Efficient way of logging and managing training models
- [ ] Terminal GUI for tracking experiments
- [ ] Export minimal code and run in Google Colab or remote instance

#
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

    def predict(self, batch):
        ...

    def forward(self, batch):
        ...

    def loss(self, batch):
        ...
```

# Configuration example

```yaml
model: autoencoder
dataset:
  name: mnist
batch_size: 16
num_epochs: 30
optimizer:
  name: adam
  learning_rate: 0.01
  weight_decay: 1e-5
```

# Training & Evaluation

```
python src/train.py -c <config_path>
```

# Models

|Model|Usage|
|------|-----|
| [rnn-crf](docs/models/rnn-crf.md) | Sequence labelling |
| cnn | Image recognition |
| autoencoder |  |
| [attention](docs/models/attention.md) | Sequence to sequence |


# Datasets

|Dataset|Description|
|------|-----|
| mnist | |
| vnpos | |
