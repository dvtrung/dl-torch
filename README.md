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
| rnn-crf | Sequence labelling |
| cnn | Image recognition |
| autoencoder |  |
| attention | Sequence to sequence |


# Datasets

|Dataset|Description|
|------|-----|
| mnist | |
| vnPOS | |
