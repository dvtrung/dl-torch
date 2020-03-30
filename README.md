[!] This code is under development and mainly for my personal use. This project is for fast prototyping of deep learning and machine learning model with minimal code. Some parts of the code may not be well-commented or lack of citation.

dlex is an open source framework for machine learning scientific experiment. 

# Features

- [ ] Configuration-based experiment setup. Less code for more efficiency and reproducibility
- [ ] Pytorch or Tensorflow 2.0 or scikit-learn as backend with similar training flow
- [ ] Convenient "environment" for training similar models or tuning hyperparameter

![anim](anim.gif)

# Install

To install the current release

```
pip install dlex
```

Try your first dlex program

```python
from dlex import yaml_configs, Configs
from dlex.torch import PytorchBackend


@yaml_configs("""backend: pytorch
model:
    name: dlex.torch.models.DNN
    layers: [200, 100]
dataset:
    name: dlex.datasets.MNIST
    num_train: 100
    num_test: 10
    num_classes: 5
train:
    num_epochs: 10
    batch_size: 128
    optimizer:
        name: adam
        lr: 0.01
test:
    metrics: [acc]""")
def train(configs: Configs):
    params = configs.get_default_params()
    report = PytorchBackend(params).run_train()
    print(report.results)


if __name__ == "__main__":
    train()
```

# Resources

- [Documentation](https://trungd.github.io/dlex/)
- [Getting Started](https://trungd.github.io/dlex/getting_started.html)
- [Various model implementations](dlex_impl/README.md)
- [Implementations of machine learning algorithms for graph](https://github.com/trungd/ml-graph/)

# License

# Contributing

Contributions are more than welcome! Please get in touch if you would like to help out.