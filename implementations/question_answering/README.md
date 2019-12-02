Implementations of question answering models

# Datasets

- SQuAD v1

# Models

- Dynamic Coattention Networks (DCN)
- QANet ([paper](https://arxiv.org/abs/1804.09541))
  - [SQuAD](./model_configs/squad_qanet.yml)
  
# Run

```
python -m dlex.train -c squad_qanet
```