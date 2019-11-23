Getting Started
================================

Install
------------

::

  pip install dlex

Set up an experiment
------------------------

Step 1:  Folder structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

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


Model parameters and outputs are saved to ``./saved_models`` and ``./model_outputs`` unless ``DLEX_SAVED_MODELS_PATH`` and ``DLEX_MODEL_OUTPUTS_PATH`` is specified

Step 2: Define dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Dataset Builder`: handle downloading and preprocessing data. `DatasetBuilder` should be framework and config independent.
- `PytorchDataset`, `TensorflowDataset`: handle loading dataset from the storage, shuffle, sort, batchify, etc. using concepts from each framework

.. code-block:: python

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


Step 3: Construct model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Model supports loss calculation, training, predicting and outputting prediction to specified format.

.. code-block::python

    import torch.nn.functional as F
    import torch.nn as nn

    from dlex.torch.models import ClassificationBaseModel
    from dlex.torch import Batch


    class SimpleModel(ClassificationBaseModel):
        def __init__(self, params, dataset):
            super().__init__(params, dataset)
            self.conv1 = nn.Conv2d(
                in_channels=dataset.num_channels,
                out_channels=20,
                kernel_size=5,
                stride=1, padding=2)
            self.conv2 = nn.Conv2d(
                in_channels=20,
                out_channels=50,
                kernel_size=5,
                stride=1, padding=2)
            self.fc1 = nn.Linear((dataset.input_shape[0] // 4) * (dataset.input_shape[1] // 4) * 50, 500)
            self.fc2 = nn.Linear(500, dataset.num_classes)

        def forward(self, batch: Batch):
            x = batch.X
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

Step 4: Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lots of code can be reduced with configurations. See the complete guide in :ref:`configs`.

Step 5: Train & evaluate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  dlex train -c <config_path>
  dlex evaluate -c <config_path>
  dlex infer -c <config_path>

or

.. code-block:: bash

  python -m dlex.train -c <config_path>