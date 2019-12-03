Getting Started
================================

Install
------------

dlex can be install using ``pip``

::

  pip install dlex

Set up an experiment
------------------------

Step 1:  Folder structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following folder structure is for best practice

::

  Experiment/
  |-- model_configs
  |-- model_outputs*
  |-- model_reports*
  |-- logs*
  |-- saved_models*
  |-- src
  |   |-- datasets
  |   |   |-- <dataset>.py
  |   |-- models
  |   |   |-- <model>.py
  |-- README.md


Model parameters and outputs are saved to ``./saved_models`` and ``./model_outputs`` unless ``DLEX_SAVED_MODELS_PATH`` and ``DLEX_MODEL_OUTPUTS_PATH`` is specified. The folders with * do not need to be created manually, instead they will be created during training and evaluation if missing.

Step 2: Define dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two classes are required to load and inject data to model.

- ``DatasetBuilder``: handle downloading and preprocessing data. ``DatasetBuilder`` should be framework and config independent.
- ``PytorchDataset``, ``TensorflowDataset``, ``SklearnDataset``: handle loading dataset from the storage; perform shuffle, sorting, batchification, etc. using concepts from each framework

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

In this example, we use ``dlex.datasets.image.MNIST`` and ``dlex.datasets.image.CIFAR10``. You do not need to write even a line of code!

Step 3: Construct model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Besides standard module definition, model also needs to support loss calculation, training, predicting and outputting prediction to specified format.

Model receives a ``MainConfig`` instance containing all configurations and a dataset instance (which is one of ``PytorchDataset``, ``TensorflowDataset`` or ``SklearnDataset`` depending on the framework).

.. literalinclude:: ../../implementations/image_classification/src/models/vgg.py


Step 4: Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lots of code can be reduced with configurations. See the complete guide in :ref:`configs`.

.. literalinclude:: ../../implementations/image_classification/model_configs/vgg16.yml
  :language: yml

Step 5: Train & evaluate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training and evaluation are handled by dlex. Simply execute these commands:

.. code-block:: bash

  dlex train -c <config_path>
  dlex evaluate -c <config_path>
  dlex infer -c <config_path>

or

.. code-block:: bash

  python -m dlex.train -c <config_path>