.. _configs:

Configurations
================================

Top-level entries
-----------------

backend
  Must be included. The following backends are supported

  - pytorch / torch: use ``dlex.torch.BaseModel`` as base model and ``dlex.datasets.torch.Dataset`` as base dataset
  - tensorflow / tf: partially supported
  - sklearn: partially supported

random_seed
  Random seed for all randomization engines (numpy, random, torch, tensorflow etc.)

Environment
-----------

Assume that you want to tune hyper-parameters or run several experiments with similar configurations (e.g. run a model on different dataset). Environment provides a convenient interface to define multiple experiments within a single configuration file.

Each environment has a name and several entries. The argument ``--env``, if specified, indicates which environment(s) to run.

.. code-block::

  python -m dlex.train -c ./model_configs/demo.yml --env small

Entries for each variable:

variables
  Each environment comes with a list of variables and values. Value of each variable can be a single value or a list. If it is a list, all possible combinations of variable values will be examined.

report
  When some variables are assigned multiple values, each combination of them will give one or some results. Use ``report`` to specify how these results are displayed in report.

  - ``type``: ``table`` or ``raw``
  - ``row`` / ``col``: when type is table, indicate name of the variable displayed as row / col

default
  Set to false if the env is not included in default execution. In that case, it can only be run with ``--env`` in the command. All the environments are run by default.

Below is an example of a config file with environments.

.. code-block::

  env:
    small:
      variables:
        dataset: [list of data sets]
        num_layers: [list of model properties 'num_layers']
        batch_size: 128
      report:
        type: table
        row: dataset
        col: num_layers
    large:
      variables:
        dataset: [list of data sets]
        batch_size: 32
        num_layers: 19
  dataset:
    name: ~dataset
    ...
  model:
    num_layers: ~num_layers
    ...
  train:
    batch_size: ~batch_size

Model
------

Model option is defined in ``model`` at top level, which is passed to the model class and can be accessed in ``self.configs``. Standard configs include:

name:
  relative path to model class (similar to importing a module)

and model hyper-parameters (dimension, number of layers, etc.)


Data Set
---------

Data set option is defined in ``dataset`` at top level, which is passed to the model class and can be accessed in ``self.configs``. Standard configs include:

name:
  relative path to database class (inherited from ``dlex.datasets.DatasetBuilder``)

Train
-----

``train`` entry is mapped into ``TrainConfig`` with the following fields and methods

.. autoclass:: dlex.configs.TrainConfig
  :members:

.. autoclass:: dlex.configs.OptimizerConfig
  :members:

Test
----

``test`` entry is mapped into ``TestConfig`` with the following fields and methods

.. autoclass:: dlex.configs.TestConfig
  :members: