Welcome to dlex's documentation!
================================

dlex is an open-source framework for machine learning scientific experiment. Features include:

- Configuration-based experiment setup. Less code for more efficiency and reproducibility
- Pytorch or Tensorflow 2.0 or scikit-learn as backend with similar training flow
- Convenient "environment" for training similar models or tuning hyperparameter


dlex is built on top of other libraries: it provides wrappers for many built-in modules, data loaders and models from other open-source projects (torchtext, torchvision, etc.) so that they can all be used the same manner.

Resources
=================

- `Various model implementations <https://github.com/trungd/dlex/tree/master/implementations>`_
- `Implementations of machine learning algorithms for graph <https://github.com/trungd/ml-graph/>`_

.. toctree::
  :glob:
  :maxdepth: 2
  :caption: Contents:
  :name: globaltoc

  getting_started
  configs

  modules/*



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
