# Optuna-Integration

[![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/optuna/optuna-integration)
[![Codecov](https://codecov.io/gh/optuna/optuna-integration/branch/main/graph/badge.svg)](https://codecov.io/gh/optuna/optuna-integration/branch/main)
<!-- [![pypi](https://img.shields.io/pypi/v/optuna.svg)](https://pypi.python.org/pypi/optuna-integration) -->
<!-- [![conda](https://img.shields.io/conda/vn/conda-forge/optuna.svg)](https://anaconda.org/conda-forge/optuna-integration) -->
[![Read the Docs](https://readthedocs.org/projects/optuna-integration/badge/?version=stable)](https://optuna-integration.readthedocs.io/en/stable/)

[**Docs**](https://optuna-integration.readthedocs.io/en/stable/)

*Optuna-Integration* is an integration module of [Optuna](https://github.com/optuna/optuna).
This package allows us to use Optuna, an automatic Hyperparameter optimization software framework,
integrated with many useful tools like PyTorch, sklearn, TensorFlow, etc.

## Integrations

Optuna-Integration API reference is [here](https://optuna-integration.readthedocs.io/en/stable/reference/index.html).

* [AllenNLP](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#allennlp)  ([example](https://github.com/optuna/optuna-examples/tree/main/allennlp))
* [BoTorch](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#botorch)
* [Catalyst](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#catalyst)  ([example](https://github.com/optuna/optuna-examples/blob/main/pytorch/catalyst_simple.py))
* [CatBoost](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#catboost)  ([example](https://github.com/optuna/optuna-examples/blob/main/catboost/catboost_pruning.py))
* [Chainer](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#chainer)  ([example](https://github.com/optuna/optuna-examples/tree/main/chainer/chainer_integration.py))
* [ChainerMN](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#chainermn)  ([example](https://github.com/optuna/optuna-examples/tree/main/chainer/chainermn_simple.py))
* [Dask](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#dask) ([example](https://github.com/optuna/optuna-examples/tree/main/dask/dask_simple.py))
* FastAI ([V1](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#fastaiv1)  ([example](https://github.com/optuna/optuna-examples/tree/main/fastai/fastaiv1_simple.py)), ([V2](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#fastaiv2)  ([example]https://github.com/optuna/optuna-examples/tree/main/fastai/fastaiv2_simple.py)))
* [Keras](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#keras)  ([example](https://github.com/optuna/optuna-examples/tree/main/keras))
* [LightGBM](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#lightgbm)  ([example](https://github.com/optuna/optuna-examples/tree/main/lightgbm))
* [MXNet](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#mxnet)  ([example](https://github.com/optuna/optuna-examples/tree/main/mxnet))
* [pycma](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#pycma)
* [scikit-optimize](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#skopt)
* [SHAP](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#shap)
* [sklearn](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#sklearn)  ([example](https://github.com/optuna/optuna-examples/tree/main/sklearn/sklearn_optuna_search_cv_simple.py))
* [skorch](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#skorch)  ([example](https://github.com/optuna/optuna-examples/tree/main/pytorch/skorch_simple.py))
* [TensorBoard](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#tensorboard)  ([example](https://github.com/optuna/optuna-examples/tree/main/tensorboard/tensorboard_simple.py))
* [tf.keras](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#tensorflow)  ([example](https://github.com/optuna/optuna-examples/tree/main/tfkeras/tfkeras_integration.py))
* [Weights & Biases](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#wandb)  ([example](https://github.com/optuna/optuna-examples/blob/main/wandb/wandb_integration.py))
* [XGBoost](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#xgboost)  ([example](https://github.com/optuna/optuna-examples/tree/main/xgboost/xgboost_integration.py))

## Installation

Optuna-Integration is available at [the Python Package Index](https://pypi.org/project/optuna-integration/) and on [Anaconda Cloud](https://anaconda.org/conda-forge/optuna-integration).

```bash
# PyPI
$ pip install optuna-integration
```

```bash
# Anaconda Cloud
$ conda install -c conda-forge optuna-integration
```

Optuna-Integration supports from Python 3.7 to Python 3.10.

Also, we also provide Optuna docker images on [DockerHub](https://hub.docker.com/r/optuna/optuna).

## Communication

* [GitHub Discussions] for questions.
* [GitHub Issues] for bug reports and feature requests.

[GitHub Discussions]: https://github.com/optuna/optuna-integration/discussions
[GitHub issues]: https://github.com/optuna/optuna-integration/issues

## Contribution

Any contributions to Optuna-Integration are more than welcome!

For general guidelines how to contribute to the project, take a look at [CONTRIBUTING.md](./CONTRIBUTING.md).

## Reference

Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019.
Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD ([arXiv](https://arxiv.org/abs/1907.10902)).
