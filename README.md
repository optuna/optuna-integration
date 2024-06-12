# Optuna-Integration

[![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/optuna/optuna-integration)
[![Codecov](https://codecov.io/gh/optuna/optuna-integration/branch/main/graph/badge.svg)](https://codecov.io/gh/optuna/optuna-integration/branch/main)
[![Read the Docs](https://readthedocs.org/projects/optuna-integration/badge/?version=stable)](https://optuna-integration.readthedocs.io/en/stable/)

This package is an integration module of [Optuna](https://github.com/optuna/optuna), an automatic Hyperparameter optimization software framework.
The modules in this package provide users with extended functionalities for Optuna in combination with third-party libraries such as PyTorch, sklearn, and TensorFlow.

> [!NOTE]
> You can find more information in [**our official documentations**](https://optuna-integration.readthedocs.io/en/stable/) and [**API reference**](https://optuna-integration.readthedocs.io/en/stable/reference/index.html).

## Integration Modules

Here is the table of optuna-integration modules:

|Third Party Library|Integration Module| Example |
|:--|:--|:--|
|AllenNLP|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#allennlp)|[Example](https://github.com/optuna/optuna-examples/tree/main/allennlp)|
|BoTorch|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#botorch)| Unavailable |
|CatBoost|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#catboost)|[Example](https://github.com/optuna/optuna-examples/blob/main/catboost/catboost_pruning.py)|
|Chainer|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#chainer)|[Example](https://github.com/optuna/optuna-examples/tree/main/chainer/chainer_integration.py)|
|ChainerMN| [Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#chainermn) | [Example](https://github.com/optuna/optuna-examples/tree/main/chainer/chainermn_simple.py) |
|Dask|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#dask)|[Example](https://github.com/optuna/optuna-examples/tree/main/dask/dask_simple.py)|
|FastAI|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#fast-ai)|[Example](https://github.com/optuna/optuna-examples/tree/main/fastai/fastai_simple.py)|
|Keras|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#keras)|[Example](https://github.com/optuna/optuna-examples/tree/main/keras)|
|LightGBM|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#lightgbm)|[Example](https://github.com/optuna/optuna-examples/tree/main/lightgbm)|
|MLflow|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#mlflow)|[Example](https://github.com/optuna/optuna-examples/blob/main/mlflow/)|
|MXNet|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#mxnet)|[Example](https://github.com/optuna/optuna-examples/tree/main/mxnet)|
|PyTorch|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#pytorch)|[Example](https://github.com/optuna/optuna-examples/tree/main/pytorch)|
|pycma|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#pycma)|Unavailable|
|SHAP|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#shap)|Unavailable|
|scikit-learn|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#sklearn)|[Example](https://github.com/optuna/optuna-examples/tree/main/sklearn/sklearn_optuna_search_cv_simple.py)|
|skorch|[skorch](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#skorch)|[Example](https://github.com/optuna/optuna-examples/tree/main/pytorch/skorch_simple.py)|
|TensorBoard|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#tensorboard)|[Example](https://github.com/optuna/optuna-examples/tree/main/tensorboard/tensorboard_simple.py)|
|tf.keras|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#tensorflow)|[Example](https://github.com/optuna/optuna-examples/tree/main/tfkeras/tfkeras_integration.py)|
|Weights & Biases|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#wandb)|[Example](https://github.com/optuna/optuna-examples/blob/main/wandb/wandb_integration.py)|
|XGBoost|[Link](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#xgboost)|[Example](https://github.com/optuna/optuna-examples/tree/main/xgboost/xgboost_integration.py)|

## Installation

Optuna-Integration is available at [the Python Package Index](https://pypi.org/project/optuna-integration/) and
on [Anaconda Cloud](https://anaconda.org/conda-forge/optuna-integration).

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
