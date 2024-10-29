# Optuna-Integration

[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/optuna-integration.svg)](https://pypi.python.org/pypi/optuna-integration)
[![conda](https://img.shields.io/conda/vn/conda-forge/optuna-integration.svg)](https://anaconda.org/conda-forge/optuna-integration)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/optuna/optuna-integration)
[![Codecov](https://codecov.io/gh/optuna/optuna-integration/branch/main/graph/badge.svg)](https://codecov.io/gh/optuna/optuna-integration/branch/main)
[![Read the Docs](https://readthedocs.org/projects/optuna-integration/badge/?version=stable)](https://optuna-integration.readthedocs.io/en/stable/)

This package is an integration module of [Optuna](https://github.com/optuna/optuna), an automatic Hyperparameter optimization software framework.
The modules in this package provide users with extended functionalities for Optuna in combination with third-party libraries such as PyTorch, sklearn, and TensorFlow.

> [!NOTE]
> You can find more information in [**our official documentations**](https://optuna-integration.readthedocs.io/en/stable/) and [**API reference**](https://optuna-integration.readthedocs.io/en/stable/reference/index.html).

## Installation

Optuna-Integration is available via [pip](https://pypi.org/project/optuna-integration/) and
on [conda](https://anaconda.org/conda-forge/optuna-integration).

```bash
# PyPI
$ pip install optuna-integration

# Anaconda Cloud
$ conda install -c conda-forge optuna-integration
```

> [!IMPORTANT]
> As dependencies of all the modules are large and complicated, the commands above install only the common dependencies.
> Dependencies for each module can be installed via pip.
> For example, if you would like to install the dependencies of `optuna_integration.botorch` and `optuna_integration.lightgbm`, you can install them via:
> ```shell
> $ pip install optuna-integration[botorch,lightgbm]
> ```

> [!NOTE]
> Optuna-Integration supports from Python 3.8 to Python 3.12.
> Optuna Docker image is also provided at [DockerHub](https://hub.docker.com/r/optuna/optuna).

## Integration Modules

Here is the table of `optuna-integration` modules:

|Third Party Library| Example |
|:--|:--|
|[BoTorch](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#botorch)| Unavailable |
|[CatBoost](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#catboost)|[CatBoostPruningCallback](https://github.com/optuna/optuna-examples/blob/main/catboost/catboost_pruning.py)|
|[Dask](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#dask)|[DaskStorage](https://github.com/optuna/optuna-examples/tree/main/dask/dask_simple.py)|
|[FastAI](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#fast-ai)|[FastAIPruningCallback](https://github.com/optuna/optuna-examples/tree/main/fastai/fastai_simple.py)|
|[Keras](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#keras)|[KerasPruningCallback](https://github.com/optuna/optuna-examples/blob/main/keras/keras_integration.py)|
|[LightGBM](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#lightgbm)|[LightGBMPruningCallback](https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_integration.py) / [LightGBMTuner](https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_tuner_simple.py)|
|[MLflow](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#mlflow)|[MLflowCallback](https://github.com/optuna/optuna-examples/blob/main/mlflow/keras_mlflow.py)|
|[MXNet](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#mxnet)|Unavailable|
|[PyTorch Distributed](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#pytorch)|[TorchDistributedTrial](https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py)|
|[PyTorch Ignite](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#pytorch)|[PyTorchIgnitePruningHandler](https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_ignite_simple.py)|
|[PyTorch Lightning](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#pytorch)|[PyTorchLightningPruningCallback](https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py)|
|[pycma](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#pycma)|Unavailable|
|[SHAP](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#shap)|Unavailable|
|[scikit-learn](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#sklearn)|[OptunaSearchCV](https://github.com/optuna/optuna-examples/tree/main/sklearn/sklearn_optuna_search_cv_simple.py)|
|[skorch](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#skorch)|[SkorchPruningCallback](https://github.com/optuna/optuna-examples/tree/main/pytorch/skorch_simple.py)|
|[TensorBoard](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#tensorboard)|[TensorBoardCallback](https://github.com/optuna/optuna-examples/tree/main/tensorboard/tensorboard_simple.py)|
|[tf.keras](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#tensorflow)|[TFKerasPruningCallback](https://github.com/optuna/optuna-examples/tree/main/tfkeras/tfkeras_integration.py)|
|[Weights & Biases](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#wandb)|[WeightsAndBiasesCallback](https://github.com/optuna/optuna-examples/blob/main/wandb/wandb_integration.py)|
|[XGBoost](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#xgboost)|[XGBoostPruningCallback](https://github.com/optuna/optuna-examples/tree/main/xgboost/xgboost_integration.py)|
|[AllenNLP](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#allennlp)*|[AllenNLPPruningCallback](https://github.com/optuna/optuna-examples/blob/main/allennlp/allennlp_simple.py)|
|[Chainer](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#chainer)*|[ChainerPruningExtension](https://github.com/optuna/optuna-examples/tree/main/chainer/chainer_integration.py)|
|[ChainerMN](https://optuna-integration.readthedocs.io/en/stable/reference/index.html#chainermn)* | [ChainerMNStudy](https://github.com/optuna/optuna-examples/tree/main/chainer/chainermn_simple.py) |

> [!WARNING]
> `*` shows deprecated modules and they might be removed in the future.

## Communication

* [GitHub Discussions] for questions.
* [GitHub Issues] for bug reports and feature requests.

[GitHub Discussions]: https://github.com/optuna/optuna-integration/discussions

[GitHub issues]: https://github.com/optuna/optuna-integration/issues

## Contribution

Any contributions to Optuna-Integration are more than welcome!

For general guidelines how to contribute to the project, take a look at [CONTRIBUTING.md](./CONTRIBUTING.md).

## Reference

If you use Optuna in one of your research projects, please cite [our KDD paper](https://doi.org/10.1145/3292500.3330701) "Optuna: A Next-generation Hyperparameter Optimization Framework":

<details open>
<summary>BibTeX</summary>

```bibtex
@inproceedings{akiba2019optuna,
  title={{O}ptuna: A Next-Generation Hyperparameter Optimization Framework},
  author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle={The 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2623--2631},
  year={2019}
}
```
</details>
