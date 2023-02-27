# Optuna-Integration

[![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/optuna/optuna-integration)
[![Codecov](https://codecov.io/gh/optuna/optuna-integration/branch/master/graph/badge.svg)](https://codecov.io/gh/optuna/optuna-integration/branch/master)
<!-- [![pypi](https://img.shields.io/pypi/v/optuna.svg)](https://pypi.python.org/pypi/optuna-integration) -->
<!-- [![conda](https://img.shields.io/conda/vn/conda-forge/optuna.svg)](https://anaconda.org/conda-forge/optuna-integration) -->
<!-- [![Read the Docs](https://readthedocs.org/projects/optuna/badge/?version=stable)](https://optuna.readthedocs.io/en/stable/) -->

*Optuna-Integration* is an integration module of [Optuna](https://github.com/optuna/optuna).
This package allows us to use Optuna, an automatic Hyperparameter optimization software framework,
integrated with many useful tools like PyTorch, sklearn, TensorFlow, etc.

## Integrations

Optuna-Integration API reference is [here](https://optuna.readthedocs.io/en/stable/reference/integration.html).

* [Chainer](https://optuna.readthedocs.io/en/stable/reference/integration.html#chainer)  ([example](https://github.com/optuna/optuna-examples/tree/main/chainer/chainer_integration.py))

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

- [GitHub Discussions] for questions.
- [GitHub Issues] for bug reports and feature requests.

[GitHub Discussions]: https://github.com/optuna/optuna-integration/discussions
[GitHub issues]: https://github.com/optuna/optuna-integration/issues

## Contribution

Any contributions to Optuna-Integration are more than welcome!

For general guidelines how to contribute to the project, take a look at [CONTRIBUTING.md](./CONTRIBUTING.md).

## Reference

Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019.
Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD ([arXiv](https://arxiv.org/abs/1907.10902)).
