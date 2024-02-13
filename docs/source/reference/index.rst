API Reference for Optuna-Integration
====================================


The Optuna-Integration package contains classes used to integrate Optuna with external machine learning frameworks.

All of these classes can be imported in two ways. One is "`from optuna.integration import xxx`" like a module in Optuna,
and the other is "`from optuna_integration import xxx`" as an Optuna-Integration specific module.
The former is provided for backward compatibility.

For most of the ML frameworks supported by Optuna, the corresponding Optuna integration class serves only to implement a callback object and functions, compliant with the framework's specific callback API, to be called with each intermediate step in the model training. The functionality implemented in these callbacks across the different ML frameworks includes:

(1) Reporting intermediate model scores back to the Optuna trial using `optuna.trial.Trial.report <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.report>`_,
(2) According to the results of `optuna.trial.Trial.should_prune <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.should_prune>`_, pruning the current model by raising `optuna.TrialPruned <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.TrialPruned.html#optuna.TrialPruned>`_, and
(3) Reporting intermediate Optuna data such as the current trial number back to the framework, as done in :class:`~optuna_integration.MLflowCallback`.

For scikit-learn, an integrated :class:`~optuna_integration.OptunaSearchCV` estimator is available that combines scikit-learn BaseEstimator functionality with access to a class-level ``Study`` object.

AllenNLP
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna_integration.AllenNLPExecutor
   optuna_integration.allennlp.dump_best_config
   optuna_integration.AllenNLPPruningCallback

BoTorch
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna_integration.BoTorchSampler
   optuna_integration.botorch.ehvi_candidates_func
   optuna_integration.botorch.logei_candidates_func
   optuna_integration.botorch.qei_candidates_func
   optuna_integration.botorch.qnei_candidates_func
   optuna_integration.botorch.qehvi_candidates_func
   optuna_integration.botorch.qnehvi_candidates_func
   optuna_integration.botorch.qparego_candidates_func

Catalyst
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna_integration.CatalystPruningCallback

CatBoost
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna_integration.CatBoostPruningCallback

Chainer
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna_integration.ChainerPruningExtension
   optuna_integration.ChainerMNStudy

Dask
----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna_integration.DaskStorage

fast.ai
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna_integration.FastAIV1PruningCallback
   optuna_integration.FastAIV2PruningCallback
   optuna_integration.FastAIPruningCallback

Keras
-----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna_integration.KerasPruningCallback

MLflow
------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna_integration.MLflowCallback

MXNet
-----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna_integration.MXNetPruningCallback

pycma
-----
.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna_integration.CmaEsSampler
   optuna_integration.PyCmaSampler

scikit-optimize
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna_integration.SkoptSampler

SHAP
----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna_integration.ShapleyImportanceEvaluator

sklearn
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    optuna_integration.OptunaSearchCV

skorch
------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    optuna_integration.SkorchPruningCallback

TensorBoard
-----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna_integration.TensorBoardCallback

TensorFlow
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna_integration.TFKerasPruningCallback

Weights & Biases
----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna_integration.WeightsAndBiasesCallback

XGBoost
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna_integration.XGBoostPruningCallback
