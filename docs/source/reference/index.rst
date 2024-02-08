API Reference for Optuna-Integration
====================================


The Optuna-Integration package contains classes used to integrate Optuna with external machine learning frameworks.

All of these classes can be imported in two ways. One is "`from optuna.integration import xxx`" like a module in Optuna, 
and the other is "`from optuna_integration import xxx`" as an Optuna-Integration specific module. 
The former is provided for backward compatibility.

For most of the ML frameworks supported by Optuna, the corresponding Optuna integration class serves only to implement a callback object and functions, compliant with the framework's specific callback API, to be called with each intermediate step in the model training. The functionality implemented in these callbacks across the different ML frameworks includes:

(1) Reporting intermediate model scores back to the Optuna trial using :func:`optuna.trial.Trial.report`,
(2) According to the results of :func:`optuna.trial.Trial.should_prune`, pruning the current model by raising :func:`optuna.TrialPruned`, and
(3) Reporting intermediate Optuna data such as the current trial number back to the framework, as done in :class:`~optuna.integration.MLflowCallback`.

For scikit-learn, an integrated :class:`~optuna.integration.OptunaSearchCV` estimator is available that combines scikit-learn BaseEstimator functionality with access to a class-level ``Study`` object.

AllenNLP
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.AllenNLPExecutor
   optuna.integration.allennlp.dump_best_config
   optuna.integration.AllenNLPPruningCallback

Catalyst
--------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.CatalystPruningCallback

Chainer
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.ChainerPruningExtension
   optuna.integration.ChainerMNStudy

Keras
-----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.KerasPruningCallback

MXNet
-----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.MXNetPruningCallback

scikit-optimize
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.SkoptSampler

SHAP
----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.ShapleyImportanceEvaluator

skorch
------

.. autosummary::
   :toctree: generated/
   :nosignatures:

    optuna.integration.SkorchPruningCallback

TensorFlow
----------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   optuna.integration.TFKerasPruningCallback