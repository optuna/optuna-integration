Installation
============

Optuna-Integration supports Python 3.7 or newer.

We recommend to install Optuna-Integration via pip:

.. code-block:: bash

    $ pip install optuna-integration

If you would like to install the dependencies of specific integrations, you can install via:

.. code-block:: bash

    # Optuna-Integration with the dependencies for optuna_integration.lightgbm.
    $ pip install optuna-integration[lightgbm]

    # Optuna-Integration with the dependencies for optuna_integration.lightgbm and optuna_integration.botorch.
    $ pip install optuna-integration[botorch,lightgbm]

You can also install the development version of Optuna-Integration from main branch of Git repository:

.. code-block:: bash

    $ pip install git+https://github.com/optuna/optuna-integration.git
