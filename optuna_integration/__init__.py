import os
import sys
from types import ModuleType
from typing import Any
from typing import TYPE_CHECKING


_import_structure = {
    "allennlp": ["AllenNLPExecutor", "AllenNLPPruningCallback"],
    "botorch": ["BoTorchSampler"],
    "catalyst": ["CatalystPruningCallback"],
    "catboost": ["CatBoostPruningCallback"],
    "chainer": ["ChainerPruningExtension"],
    "chainermn": ["ChainerMNStudy"],
    "dask": ["DaskStorage"],
    "fastaiv1": ["FastAIV1PruningCallback"],
    "fastaiv2": ["FastAIV2PruningCallback", "FastAIPruningCallback"],
    "keras": ["KerasPruningCallback"],
    "mxnet": ["MXNetPruningCallback"],
    "shap": ["ShapleyImportanceEvaluator"],
    "sklearn": ["OptunaSearchCV"],
    "skorch": ["SkorchPruningCallback"],
    "tensorboard": ["TensorBoardCallback"],
    "tensorflow": ["TensorFlowPruningHook"],
    "tfkeras": ["TFKerasPruningCallback"],
}


if TYPE_CHECKING:
    from optuna_integration.allennlp import AllenNLPExecutor
    from optuna_integration.allennlp import AllenNLPPruningCallback
    from optuna_integration.botorch import BoTorchSampler
    from optuna_integration.catalyst import CatalystPruningCallback
    from optuna_integration.catboost import CatBoostPruningCallback
    from optuna_integration.chainer import ChainerPruningExtension
    from optuna_integration.chainermn import ChainerMNStudy
    from optuna_integration.dask import DaskStorage
    from optuna_integration.fastaiv1 import FastAIV1PruningCallback
    from optuna_integration.fastaiv2 import FastAIPruningCallback
    from optuna_integration.fastaiv2 import FastAIV2PruningCallback
    from optuna_integration.keras import KerasPruningCallback
    from optuna_integration.mxnet import MXNetPruningCallback
    from optuna_integration.shap import ShapleyImportanceEvaluator
    from optuna_integration.sklearn import OptunaSearchCV
    from optuna_integration.skorch import SkorchPruningCallback
    from optuna_integration.tensorboard import TensorBoardCallback
    from optuna_integration.tensorflow import TensorFlowPruningHook
    from optuna_integration.tfkeras import TFKerasPruningCallback
else:

    class _IntegrationModule(ModuleType):
        """Module class that implements `optuna_integration` package.

        This class applies lazy import under `optuna_integration`, where submodules are imported
        when they are actually accessed. Otherwise, `import optuna` becomes much slower because it
        imports all submodules and their dependencies (e.g., chainer, keras, lightgbm) all at once.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        _modules = set(_import_structure.keys())
        _class_to_module = {}
        for key, values in _import_structure.items():
            for value in values:
                _class_to_module[value] = key

        def __getattr__(self, name: str) -> Any:
            if name in self._modules:
                value = self._get_module(name)
            elif name in self._class_to_module.keys():
                module = self._get_module(self._class_to_module[name])
                value = getattr(module, name)
            else:
                raise AttributeError("module {} has no attribute {}".format(self.__name__, name))

            setattr(self, name, value)
            return value

        def _get_module(self, module_name: str) -> ModuleType:
            import importlib

            try:
                return importlib.import_module("." + module_name, self.__name__)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Optuna's integration modules for third-party libraries have started "
                    "migrating from Optuna itself to a package called `optuna-integration`. "
                    "The module you are trying to use has already been migrated to "
                    "`optuna-integration`. Please install the package by running "
                    "`pip install optuna-integration`."
                )

    sys.modules[__name__] = _IntegrationModule(__name__)

__all__ = [
    "AllenNLPExecutor",
    "AllenNLPPruningCallback",
    "BoTorchSampler",
    "CatalystPruningCallback",
    "CatBoostPruningCallback",
    "ChainerMNStudy",
    "ChainerPruningExtension",
    "DaskStorage",
    "FastAIPruningCallback",
    "FastAIV1PruningCallback",
    "FastAIV2PruningCallback",
    "KerasPruningCallback",
    "MXNetPruningCallback",
    "OptunaSearchCV",
    "ShapleyImportanceEvaluator",
    "SkorchPruningCallback",
    "TensorBoardCallback",
    "TensorFlowPruningHook",
    "TFKerasPruningCallback",
]
