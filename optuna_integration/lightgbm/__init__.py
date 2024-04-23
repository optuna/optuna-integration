import sys

from optuna._imports import try_import

from optuna_integration import _lightgbm_tuner as tuner

from .lightgbm import LightGBMPruningCallback


with try_import() as _imports:
    import lightgbm as lgb

# Attach lightgbm API.
if _imports.is_successful():
    # To pass tests/lightgbm_tuner_tests/test_optimize.py.
    from lightgbm import Dataset
    from optuna_integration._lightgbm_tuner import LightGBMTuner
    from optuna_integration._lightgbm_tuner import LightGBMTunerCV

    _names_from_tuners = ["train", "LGBMModel", "LGBMClassifier", "LGBMRegressor"]

    # API from lightgbm.
    for api_name in lgb.__dict__["__all__"]:
        if api_name in _names_from_tuners:
            continue
        setattr(sys.modules[__name__], api_name, lgb.__dict__[api_name])

    # API from lightgbm_tuner.
    for api_name in _names_from_tuners:
        setattr(sys.modules[__name__], api_name, tuner.__dict__[api_name])
else:
    # To create docstring of train.
    setattr(sys.modules[__name__], "train", tuner.__dict__["train"])
    setattr(sys.modules[__name__], "LightGBMTuner", tuner.__dict__["LightGBMTuner"])
    setattr(sys.modules[__name__], "LightGBMTunerCV", tuner.__dict__["LightGBMTunerCV"])

__all__ = [
    "LightGBMTuner",
    "LightGBMTunerCV",
    "LightGBMPruningCallback",
    "Dataset",
]
