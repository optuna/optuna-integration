from __future__ import annotations

import abc
from collections.abc import Callable
from collections.abc import Container
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
import copy
import json
import os
import pickle
import time
from typing import Any
from typing import cast
from typing import List
from typing import Protocol
import warnings

import numpy as np
import optuna
from optuna._imports import try_import
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import tqdm

from optuna_integration.lightgbm._lightgbm_tuner.alias import _handling_alias_metrics
from optuna_integration.lightgbm._lightgbm_tuner.alias import _handling_alias_parameters


with try_import() as _imports:
    import lightgbm as lgb
    from sklearn.model_selection import BaseCrossValidator


# Define key names of `Trial.system_attrs`.
_ELAPSED_SECS_KEY = "lightgbm_tuner:elapsed_secs"
_AVERAGE_ITERATION_TIME_KEY = "lightgbm_tuner:average_iteration_time"
_STEP_NAME_KEY = "lightgbm_tuner:step_name"
_LGBM_PARAMS_KEY = "lightgbm_tuner:lgbm_params"

# EPS is used to ensure that a sampled parameter value is in pre-defined value range.
_EPS = 1e-12

# Default value of tree_depth, used for upper bound of num_leaves.
_DEFAULT_TUNER_TREE_DEPTH = 8

# Default parameter values described in the official webpage.
_DEFAULT_LIGHTGBM_PARAMETERS = {
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "num_leaves": 31,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "min_child_samples": 20,
}

_logger = optuna.logging.get_logger(__name__)


class _CustomObjectiveType(Protocol):
    def __call__(self, preds: np.ndarray, train: "lgb.Dataset") -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


def _get_custom_objective(lgbm_kwargs: dict[str, Any]) -> _CustomObjectiveType | None:
    objective = lgbm_kwargs.get("objective")
    if objective is not None and not isinstance(objective, str):
        return objective
    else:
        return None


class _BaseTuner:
    def __init__(
        self,
        lgbm_params: dict[str, Any] | None = None,
        lgbm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # Handling alias metrics.
        if lgbm_params is not None:
            _handling_alias_metrics(lgbm_params)

        self.lgbm_params = lgbm_params or {}
        self.lgbm_kwargs = lgbm_kwargs or {}

    def _get_metric_for_objective(self) -> str:
        metric = self.lgbm_params.get("metric", "binary_logloss")

        # todo (smly): This implementation is different logic from the LightGBM's python bindings.
        if isinstance(metric, str):
            pass
        elif isinstance(metric, Sequence):
            metric = metric[-1]
        elif isinstance(metric, Iterable):
            metric = list(metric)[-1]
        else:
            raise NotImplementedError
        metric = self._metric_with_eval_at(metric)

        return metric

    def _get_booster_best_score(self, booster: "lgb.Booster") -> float:
        metric = self._get_metric_for_objective()
        valid_sets: list["lgb.Dataset"] | tuple["lgb.Dataset", ...] | "lgb.Dataset" | None = (
            self.lgbm_kwargs.get("valid_sets")
        )

        if self.lgbm_kwargs.get("valid_names") is not None:
            if isinstance(self.lgbm_kwargs["valid_names"], str):
                valid_name = self.lgbm_kwargs["valid_names"]
            elif isinstance(self.lgbm_kwargs["valid_names"], Sequence):
                valid_name = self.lgbm_kwargs["valid_names"][-1]
            else:
                raise NotImplementedError

        elif isinstance(valid_sets, lgb.Dataset):
            valid_name = "valid_0"

        elif isinstance(valid_sets, Sequence) and len(valid_sets) > 0:
            valid_set_idx = len(valid_sets) - 1
            valid_name = f"valid_{valid_set_idx}"

        else:
            raise NotImplementedError

        val_score = booster.best_score[valid_name][metric]
        return val_score

    def _metric_with_eval_at(self, metric: str) -> str:
        # The parameter eval_at is only available when the metric is ndcg or map
        if metric not in ["ndcg", "map"]:
            return metric

        eval_at = (
            self.lgbm_params.get("eval_at")
            or self.lgbm_params.get(f"{metric}_at")
            or self.lgbm_params.get(f"{metric}_eval_at")
            # Set default value of LightGBM when no possible key is absent.
            # See https://lightgbm.readthedocs.io/en/latest/Parameters.html#eval_at.
            or [1, 2, 3, 4, 5]
        )

        # Optuna can handle only a single metric. Choose first one.
        if isinstance(eval_at, (list, tuple)):
            return f"{metric}@{eval_at[0]}"
        if isinstance(eval_at, int):
            return f"{metric}@{eval_at}"
        raise ValueError(
            f"The value of eval_at is expected to be int or a list/tuple of int. '{eval_at}' is "
            "specified."
        )

    def higher_is_better(self) -> bool:
        metric_name = self.lgbm_params.get("metric", "binary_logloss")
        return metric_name in ("auc", "auc_mu", "ndcg", "map", "average_precision")

    def compare_validation_metrics(self, val_score: float, best_score: float) -> bool:
        if self.higher_is_better():
            return val_score > best_score
        else:
            return val_score < best_score


class _OptunaObjective(_BaseTuner):
    """Objective for hyperparameter-tuning with Optuna."""

    def __init__(
        self,
        target_param_names: list[str],
        lgbm_params: dict[str, Any],
        train_set: "lgb.Dataset",
        lgbm_kwargs: dict[str, Any],
        best_score: float,
        step_name: str,
        model_dir: str | None,
        pbar: tqdm.tqdm | None = None,
    ):
        self.target_param_names = target_param_names
        self.pbar = pbar
        self.lgbm_params = lgbm_params
        self.lgbm_kwargs = lgbm_kwargs
        self.train_set = train_set

        self.trial_count = 0
        self.best_score = best_score
        self.best_booster_with_trial_number: tuple["lgb.Booster" | "lgb.CVBooster", int] | None = (
            None
        )
        self.step_name = step_name
        self.model_dir = model_dir

        self._check_target_names_supported()
        self.pbar_fmt = "{}, val_score: {:.6f}"

    def _check_target_names_supported(self) -> None:
        for target_param_name in self.target_param_names:
            if target_param_name in _DEFAULT_LIGHTGBM_PARAMETERS:
                continue
            raise NotImplementedError(
                f"Parameter `{target_param_name}` is not supported for tuning."
            )

    def _preprocess(self, trial: optuna.trial.Trial) -> None:
        if self.pbar is not None:
            self.pbar.set_description(self.pbar_fmt.format(self.step_name, self.best_score))

        if "lambda_l1" in self.target_param_names:
            self.lgbm_params["lambda_l1"] = trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True)
        if "lambda_l2" in self.target_param_names:
            self.lgbm_params["lambda_l2"] = trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True)
        if "num_leaves" in self.target_param_names:
            tree_depth = self.lgbm_params.get("max_depth", _DEFAULT_TUNER_TREE_DEPTH)
            max_num_leaves = 2**tree_depth if tree_depth > 0 else 2**_DEFAULT_TUNER_TREE_DEPTH
            self.lgbm_params["num_leaves"] = trial.suggest_int("num_leaves", 2, max_num_leaves)
        if "feature_fraction" in self.target_param_names:
            # `GridSampler` is used for sampling feature_fraction value.
            # The value 1.0 for the hyperparameter is always sampled.
            param_value = min(trial.suggest_float("feature_fraction", 0.4, 1.0 + _EPS), 1.0)
            self.lgbm_params["feature_fraction"] = param_value
        if "bagging_fraction" in self.target_param_names:
            # `TPESampler` is used for sampling bagging_fraction value.
            # The value 1.0 for the hyperparameter might by sampled.
            param_value = min(trial.suggest_float("bagging_fraction", 0.4, 1.0 + _EPS), 1.0)
            self.lgbm_params["bagging_fraction"] = param_value
        if "bagging_freq" in self.target_param_names:
            self.lgbm_params["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 7)
        if "min_child_samples" in self.target_param_names:
            # `GridSampler` is used for sampling min_child_samples value.
            # The value 1.0 for the hyperparameter is always sampled.
            param_value = trial.suggest_int("min_child_samples", 5, 100)
            self.lgbm_params["min_child_samples"] = param_value

    def _copy_valid_sets(
        self, valid_sets: list["lgb.Dataset"] | tuple["lgb.Dataset", ...] | "lgb.Dataset"
    ) -> list["lgb.Dataset"] | tuple["lgb.Dataset", ...] | "lgb.Dataset":
        if isinstance(valid_sets, list):
            return [copy.copy(d) for d in valid_sets]
        if isinstance(valid_sets, tuple):
            return tuple([copy.copy(d) for d in valid_sets])
        return copy.copy(valid_sets)

    def __call__(self, trial: optuna.trial.Trial) -> float:
        self._preprocess(trial)

        start_time = time.time()
        train_set = copy.copy(self.train_set)
        kwargs = copy.copy(self.lgbm_kwargs)
        kwargs["valid_sets"] = self._copy_valid_sets(kwargs["valid_sets"])
        booster = lgb.train(self.lgbm_params, train_set, **kwargs)

        val_score = self._get_booster_best_score(booster)
        elapsed_secs = time.time() - start_time
        average_iteration_time = elapsed_secs / booster.current_iteration()

        if self.model_dir is not None:
            path = os.path.join(self.model_dir, f"{trial.number}.pkl")
            with open(path, "wb") as fout:
                pickle.dump(booster, fout)
            _logger.info(f"The booster of trial#{trial.number} was saved as {path}.")

        if self.compare_validation_metrics(val_score, self.best_score):
            self.best_score = val_score
            self.best_booster_with_trial_number = (booster, trial.number)

        self._postprocess(trial, elapsed_secs, average_iteration_time)

        return val_score

    def _postprocess(
        self, trial: optuna.trial.Trial, elapsed_secs: float, average_iteration_time: float
    ) -> None:
        if self.pbar is not None:
            self.pbar.set_description(self.pbar_fmt.format(self.step_name, self.best_score))
            self.pbar.update(1)

        trial.storage.set_trial_system_attr(trial._trial_id, _ELAPSED_SECS_KEY, elapsed_secs)
        trial.storage.set_trial_system_attr(
            trial._trial_id, _AVERAGE_ITERATION_TIME_KEY, average_iteration_time
        )
        trial.storage.set_trial_system_attr(trial._trial_id, _STEP_NAME_KEY, self.step_name)
        lgbm_params = copy.deepcopy(self.lgbm_params)
        custom_objective = _get_custom_objective(lgbm_params)
        if custom_objective is not None:
            # NOTE(nabenabe): If custom_objective is not None, custom_objective is not
            # serializable, so we store its name instead.
            lgbm_params["objective"] = (
                custom_objective.__name__
                if hasattr(custom_objective, "__name__")
                else str(custom_objective)
            )
        trial.storage.set_trial_system_attr(
            trial._trial_id, _LGBM_PARAMS_KEY, json.dumps(lgbm_params)
        )

        self.trial_count += 1


class _OptunaObjectiveCV(_OptunaObjective):
    def __init__(
        self,
        target_param_names: list[str],
        lgbm_params: dict[str, Any],
        train_set: "lgb.Dataset",
        lgbm_kwargs: dict[str, Any],
        best_score: float,
        step_name: str,
        model_dir: str | None,
        pbar: tqdm.tqdm | None = None,
    ):
        super().__init__(
            target_param_names,
            lgbm_params,
            train_set,
            lgbm_kwargs,
            best_score,
            step_name,
            model_dir,
            pbar=pbar,
        )

    def _get_cv_scores(self, cv_results: dict[str, list[float] | "lgb.CVBooster"]) -> list[float]:
        metric = self._get_metric_for_objective()
        metric_key = f"{metric}-mean"
        # The prefix "valid " is added to metric name since LightGBM v4.0.0.
        val_scores = (
            cv_results[metric_key]
            if metric_key in cv_results
            else cv_results["valid " + metric_key]
        )
        assert not isinstance(val_scores, lgb.CVBooster)
        return val_scores

    def __call__(self, trial: optuna.trial.Trial) -> float:
        self._preprocess(trial)

        start_time = time.time()
        train_set = copy.copy(self.train_set)
        cv_results = lgb.cv(self.lgbm_params, train_set, **self.lgbm_kwargs)

        val_scores = self._get_cv_scores(cv_results)
        val_score = val_scores[-1]
        elapsed_secs = time.time() - start_time
        average_iteration_time = elapsed_secs / len(val_scores)

        if self.model_dir is not None and self.lgbm_kwargs.get("return_cvbooster"):
            path = os.path.join(self.model_dir, f"{trial.number}.pkl")
            with open(path, "wb") as fout:
                # At version `lightgbm==3.0.0`, :class:`lightgbm.CVBooster` does not
                # have `__getstate__` which is required for pickle serialization.
                cvbooster = cv_results["cvbooster"]
                assert isinstance(cvbooster, lgb.CVBooster)
                pickle.dump((cvbooster.boosters, cvbooster.best_iteration), fout)
            _logger.info(f"The booster of trial#{trial.number} was saved as {path}.")

        if self.compare_validation_metrics(val_score, self.best_score):
            self.best_score = val_score
            if self.lgbm_kwargs.get("return_cvbooster"):
                assert not isinstance(cv_results["cvbooster"], list)
                self.best_booster_with_trial_number = (cv_results["cvbooster"], trial.number)

        self._postprocess(trial, elapsed_secs, average_iteration_time)

        return val_score


class _LightGBMBaseTuner(_BaseTuner):
    """Base class of LightGBM Tuners.

    This class has common attributes and methods of
    :class:`~optuna_integration.lightgbm.LightGBMTuner` and
    :class:`~optuna_integration.lightgbm.LightGBMTunerCV`.
    """

    def __init__(
        self,
        params: dict[str, Any],
        train_set: "lgb.Dataset",
        callbacks: list[Callable[..., Any]] | None = None,
        num_boost_round: int = 1000,
        feval: Callable[..., Any] | None = None,
        feature_name: str | None = None,
        categorical_feature: str | None = None,
        time_budget: int | None = None,
        sample_size: int | None = None,
        study: optuna.study.Study | None = None,
        optuna_callbacks: list[Callable[[Study, FrozenTrial], None]] | None = None,
        *,
        show_progress_bar: bool = True,
        model_dir: str | None = None,
        optuna_seed: int | None = None,
    ) -> None:
        _imports.check()

        params = copy.deepcopy(params)

        # Handling alias metrics.
        _handling_alias_metrics(params)
        args = [params, train_set]
        kwargs: dict[str, Any] = dict(
            num_boost_round=num_boost_round,
            feval=feval,
            callbacks=callbacks,
            time_budget=time_budget,
            sample_size=sample_size,
            show_progress_bar=show_progress_bar,
        )

        deprecated_arg_warning = (
            "Support for lgb.train and lgb.cv with argument {deprecated_arg} was removed "
            "from lightgbm 4.6.0 and will not be supported by optuna in the future."
        )
        if feature_name:
            warnings.warn(deprecated_arg_warning.format(deprecated_arg="feature_name"))
            kwargs["feature_name"] = feature_name
        if categorical_feature:
            warnings.warn(deprecated_arg_warning.format(deprecated_arg="categorical_feature"))
            kwargs["categorical_feature"] = categorical_feature

        self._parse_args(*args, **kwargs)
        self._start_time: float | None = None
        self._optuna_callbacks = optuna_callbacks
        self._best_booster_with_trial_number: tuple[lgb.Booster | lgb.CVBooster, int] | None = None
        self._model_dir = model_dir
        self._optuna_seed = optuna_seed
        self._custom_objective = _get_custom_objective(params)

        # Should not alter data since `min_child_samples` is tuned.
        # https://lightgbm.readthedocs.io/en/latest/Parameters.html#feature_pre_filter
        if self.lgbm_params.get("feature_pre_filter", False):
            warnings.warn(
                "feature_pre_filter is given as True but will be set to False. This is required "
                "for the tuner to tune min_child_samples."
            )
        self.lgbm_params["feature_pre_filter"] = False

        if study is None:
            self.study = optuna.create_study(
                direction="maximize" if self.higher_is_better() else "minimize"
            )
        else:
            self.study = study

        if self.higher_is_better():
            if self.study.direction != optuna.study.StudyDirection.MAXIMIZE:
                metric_name = self.lgbm_params.get("metric", "binary_logloss")
                raise ValueError(
                    f"Study direction is inconsistent with the metric {metric_name}. "
                    "Please set 'maximize' as the direction."
                )
        else:
            if self.study.direction != optuna.study.StudyDirection.MINIMIZE:
                metric_name = self.lgbm_params.get("metric", "binary_logloss")
                raise ValueError(
                    f"Study direction is inconsistent with the metric {metric_name}. "
                    "Please set 'minimize' as the direction."
                )

        if self._model_dir is not None and not os.path.exists(self._model_dir):
            os.mkdir(self._model_dir)

    @property
    def best_score(self) -> float:
        """Return the score of the best booster."""
        try:
            return self.study.best_value
        except ValueError:
            # Return the default score because no trials have completed.
            return -np.inf if self.higher_is_better() else np.inf

    @property
    def best_params(self) -> dict[str, Any]:
        """Return parameters of the best booster."""
        try:
            params = json.loads(self.study.best_trial.system_attrs[_LGBM_PARAMS_KEY])
        except ValueError:
            # Return the default score because no trials have completed.
            params = copy.deepcopy(_DEFAULT_LIGHTGBM_PARAMETERS)
            # self.lgbm_params may contain parameters given by users.
            params.update(self.lgbm_params)

        if self._custom_objective is not None:
            # NOTE(nabenabe): custom_objective is not serializable, so we store it separately.
            params["objective"] = self._custom_objective

        return params

    def _parse_args(self, *args: Any, **kwargs: Any) -> None:
        self.auto_options = {
            option_name: kwargs.get(option_name)
            for option_name in ["time_budget", "sample_size", "show_progress_bar"]
        }

        # Split options.
        for option_name in self.auto_options.keys():
            if option_name in kwargs:
                del kwargs[option_name]

        self.lgbm_params = args[0]
        self.train_set = args[1]
        self.train_subset = None  # Use for sampling.
        self.lgbm_kwargs = kwargs

    def run(self) -> None:
        """Perform the hyperparameter-tuning with given parameters."""

        # Handling aliases.
        _handling_alias_parameters(self.lgbm_params)

        # Sampling.
        self.sample_train_set()

        self.tune_feature_fraction()
        self.tune_num_leaves()
        self.tune_bagging()
        self.tune_feature_fraction_stage2()
        self.tune_regularization_factors()
        self.tune_min_data_in_leaf()

    def sample_train_set(self) -> None:
        """Make subset of `self.train_set` Dataset object."""

        if self.auto_options["sample_size"] is None:
            return

        self.train_set.construct()
        n_train_instance = self.train_set.get_label().shape[0]
        if n_train_instance > self.auto_options["sample_size"]:
            offset = n_train_instance - self.auto_options["sample_size"]
            idx_list = offset + np.arange(self.auto_options["sample_size"])
            self.train_subset = self.train_set.subset(idx_list)

    def tune_feature_fraction(self, n_trials: int = 7) -> None:
        param_name = "feature_fraction"
        param_values = cast(list, np.linspace(0.4, 1.0, n_trials).tolist())
        sampler = optuna.samplers.GridSampler({param_name: param_values}, seed=self._optuna_seed)
        self._tune_params([param_name], len(param_values), sampler, "feature_fraction")

    def tune_num_leaves(self, n_trials: int = 20) -> None:
        self._tune_params(
            ["num_leaves"],
            n_trials,
            optuna.samplers.TPESampler(seed=self._optuna_seed),
            "num_leaves",
        )

    def tune_bagging(self, n_trials: int = 10) -> None:
        self._tune_params(
            ["bagging_fraction", "bagging_freq"],
            n_trials,
            optuna.samplers.TPESampler(seed=self._optuna_seed),
            "bagging",
        )

    def tune_feature_fraction_stage2(self, n_trials: int = 6) -> None:
        param_name = "feature_fraction"
        best_feature_fraction = self.best_params[param_name]
        param_values: list[float] = cast(
            List[float],
            np.linspace(
                best_feature_fraction - 0.08, best_feature_fraction + 0.08, n_trials
            ).tolist(),
        )
        param_values = [val for val in param_values if val >= 0.4 and val <= 1.0]

        sampler = optuna.samplers.GridSampler({param_name: param_values}, seed=self._optuna_seed)
        self._tune_params([param_name], len(param_values), sampler, "feature_fraction_stage2")

    def tune_regularization_factors(self, n_trials: int = 20) -> None:
        self._tune_params(
            ["lambda_l1", "lambda_l2"],
            n_trials,
            optuna.samplers.TPESampler(seed=self._optuna_seed),
            "regularization_factors",
        )

    def tune_min_data_in_leaf(self) -> None:
        param_name = "min_child_samples"
        param_values = [5, 10, 25, 50, 100]

        sampler = optuna.samplers.GridSampler({param_name: param_values}, seed=self._optuna_seed)
        self._tune_params([param_name], len(param_values), sampler, "min_child_samples")

    def _tune_params(
        self,
        target_param_names: list[str],
        n_trials: int,
        sampler: optuna.samplers.BaseSampler,
        step_name: str,
    ) -> _OptunaObjective:
        pbar = (
            tqdm.tqdm(total=n_trials, ascii=True)
            if self.auto_options["show_progress_bar"]
            else None
        )

        # Set current best parameters.
        self.lgbm_params.update(self.best_params)

        train_set = self.train_set
        if self.train_subset is not None:
            train_set = self.train_subset

        objective = self._create_objective(target_param_names, train_set, step_name, pbar)

        study = self._create_stepwise_study(self.study, step_name)
        study.sampler = sampler

        complete_trials = study.get_trials(
            deepcopy=True,
            states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED),
        )
        _n_trials = n_trials - len(complete_trials)

        if self._start_time is None:
            self._start_time = time.time()

        if self.auto_options["time_budget"] is not None:
            _timeout = self.auto_options["time_budget"] - (time.time() - self._start_time)
        else:
            _timeout = None
        if _n_trials > 0:
            study.optimize(
                objective,
                n_trials=_n_trials,
                timeout=_timeout,
                catch=(),
                callbacks=self._optuna_callbacks,
            )

        if pbar:
            pbar.close()
            del pbar

        if objective.best_booster_with_trial_number is not None:
            self._best_booster_with_trial_number = objective.best_booster_with_trial_number

        return objective

    @abc.abstractmethod
    def _create_objective(
        self,
        target_param_names: list[str],
        train_set: "lgb.Dataset",
        step_name: str,
        pbar: tqdm.tqdm | None,
    ) -> _OptunaObjective:
        raise NotImplementedError

    def _create_stepwise_study(
        self, study: optuna.study.Study, step_name: str
    ) -> optuna.study.Study:
        # This class is assumed to be passed to a sampler and a pruner corresponding to the step.
        class _StepwiseStudy(optuna.study.Study):
            def __init__(self, study: optuna.study.Study, step_name: str) -> None:
                super().__init__(
                    study_name=study.study_name,
                    storage=study._storage,
                    sampler=study.sampler,
                    pruner=study.pruner,
                )
                self._step_name = step_name

            def get_trials(
                self,
                deepcopy: bool = True,
                states: Container[TrialState] | None = None,
            ) -> list[optuna.trial.FrozenTrial]:
                trials = super()._get_trials(deepcopy=deepcopy, states=states)
                return [t for t in trials if t.system_attrs.get(_STEP_NAME_KEY) == self._step_name]

            @property
            def best_trial(self) -> optuna.trial.FrozenTrial:
                """Return the best trial in the study.

                Returns:
                    A :class:`~optuna.trial.FrozenTrial` object of the best trial.
                """

                trials = self.get_trials(deepcopy=False)
                trials = [t for t in trials if t.state is optuna.trial.TrialState.COMPLETE]

                if len(trials) == 0:
                    raise ValueError("No trials are completed yet.")

                if self.direction == optuna.study.StudyDirection.MINIMIZE:
                    best_trial = min(trials, key=lambda t: cast(float, t.value))
                else:
                    best_trial = max(trials, key=lambda t: cast(float, t.value))
                return copy.deepcopy(best_trial)

        return _StepwiseStudy(study, step_name)


class LightGBMTuner(_LightGBMBaseTuner):
    """Hyperparameter tuner for LightGBM.

    It optimizes the following hyperparameters in a stepwise manner:
    ``lambda_l1``, ``lambda_l2``, ``num_leaves``, ``feature_fraction``, ``bagging_fraction``,
    ``bagging_freq`` and ``min_child_samples``.

    You can find the details of the algorithm and benchmark results in `this blog article <https:/
    /medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b709
    5e99258>`_ by `Kohei Ozaki <https://www.kaggle.com/confirm>`_, a Kaggle Grandmaster.

    .. note::
        Arguments and keyword arguments for `lightgbm.train()
        <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html>`_ can be passed.
        For ``params``, please check `the official documentation for LightGBM
        <https://lightgbm.readthedocs.io/en/latest/Parameters.html>`_.

    .. warning::
        Arguments ``feature_name`` and ``categorical_feature`` were deprecated in v4.2.2 and
        will be removed in the future. The removal of these arguments is currently scheduled
        for v6.0.0, but this schedule is subject to change.
        See https://github.com/optuna/optuna-integration/releases/tag/v4.2.2.

    The arguments that only :class:`~optuna_integration.lightgbm.LightGBMTuner` has are
    listed below:

    Args:
        time_budget:
            A time budget for parameter tuning in seconds.

        study:
            A :class:`~optuna.study.Study` instance to store optimization results. The
            :class:`~optuna.trial.Trial` instances in it has the following user attributes:
            ``elapsed_secs`` is the elapsed time since the optimization starts.
            ``average_iteration_time`` is the average time of iteration to train the booster
            model in the trial. ``lgbm_params`` is a JSON-serialized dictionary of LightGBM
            parameters used in the trial.

        optuna_callbacks:
            List of Optuna callback functions that are invoked at the end of each trial.
            Each function must accept two parameters with the following types in this order:
            :class:`~optuna.study.Study` and :class:`~optuna.trial.FrozenTrial`.
            Please note that this is not a ``callbacks`` argument of `lightgbm.train()`_ .

        model_dir:
            A directory to save boosters. By default, it is set to :obj:`None` and no boosters are
            saved. Please set shared directory (e.g., directories on NFS) if you want to access
            :meth:`~optuna_integration.lightgbm.LightGBMTuner.get_best_booster` in distributed
            environments. Otherwise, it may raise :obj:`ValueError`. If the directory does not
            exist, it will be created. The filenames of the boosters will be
            ``{model_dir}/{trial_number}.pkl`` (e.g., ``./boosters/0.pkl``).

        show_progress_bar:
            Flag to show progress bars or not. To disable progress bar, set this :obj:`False`.

            .. note::
                Progress bars will be fragmented by logging messages of LightGBM and Optuna.
                Please suppress such messages to show the progress bars properly.

        optuna_seed:
            ``seed`` of :class:`~optuna.samplers.TPESampler` for random number generator
            that affects sampling for ``num_leaves``, ``bagging_fraction``, ``bagging_freq``,
            ``lambda_l1``, and ``lambda_l2``.

            .. note::
                The `deterministic`_ parameter of LightGBM makes training reproducible.
                Please enable it when you use this argument.

    .. _lightgbm.train(): https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
    .. _LightGBM's verbosity: https://lightgbm.readthedocs.io/en/latest/Parameters.html#verbosity
    .. _deterministic: https://lightgbm.readthedocs.io/en/latest/Parameters.html#deterministic
    """

    def __init__(
        self,
        params: dict[str, Any],
        train_set: "lgb.Dataset",
        num_boost_round: int = 1000,
        valid_sets: list["lgb.Dataset"] | tuple["lgb.Dataset", ...] | "lgb.Dataset" | None = None,
        valid_names: Any | None = None,
        feval: Callable[..., Any] | None = None,
        feature_name: str | None = None,
        categorical_feature: str | None = None,
        keep_training_booster: bool = False,
        callbacks: list[Callable[..., Any]] | None = None,
        time_budget: int | None = None,
        sample_size: int | None = None,
        study: optuna.study.Study | None = None,
        optuna_callbacks: list[Callable[[Study, FrozenTrial], None]] | None = None,
        model_dir: str | None = None,
        *,
        show_progress_bar: bool = True,
        optuna_seed: int | None = None,
    ) -> None:
        super().__init__(
            params,
            train_set,
            callbacks=callbacks,
            num_boost_round=num_boost_round,
            feval=feval,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            time_budget=time_budget,
            sample_size=sample_size,
            study=study,
            optuna_callbacks=optuna_callbacks,
            show_progress_bar=show_progress_bar,
            model_dir=model_dir,
            optuna_seed=optuna_seed,
        )

        self.lgbm_kwargs["valid_sets"] = valid_sets
        self.lgbm_kwargs["valid_names"] = valid_names
        self.lgbm_kwargs["keep_training_booster"] = keep_training_booster

        self._best_booster_with_trial_number: tuple[lgb.Booster, int] | None = None

        if valid_sets is None:
            raise ValueError("`valid_sets` is required.")

    def _create_objective(
        self,
        target_param_names: list[str],
        train_set: "lgb.Dataset",
        step_name: str,
        pbar: tqdm.tqdm | None,
    ) -> _OptunaObjective:
        return _OptunaObjective(
            target_param_names,
            self.lgbm_params,
            train_set,
            self.lgbm_kwargs,
            self.best_score,
            step_name=step_name,
            model_dir=self._model_dir,
            pbar=pbar,
        )

    def get_best_booster(self) -> "lgb.Booster":
        """Return the best booster.

        If the best booster cannot be found, :class:`ValueError` will be raised. To prevent the
        errors, please save boosters by specifying the ``model_dir`` argument of
        :meth:`~optuna_integration.lightgbm.LightGBMTuner.__init__`,
        when you resume tuning or you run tuning in parallel.
        """
        if self._best_booster_with_trial_number is not None:
            if self._best_booster_with_trial_number[1] == self.study.best_trial.number:
                return self._best_booster_with_trial_number[0]
        if len(self.study.trials) == 0:
            raise ValueError("The best booster is not available because no trials completed.")

        # The best booster exists, but this instance does not have it.
        # This may be due to resuming or parallelization.
        if self._model_dir is None:
            raise ValueError(
                "The best booster cannot be found. It may be found in the other processes due to "
                "resuming or distributed computing. Please set the `model_dir` argument of "
                "`LightGBMTuner.__init__` and make sure that boosters are shared with all "
                "processes."
            )

        best_trial = self.study.best_trial
        path = os.path.join(self._model_dir, f"{best_trial.number}.pkl")
        if not os.path.exists(path):
            raise ValueError(
                f"The best booster cannot be found in {self._model_dir}. If you execute "
                "`LightGBMTuner` in distributed environment, please use network file system "
                "(e.g., NFS) to share models with multiple workers."
            )

        with open(path, "rb") as fin:
            booster = pickle.load(fin)

        return booster


class LightGBMTunerCV(_LightGBMBaseTuner):
    """Hyperparameter tuner for LightGBM with cross-validation.

    It employs the same stepwise approach as
    :class:`~optuna_integration.lightgbm.LightGBMTuner`.
    :class:`~optuna_integration.lightgbm.LightGBMTunerCV` invokes `lightgbm.cv()`_ to train
    and validate boosters while :class:`~optuna_integration.lightgbm.LightGBMTuner` invokes
    `lightgbm.train()`_. See
    `a simple example <https://github.com/optuna/optuna-examples/tree/main/lightgbm/
    lightgbm_tuner_cv.py>`_ which optimizes the validation log loss of cancer detection.

    .. note::
        Arguments and keyword arguments for `lightgbm.cv()`_ can be passed except
        ``metrics``, ``init_model`` and ``eval_train_metric``.
        For ``params``, please check `the official documentation for LightGBM
        <https://lightgbm.readthedocs.io/en/latest/Parameters.html>`_.

    .. warning::
        Arguments ``feature_name`` and ``categorical_feature`` were deprecated in v4.2.2 and
        will be removed in the future. The removal of these arguments is currently scheduled
        for v6.0.0, but this schedule is subject to change.
        See https://github.com/optuna/optuna-integration/releases/tag/v4.2.2.

    The arguments that only :class:`~optuna_integration.lightgbm.LightGBMTunerCV` has are
    listed below:

    Args:
        time_budget:
            A time budget for parameter tuning in seconds.

        study:
            A :class:`~optuna.study.Study` instance to store optimization results. The
            :class:`~optuna.trial.Trial` instances in it has the following user attributes:
            ``elapsed_secs`` is the elapsed time since the optimization starts.
            ``average_iteration_time`` is the average time of iteration to train the booster
            model in the trial. ``lgbm_params`` is a JSON-serialized dictionary of LightGBM
            parameters used in the trial.

        optuna_callbacks:
            List of Optuna callback functions that are invoked at the end of each trial.
            Each function must accept two parameters with the following types in this order:
            :class:`~optuna.study.Study` and :class:`~optuna.trial.FrozenTrial`.
            Please note that this is not a ``callbacks`` argument of `lightgbm.train()`_ .

        model_dir:
            A directory to save boosters. By default, it is set to :obj:`None` and no boosters are
            saved. Please set shared directory (e.g., directories on NFS) if you want to access
            :meth:`~optuna_integration.lightgbm.LightGBMTunerCV.get_best_booster`
            in distributed environments.
            Otherwise, it may raise :obj:`ValueError`. If the directory does not exist, it will be
            created. The filenames of the boosters will be ``{model_dir}/{trial_number}.pkl``
            (e.g., ``./boosters/0.pkl``).

        show_progress_bar:
            Flag to show progress bars or not. To disable progress bar, set this :obj:`False`.

            .. note::
                Progress bars will be fragmented by logging messages of LightGBM and Optuna.
                Please suppress such messages to show the progress bars properly.

        return_cvbooster:
            Flag to enable :meth:`~optuna_integration.lightgbm.LightGBMTunerCV.get_best_booster`.

        optuna_seed:
            ``seed`` of :class:`~optuna.samplers.TPESampler` for random number generator
            that affects sampling for ``num_leaves``, ``bagging_fraction``, ``bagging_freq``,
            ``lambda_l1``, and ``lambda_l2``.

            .. note::
                The `deterministic`_ parameter of LightGBM makes training reproducible.
                Please enable it when you use this argument.

    .. _lightgbm.train(): https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
    .. _lightgbm.cv(): https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.cv.html
    .. _LightGBM's verbosity: https://lightgbm.readthedocs.io/en/latest/Parameters.html#verbosity
    .. _deterministic: https://lightgbm.readthedocs.io/en/latest/Parameters.html#deterministic
    """

    def __init__(
        self,
        params: dict[str, Any],
        train_set: "lgb.Dataset",
        num_boost_round: int = 1000,
        folds: (
            Generator[tuple[int, int], None, None]
            | Iterator[tuple[int, int]]
            | "BaseCrossValidator"
            | None
        ) = None,
        nfold: int = 5,
        stratified: bool = True,
        shuffle: bool = True,
        feval: Callable[..., Any] | None = None,
        feature_name: str | None = None,
        categorical_feature: str | None = None,
        fpreproc: Callable[..., Any] | None = None,
        seed: int = 0,
        callbacks: list[Callable[..., Any]] | None = None,
        time_budget: int | None = None,
        sample_size: int | None = None,
        study: optuna.study.Study | None = None,
        optuna_callbacks: list[Callable[[Study, FrozenTrial], None]] | None = None,
        *,
        show_progress_bar: bool = True,
        model_dir: str | None = None,
        return_cvbooster: bool = False,
        optuna_seed: int | None = None,
    ) -> None:
        super().__init__(
            params,
            train_set,
            callbacks=callbacks,
            num_boost_round=num_boost_round,
            feval=feval,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            time_budget=time_budget,
            sample_size=sample_size,
            study=study,
            optuna_callbacks=optuna_callbacks,
            show_progress_bar=show_progress_bar,
            model_dir=model_dir,
            optuna_seed=optuna_seed,
        )

        self.lgbm_kwargs["folds"] = folds
        self.lgbm_kwargs["nfold"] = nfold
        self.lgbm_kwargs["stratified"] = stratified
        self.lgbm_kwargs["shuffle"] = shuffle
        self.lgbm_kwargs["seed"] = seed
        self.lgbm_kwargs["fpreproc"] = fpreproc
        self.lgbm_kwargs["return_cvbooster"] = return_cvbooster

    def _create_objective(
        self,
        target_param_names: list[str],
        train_set: "lgb.Dataset",
        step_name: str,
        pbar: tqdm.tqdm | None,
    ) -> _OptunaObjective:
        return _OptunaObjectiveCV(
            target_param_names,
            self.lgbm_params,
            train_set,
            self.lgbm_kwargs,
            self.best_score,
            step_name=step_name,
            model_dir=self._model_dir,
            pbar=pbar,
        )

    def get_best_booster(self) -> "lgb.CVBooster":
        """Return the best cvbooster.

        If the best booster cannot be found, :class:`ValueError` will be raised.
        To prevent the errors, please save boosters by specifying
        both of the ``model_dir`` and the ``return_cvbooster`` arguments of
        :meth:`~optuna_integration.lightgbm.LightGBMTunerCV.__init__`,
        when you resume tuning or you run tuning in parallel.
        """
        if self.lgbm_kwargs.get("return_cvbooster") is not True:
            raise ValueError(
                "LightGBMTunerCV requires `return_cvbooster=True` for method `get_best_booster()`."
            )
        if self._best_booster_with_trial_number is not None:
            if self._best_booster_with_trial_number[1] == self.study.best_trial.number:
                assert isinstance(self._best_booster_with_trial_number[0], lgb.CVBooster)
                return self._best_booster_with_trial_number[0]
        if len(self.study.trials) == 0:
            raise ValueError("The best booster is not available because no trials completed.")

        # The best booster exists, but this instance does not have it.
        # This may be due to resuming or parallelization.
        if self._model_dir is None:
            raise ValueError(
                "The best booster cannot be found. It may be found in the other processes due to "
                "resuming or distributed computing. Please set the `model_dir` argument of "
                "`LightGBMTunerCV.__init__` and make sure that boosters are shared with all "
                "processes."
            )

        best_trial = self.study.best_trial
        path = os.path.join(self._model_dir, f"{best_trial.number}.pkl")
        if not os.path.exists(path):
            raise ValueError(
                f"The best booster cannot be found in {self._model_dir}. If you execute "
                "`LightGBMTunerCV` in distributed environment, please use network file system "
                "(e.g., NFS) to share models with multiple workers."
            )

        with open(path, "rb") as fin:
            boosters, best_iteration = pickle.load(fin)
            # At version `lightgbm==3.0.0`, :class:`lightgbm.CVBooster` does not
            # have `__getstate__` which is required for pickle serialization.
            cvbooster = lgb.CVBooster()
            cvbooster.boosters = boosters
            cvbooster.best_iteration = best_iteration

        return cvbooster
