import json
from unittest import mock

import pytest
import optuna
from optuna_integration.comet import CometCallback

pytestmark = pytest.mark.integration


def _objective_func(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", 1, 10, log=True)

    params = {
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    }

    return (x - 2) ** 2 + (y - 25) ** 2


def _multiobjective_func(trial: optuna.trial.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", 1, 10, log=True)

    params = {
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    }

    first_objective = (x - 2) ** 2 + (y - 25) ** 2
    second_objective = (x - 2) ** 3 + (y - 25) ** 3
    return first_objective, second_objective

@mock.patch("comet_ml.Experiment")
@mock.patch("comet_ml.APIExperiment")
def test_comet_callback_with_metric_names(api_experiment_mock: mock.MagicMock, experiment_mock: mock.MagicMock) -> None:
    n_trials = 3
    study = optuna.create_study(direction="minimize")
    comet_callback = CometCallback(
        study, 
        project_name="optuna_test_with_metrics",
        workspace="workspace_with_metrics",
        metric_names=["custom_metric"]
    )

    study.optimize(_objective_func, n_trials=n_trials, callbacks=[comet_callback])

    assert api_experiment_mock.call_count == 1
    assert experiment_mock.call_count == n_trials


@mock.patch("comet_ml.Experiment")
@mock.patch("comet_ml.APIExperiment")
def test_comet_callback_initialization(api_experiment_mock: mock.MagicMock, experiment_mock: mock.MagicMock) -> None:
    study = optuna.create_study(direction="minimize")
    CometCallback(study, project_name="optuna_test_init", workspace="workspace_init")

    # First call to APIExperiment upon initialization
    api_experiment_mock.assert_called_once()


@mock.patch("comet_ml.Experiment")
@mock.patch("comet_ml.APIExperiment")
def test_comet_callback_experiment_key_reuse(api_experiment_mock: mock.MagicMock, experiment_mock: mock.MagicMock) -> None:
    study = optuna.create_study(direction="minimize")
    study.set_user_attr("comet_study_experiment_key", "existing_experiment_key")
    comet_callback = CometCallback(study, project_name="optuna_test_reuse", workspace="workspace_reuse")

    # Simulate optimization to check if the existing experiment key is reused
    optuna_trial = optuna.trial.create_trial(
        params={"x": 2.5},
        distributions={"x": optuna.distributions.UniformDistribution(-5, 5)},
        value=1.5
    )
    study.add_trial(optuna_trial)

    api_experiment_mock.assert_called_with(previous_experiment="existing_experiment_key")


@mock.patch("comet_ml.Experiment")
@mock.patch("comet_ml.APIExperiment")
def test_comet_callback_track_in_comet_decorator(api_experiment_mock: mock.MagicMock, experiment_mock: mock.MagicMock) -> None:
    n_trials = 3
    study = optuna.create_study(directions=["minimize", "maximize"])
    comet_callback = CometCallback(
        study,
        project_name="optuna_test_decorator",
        workspace="workspace_decorator",
        metric_names=["first_metric", "second_metric"]
    )

    @comet_callback.track_in_comet()
    def your_objective(trial):
        x = trial.suggest_float("x", -5, 5)
        y = trial.suggest_float("y", -5, 5)
        trial.experiment.log_other("extra_info", "test")
        return (x - 2) ** 2, (y + 2) ** 2

    study.optimize(your_objective, n_trials=n_trials)

    assert experiment_mock.call_count == n_trials
    experiment_mock.return_value.log_other.assert_any_call("extra_info", "test")


@mock.patch("comet_ml.Experiment")
@mock.patch("comet_ml.APIExperiment")
def test_best_trials_logged(api_experiment_mock: mock.MagicMock, experiment_mock: mock.MagicMock) -> None:
    n_trials = 5
    study = optuna.create_study(direction="minimize")
    comet_callback = CometCallback(study, project_name="best_trials_project", workspace="best_trials_workspace")

    study.optimize(_objective_func, n_trials=n_trials, callbacks=[comet_callback])
    
    best_trials = [trial.number for trial in study.best_trials]
    # Check if the logging of the best_trials was performed
    api_experiment_mock.return_value.log_other.assert_called_with("best_trials", json.dumps(best_trials))
