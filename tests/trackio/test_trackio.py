from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn
from unittest import mock

import optuna
import pytest

from optuna_integration.trackio import TrackioCallback


def _objective_func(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", low=-10, high=10)
    y = trial.suggest_float("y", low=1, high=10, log=True)
    return (x - 2) ** 2 + (y - 25) ** 2


def _multiobjective_func(trial: optuna.trial.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", low=-10, high=10)
    y = trial.suggest_float("y", low=1, high=10, log=True)
    return (x - 2) ** 2, (y - 25) ** 2


@mock.patch("optuna_integration.trackio.trackio")
def test_run_initialized_single_run(trackio: mock.MagicMock) -> None:
    study = optuna.create_study()
    callback = TrackioCallback(project="optuna-test", as_multirun=False)

    study.optimize(_objective_func, n_trials=3, callbacks=[callback])

    assert trackio.init.call_count >= 1
    trackio.log.assert_called()


@mock.patch("optuna_integration.trackio.trackio")
def test_run_initialized_multirun(trackio: mock.MagicMock) -> None:
    callback = TrackioCallback(project="optuna-test", as_multirun=True)

    @callback.track_in_trackio()
    def objective(trial):
        return _objective_func(trial)

    study = optuna.create_study()
    study.optimize(objective, n_trials=3, callbacks=[callback])

    assert trackio.init.call_count == 3
    assert trackio.finish.call_count == 3


@mock.patch("optuna_integration.trackio.trackio")
def test_log_api_call_count(trackio: mock.MagicMock) -> None:
    callback = TrackioCallback(project="optuna-test", as_multirun=True)

    @callback.track_in_trackio()
    def objective(trial):
        result = _objective_func(trial)
        trackio.log({"result": result})
        return result

    study = optuna.create_study()
    study.optimize(objective, n_trials=5, callbacks=[callback])

    # One log from objective + one from callback per trial
    assert trackio.log.call_count == 10


@pytest.mark.parametrize(
    "metric,as_multirun,expected",
    [
        ("value", False, ["x", "y", "value"]),
        ("foo", True, ["x", "y", "foo", "trial_number"]),
    ],
)
@mock.patch("optuna_integration.trackio.trackio")
def test_values_registered_on_epoch(
    trackio: mock.MagicMock,
    metric: str,
    as_multirun: bool,
    expected: list[str],
) -> None:
    callback = TrackioCallback(
        project="optuna-test",
        metric_name=metric,
        as_multirun=as_multirun,
    )

    if as_multirun:

        @callback.track_in_trackio()
        def objective(trial):
            return _objective_func(trial)

    else:
        objective = _objective_func

    study = optuna.create_study()
    study.optimize(objective, n_trials=1, callbacks=[callback])

    logged = trackio.log.call_args[0][0]
    assert list(logged.keys()) == expected


@pytest.mark.parametrize(
    "metrics,as_multirun,expected",
    [
        ("value", False, ["x", "y", "value_0", "value_1"]),
        (("foo", "bar"), True, ["x", "y", "foo", "bar", "trial_number"]),
    ],
)
@mock.patch("optuna_integration.trackio.trackio")
def test_multiobjective_values_registered(
    trackio: mock.MagicMock,
    metrics: str | Sequence[str],
    as_multirun: bool,
    expected: list[str],
) -> None:
    callback = TrackioCallback(
        project="optuna-test",
        metric_name=metrics,
        as_multirun=as_multirun,
    )

    if as_multirun:

        @callback.track_in_trackio()
        def objective(trial):
            return _multiobjective_func(trial)

    else:
        objective = _multiobjective_func

    study = optuna.create_study(directions=["minimize", "maximize"])
    study.optimize(objective, n_trials=1, callbacks=[callback])

    logged = trackio.log.call_args[0][0]
    assert list(logged.keys()) == expected


@mock.patch("optuna_integration.trackio.trackio")
def test_values_registered_with_logging(trackio: mock.MagicMock) -> None:
    callback = TrackioCallback(metric_name="foo", as_multirun=True)

    @callback.track_in_trackio()
    def objective(trial):
        result = _objective_func(trial)
        trackio.log({"result": result})
        return result

    study = optuna.create_study()
    study.enqueue_trial({"x": 2, "y": 3})
    study.optimize(objective, n_trials=1, callbacks=[callback])

    assert trackio.log.call_count == 2
    logged_decorator = trackio.log.mock_calls[0][1][0]
    logged_callback = trackio.log.mock_calls[1][1][0]

    assert list(logged_decorator.keys()) == ["result"]
    assert list(logged_callback.keys()) == ["x", "y", "foo", "trial_number"]


@mock.patch("optuna_integration.trackio.trackio")
def test_multiobjective_raises_on_name_mismatch(trackio: mock.MagicMock) -> None:
    callback = TrackioCallback(metric_name=["foo"])

    study = optuna.create_study(directions=["minimize", "maximize"])

    with pytest.raises(ValueError):
        study.optimize(_multiobjective_func, n_trials=1, callbacks=[callback])


@pytest.mark.parametrize("exception", [optuna.exceptions.TrialPruned, ValueError])
@mock.patch("optuna_integration.trackio.trackio")
def test_none_values(trackio: mock.MagicMock, exception: type[Exception]) -> None:
    def failing_objective(trial: optuna.trial.Trial) -> NoReturn:
        trial.suggest_float("x", -10, 10)
        raise exception()

    callback = TrackioCallback(project="optuna-test")
    study = optuna.create_study()

    study.optimize(
        failing_objective,
        n_trials=1,
        callbacks=[callback],
        catch=(ValueError,),
    )

    logged_keys = list(trackio.log.call_args[0][0].keys())
    assert "value" not in logged_keys
    assert "x" in logged_keys
