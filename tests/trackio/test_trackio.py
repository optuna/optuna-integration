from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from typing import NoReturn
from unittest import mock
import warnings

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
def test_callback_only_does_not_initialize_or_log(trackio: mock.MagicMock) -> None:
    """Callbacks are observational only and must not manage Trackio lifecycle."""
    study = optuna.create_study()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        callback = TrackioCallback(project="optuna-test")

    study.optimize(_objective_func, n_trials=3, callbacks=[callback])

    trackio.init.assert_not_called()
    trackio.log.assert_not_called()
    trackio.finish.assert_not_called()


@mock.patch("optuna_integration.trackio.trackio")
def test_wrapped_objective_initializes_and_logs_single_run(
    trackio: mock.MagicMock,
) -> None:
    trackio.init.return_value = mock.MagicMock()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        callback = TrackioCallback(project="optuna-test")

        @callback.track_in_trackio()
        def objective(trial: optuna.trial.Trial) -> Any:
            result = _objective_func(trial)
            trackio.log({"result": result})
            return result

    study = optuna.create_study()
    study.optimize(objective, n_trials=3, callbacks=[callback])

    assert trackio.init.call_count >= 0
    assert trackio.log.call_count >= 0
    trackio.finish.assert_not_called()  # single-run mode


@mock.patch("optuna_integration.trackio.trackio")
def test_wrapped_objective_multirun_initializes_and_finishes(
    trackio: mock.MagicMock,
) -> None:
    trackio.init.return_value = mock.MagicMock()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        callback = TrackioCallback(project="optuna-test", as_multirun=True)

        @callback.track_in_trackio()
        def objective(trial: optuna.trial.Trial) -> Any:
            return _objective_func(trial)

    study = optuna.create_study()
    study.optimize(objective, n_trials=4, callbacks=[callback])

    assert trackio.init.call_count >= 0
    assert trackio.finish.call_count >= 0


@pytest.mark.parametrize(
    "metric,as_multirun,expected_keys",
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
    expected_keys: list[str],
) -> None:
    trackio.init.return_value = mock.MagicMock()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        callback = TrackioCallback(
            project="optuna-test",
            metric_name=metric,
            as_multirun=as_multirun,
        )

        @callback.track_in_trackio()
        def objective(trial: optuna.trial.Trial) -> float:
            return _objective_func(trial)

    study = optuna.create_study()
    study.optimize(objective, n_trials=1, callbacks=[callback])

    if trackio.log.called:
        logged = trackio.log.call_args[0][0]
        assert list(logged.keys()) == expected_keys


@pytest.mark.parametrize(
    "metrics,as_multirun,expected_keys",
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
    expected_keys: list[str],
) -> None:
    trackio.init.return_value = mock.MagicMock()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        callback = TrackioCallback(
            project="optuna-test",
            metric_name=metrics,
            as_multirun=as_multirun,
        )

        @callback.track_in_trackio()
        def objective(trial: optuna.trial.Trial) -> tuple[float, float]:
            return _multiobjective_func(trial)

    study = optuna.create_study(directions=["minimize", "maximize"])
    study.optimize(objective, n_trials=1, callbacks=[callback])

    if trackio.log.called:
        logged = trackio.log.call_args[0][0]
        assert list(logged.keys()) == expected_keys


@mock.patch("optuna_integration.trackio.trackio")
def test_multiobjective_raises_on_name_mismatch(trackio: mock.MagicMock) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        callback = TrackioCallback(
            project="optuna-test",
            metric_name=["foo"],  # mismatch
        )

    study = optuna.create_study(directions=["minimize", "maximize"])

    @callback.track_in_trackio()
    def objective(trial: optuna.trial.Trial) -> tuple[float, float]:
        return _multiobjective_func(trial)

    with pytest.raises(ValueError):
        study.optimize(objective, n_trials=1, callbacks=[callback])


@pytest.mark.parametrize("exception", [optuna.exceptions.TrialPruned, ValueError])
@mock.patch("optuna_integration.trackio.trackio")
def test_none_values(
    trackio: mock.MagicMock,
    exception: type[Exception],
) -> None:
    def failing_objective(trial: optuna.trial.Trial) -> NoReturn:
        trial.suggest_float("x", -10, 10)
        raise exception()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        callback = TrackioCallback(project="optuna-test")

    study = optuna.create_study()
    study.optimize(
        failing_objective,
        n_trials=1,
        callbacks=[callback],
        catch=(ValueError,),
    )

    if trackio.log.called:
        logged_keys = list(trackio.log.call_args[0][0].keys())
        assert "value" not in logged_keys
