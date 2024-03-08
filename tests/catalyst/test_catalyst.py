import optuna
import pytest

from optuna_integration.catalyst import CatalystPruningCallback


def test_warning() -> None:
    study = optuna.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    with pytest.warns(FutureWarning):
        CatalystPruningCallback(
            trial=trial,
            loader_key="abc",
            metric_key="def",
            minimize=False,
        )
