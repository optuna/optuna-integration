from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import functools
from typing import Any
from typing import TYPE_CHECKING

import optuna
from optuna._experimental import experimental_class
from optuna._experimental import experimental_func
from optuna._imports import try_import


with try_import() as _imports:
    import trackio


if TYPE_CHECKING:
    from optuna.study.study import ObjectiveFuncType


@experimental_class("0.1.0")
class TrackioCallback:
    """
    Callback to track Optuna trials with Trackio.

    This callback enables tracking of an Optuna study using Trackio.
    By default, the entire study is recorded as a single experiment run,
    where all suggested hyperparameters and optimized metrics are logged
    and visualized as a function of optimizer steps.

    Trackio is offline-first and does not require authentication for local
    usage. Optionally, results can be synchronized to Hugging Face Spaces
    or exported as Hugging Face Datasets for sharing, visualization, and
    reproducibility.

    .. note::
        Trackio does not require users to be logged in for local experiment
        tracking. Authentication is only required when synchronizing results
        to Hugging Face Hub (e.g., Spaces or Datasets).

    .. note::
        Unlike Weights & Biases, Trackio does not rely on global mutable state.
        Each run is explicitly initialized and finalized, which makes this
        callback safe to use in long-running processes and research pipelines.

    .. note::
        To ensure deterministic trial ordering in logged metrics, this
        callback should only be used with ``study.optimize(n_jobs=1)``.
        Parallel optimization may result in out-of-order steps.

    Example:

        Add Trackio callback to Optuna optimization.

        .. code::

            import optuna
            from optuna_integration.trackio import TrackioCallback


            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return (x - 2) ** 2


            study = optuna.create_study()

            trackioc = TrackioCallback(project="my-optuna-study")

           study.optimize(objective, n_trials=10, callbacks=[trackioc])


        Trackio logging in multi-run (one run per trial) mode.

        .. code::

            import optuna
            from optuna_integration.trackio import TrackioCallback


            trackioc = TrackioCallback(
                project="my-optuna-study",
                as_multirun=True,
            )


            @trackioc.track_in_trackio()
            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return (x - 2) ** 2


            study = optuna.create_study()
            study.optimize(objective, n_trials=10, callbacks=[trackioc])


        Publishing results to a Hugging Face Space.

        .. code::

            trackioc = TrackioCallback(
                project="my-optuna-study",
                space_id="username/optuna-dashboard",
            )


    Args:
        project:
            Name of the Trackio project. This determines the local storage
            directory and is also used when synchronizing results to a
            Hugging Face Space or Dataset.

        metric_name:
            Name assigned to the optimized metric. In case of multi-objective
            optimization, a sequence of names can be provided. These names
            are assigned to objective values in the order returned by the
            objective function.

            If a single name is provided (default: ``"value"``), it will be
            broadcast to multiple objectives using a numerical suffix,
            e.g., ``value_0``, ``value_1``.

            The number of metric names must match the number of objective
            values returned.

        as_multirun:
            If ``True``, creates a new Trackio run for each Optuna trial.
            This is useful for per-trial analysis, parameter importance
            visualizations, and sweep-style dashboards.

            If ``False`` (default), all trials are logged into a single run.

        space_id:
            Optional Hugging Face Space ID (``"username/space-name"``) to
            which the tracked project will be synchronized. If the Space
            does not exist, it will be created automatically.

        dataset_id:
            Optional Hugging Face Dataset ID (``"username/dataset-name"``)
            used to export trial metrics and parameters as a versioned
            dataset for offline analysis and reproducibility.

        private:
            Whether the Hugging Face Space or Dataset should be private.
            Defaults to the user or organization’s Hub settings.

        trackio_kwargs:
            Additional keyword arguments passed to :func:`trackio.init`,
            such as ``resume`` or UI-related configuration options.

    """

    def __init__(
        self,
        project: str,
        metric_name: str | Sequence[str] = "value",
        *,
        as_multirun: bool = False,
        space_id: str | None = None,
        dataset_id: str | None = None,
        private: bool | None = None,
        trackio_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(metric_name, (str, Sequence)):
            raise TypeError(f"metric_name must be str or sequence[str], got {type(metric_name)}")

        self._project = project
        self._metric_name = metric_name
        self._as_multirun = as_multirun
        self._space_id = space_id
        self._dataset_id = dataset_id
        self._private = private
        self._trackio_kwargs = trackio_kwargs or {}

        self._run = None

        if not self._as_multirun:
            self._run = self._init_run(name="optuna-study")

    # ------------------------
    # Optuna callback hook
    # ------------------------
    def __call__(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        if trial.values is None:
            return  # failed or pruned

        step = trial.number

        if self._as_multirun:
            self._init_run(name=f"trial-{trial.number}")

        metrics = self._build_metrics(trial)
        params = dict(trial.params)

        trackio.log(
            {
                **params,
                **metrics,
                "trial_number": trial.number,
            },
            step=step,
        )

        if self._as_multirun:
            trackio.finish()

    # ------------------------
    # Decorator support
    # ------------------------
    @experimental_func("0.1.0")
    def track_in_trackio(self) -> Callable:
        """Decorator enabling logging inside objective functions."""

        def decorator(func: ObjectiveFuncType) -> ObjectiveFuncType:
            @functools.wraps(func)
            def wrapper(trial: optuna.trial.Trial):
                if self._as_multirun and self._run is None:
                    self._run = self._init_run(name=f"trial-{trial.number}")
                return func(trial)

            return wrapper

        return decorator

    # ------------------------
    # Internals
    # ------------------------
    def _init_run(self, name: str):
        return trackio.init(
            project=self._project,
            name=name,
            space_id=self._space_id,
            dataset_id=self._dataset_id,
            private=self._private,
            config={
                "backend": "optuna",
            },
            **self._trackio_kwargs,
        )

    def _build_metrics(self, trial: optuna.trial.FrozenTrial) -> dict[str, float]:
        values = trial.values
        assert values is not None

        if isinstance(self._metric_name, str):
            if len(values) == 1:
                names = [self._metric_name]
            else:
                names = [f"{self._metric_name}_{i}" for i in range(len(values))]
        else:
            if len(self._metric_name) != len(values):
                raise ValueError(
                    "Metric names must match number of objectives "
                    f"({len(self._metric_name)} vs {len(values)})"
                )
            names = list(self._metric_name)

        return dict(zip(names, values))
