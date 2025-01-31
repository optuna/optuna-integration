# flake8: noqa

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import functools
import json

import optuna
from optuna._experimental import experimental_class
from optuna.study.study import ObjectiveFuncType

from optuna_integration._imports import try_import


with try_import() as _imports:
    import comet_ml


@experimental_class("4.0.0")
class CometCallback:
    """
    A callback for logging Optuna study trials to a Comet ML Experiment.
    Comet ML must be installed to run.

    This callback is intended for use with :meth:`optuna.study.Study.optimize` method. It ensures
    that all trials from an Optuna study are logged to a single Comet Experiment, facilitating
    organized tracking of hyperparameter optimization.
    The callback supports both single and multi-objective optimization.

    In a distributed training context, where trials from the same study might occur on different
    machines, this callback ensures consistency by logging to the same Comet Experiment using
    an experiment key stored within the study's user attributes.

    By default, Trials are logged as Comet Experiments, which will automatically log code,
    system metrics, and many other values.
    However, it also adds some computational overhead (potentially a few seconds).

    Args:
        study:
            The Optuna study object to which the callback is attached.
        workspace:
            The workspace in Comet ML where the project resides.
        project_name:
            The name of the project in Comet ML where the experiment will be logged.
            Defaults to ``"general"``.
        metric_names:
            A list of the names of your objective metrics.

    Example:

        Here is an example.

        .. code::

            study = optuna.create_study(directions=["maximize", "maximize"])
            comet_callback = CometCallback(
                study,
                metric_names=["accuracy", "top_k_accuracy"],
                project_name="your_project_name",
                workspace="your_workspace",
            )
            study.optimize(your_objective_function, n_trials=100, callbacks=[comet_callback])

    .. note:
        The callback checks for an existing Comet Experiment key in the study's user attributes.
        If present, it initializes an ``ExistingExperiment``; otherwise,
        it creates a new ``APIExperiment`` and stores its key in the study for future reference.

        You will need a Comet API key to log data to Comet.

        You can also log extra data directly to your Trial's Experiment via the objective function
        by using the ``@CometCallback.track_in_comet`` decorator,
        which exposes an ``experiment`` property on your trial, like so:

        .. code::

            study = optuna.create_study(directions=["maximize", "maximize"])
            comet_callback = CometCallback(
                study,
                metric_names=["accuracy", "top_k_accuracy"],
                project_name="your_project_name",
                workspace="your_workspace",
            )


            @comet_callback.track_in_comet()
            def your_objective(trial):
                trial.experiment.log_other("foo", "bar")
                # Rest of your objective function...


            study.optimize(your_objective, n_trials=100, callbacks=[comet_callback])

    """

    def __init__(
        self,
        study: optuna.study.Study,
        workspace: str | None = None,
        project_name: str | None = "general",
        metric_names: Sequence[str] | None = None,
    ):
        self._project_name = project_name
        self._workspace = workspace
        self._study = study

        if metric_names is None:
            metric_names = []

        self._metric_names = metric_names

        if self._workspace is None:
            API = comet_ml.api.API()
            self._workspace = API.get_default_workspace()

        # APIExperiment associated with the Optuna Study.
        self.study_experiment = self._init_optuna_study_experiment(study)

        # Log the directions of the objectives.
        for i, direction in enumerate(study.directions):
            direction_str = (
                "minimize" if direction == optuna.study.StudyDirection.MINIMIZE else "maximize"
            )
            metric_name = metric_names[i] if i < len(metric_names) else i
            self.study_experiment.log_other(f"direction_of_objective_{metric_name}", direction_str)

        # Dictionary of experiment keys associated with specific Optuna Trials.
        trial_experiments = study.user_attrs.get("trial_experiments")
        if trial_experiments is None:
            trial_experiments = {}
            study.set_user_attr("trial_experiments", trial_experiments)

        self._trial_experiments = trial_experiments

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        trial_experiment = self._init_optuna_trial_experiment(study, trial)

        trial_experiment.log_parameters(trial.params)
        trial_experiment.log_other("trial_number", trial.number)
        trial_experiment.add_tag(f"trial_number_{trial.number}")

        # Check if the study is multi-objective.
        if trial.values and len(trial.values) > 1:
            # Log each objective value separately for multi-objective optimization.
            for i, val in enumerate(trial.values):
                metric_name = self._metric_names[i] if i < len(self._metric_names) else i
                trial_experiment.log_metric(f"{metric_name}", val)
        else:
            # Log single objective value.
            metric_name = (
                self._metric_names[0] if len(self._metric_names) > 0 else "objective_value"
            )
            trial_experiment.log_optimization(
                optimization_id=study.study_name,
                metric_name=metric_name,
                metric_value=trial.value,
                parameters=trial.params,
                objective=study.direction,
            )

        # Log the best trials to the APIExperiment associated with the Study.
        self.study_experiment.log_other(
            "best_trials", json.dumps([trial.number for trial in study.best_trials])
        )

        trial_experiment.end()

    def _init_optuna_study_experiment(self, study: optuna.Study) -> "comet_ml.APIExperiment":
        # Check if we've already created an APIExperiment for this Study.
        experiment_key = study.user_attrs.get("comet_study_experiment_key")

        # Load the existing APIExperiment, if present. Else, make a new APIExperiment.
        if experiment_key:
            study_experiment = comet_ml.APIExperiment(previous_experiment=experiment_key)
        else:
            study_experiment = comet_ml.APIExperiment(
                workspace=self._workspace, project_name=self._project_name
            )
            study_experiment.add_tag("optuna_study")
            study_experiment.log_other("optuna_study_name", study.study_name)
            study_experiment.log_other("optuna_storage", type(study._storage).__name__)
            study.set_user_attr("comet_study_experiment_key", study_experiment.key)

        return study_experiment

    def _init_optuna_trial_experiment(
        self, study: optuna.study.Study, trial: optuna.trial.BaseTrial
    ) -> "comet_ml.Experiment":

        # Check to see if the Trial experiment already exists.
        experiment_key = self._trial_experiments.get(trial.number)

        # Check to see if there is a current active experiment in this environment.
        if hasattr(comet_ml, "active_experiment"):
            if experiment_key == comet_ml.active_experiment.get_key():
                # No need to re-initialize if we already have the right experiment.
                return comet_ml.active_experiment
            elif comet_ml.active_experiment.ended is False:
                comet_ml.active_experiment.end()

        # Load the existing Experiment, if present. Else, make a new Experiment.
        if experiment_key:
            experiment = comet_ml.ExistingExperiment(previous_experiment=experiment_key)
        else:
            experiment = comet_ml.Experiment(
                workspace=self._workspace, project_name=self._project_name
            )

            experiment.add_tag("optuna_trial")
            experiment.log_other("optuna_study_name", study.study_name)

            self._trial_experiments[trial.number] = experiment.get_key()
            study.set_user_attr("trial_experiments", self._trial_experiments)

        setattr(comet_ml, "active_experiment", experiment)
        return experiment

    def track_in_comet(self) -> Callable:
        def decorator(func: ObjectiveFuncType) -> ObjectiveFuncType:
            @functools.wraps(func)
            def wrapper(trial: optuna.trial.Trial) -> float | Sequence[float]:
                experiment = self._init_optuna_trial_experiment(self._study, trial)

                # Add the experiment to the trial object for easier access for the end-users.
                trial.experiment = experiment  # type: ignore
                return func(trial)

            return wrapper

        return decorator
