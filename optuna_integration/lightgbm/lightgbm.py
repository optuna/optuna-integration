from __future__ import annotations

from typing import TYPE_CHECKING

import optuna
from optuna._imports import try_import


if TYPE_CHECKING:
    from lightgbm.basic import _LGBM_BoosterEvalMethodResultType
    from lightgbm.basic import _LGBM_BoosterEvalMethodResultWithStandardDeviationType
    from lightgbm.callback import CallbackEnv


with try_import() as _imports:
    import lightgbm as lgb  # NOQA


class LightGBMPruningCallback:
    """Callback for LightGBM to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/main/
    lightgbm/lightgbm_integration.py>`__
    if you want to add a pruning callback which observes accuracy
    of a LightGBM model.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of
            the objective function.
        metric:
            An evaluation metric for pruning, e.g., ``binary_error`` and ``multi_error``.
            Please refer to
            `LightGBM reference
            <https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric>`_
            for further details.
        valid_name:
            The name of the target validation.
            Validation names are specified by ``valid_names`` option of
            `train method
            <https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.train>`_.
            If omitted, ``valid_0`` is used which is the default name of the first validation.
            Note that this argument will be ignored if you are calling
            `cv method <https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.cv>`_
            instead of train method.
        report_interval:
            Check if the trial should report intermediate values for pruning every n-th boosting
            iteration. By default ``report_interval=1`` and reporting is performed after every
            iteration. Note that the pruning itself is performed according to the interval
            definition of the pruner.
    """

    def __init__(
        self,
        trial: optuna.trial.Trial,
        metric: str,
        valid_name: str = "valid_0",
        report_interval: int = 1,
    ) -> None:
        _imports.check()

        self._trial = trial
        self._valid_name = valid_name
        self._metric = metric
        self._report_interval = report_interval

    def _find_evaluation_result(
        self, target_valid_name: str, env: CallbackEnv
    ) -> (
        _LGBM_BoosterEvalMethodResultType
        | _LGBM_BoosterEvalMethodResultWithStandardDeviationType
        | None
    ):
        evaluation_result_list = env.evaluation_result_list
        if evaluation_result_list is None:
            return None

        # The structure of each member of `evaluation_result_list` as of LightGBM v4.6.0.
        # [
        #     (<dataset_name>, <metric_name>, avg(<values>), <is_higher_better>, std_dev(<values>))
        # ]
        for evaluation_result in evaluation_result_list:
            valid_name, metric, current_score, is_higher_better = evaluation_result[:4]
            # The prefix "valid " is added to metric name since LightGBM v4.0.0.
            if valid_name != target_valid_name or (
                metric != "valid " + self._metric and metric != self._metric
            ):
                continue
            return evaluation_result

        return None

    def __call__(self, env: CallbackEnv) -> None:
        if (env.iteration + 1) % self._report_interval == 0:
            # If this callback has been passed to `lightgbm.cv` function,
            # the value of `is_cv` becomes `True`. See also:
            # https://github.com/microsoft/LightGBM/blob/v4.1.0/python-package/lightgbm/engine.py#L533
            # Note that `5` is not the number of folds but the length of sequence.
            evaluation_result_list = env.evaluation_result_list
            is_cv = (
                evaluation_result_list is not None
                and len(evaluation_result_list) > 0
                and len(evaluation_result_list[0]) == 5
            )
            if is_cv:
                target_valid_name = "valid"
            else:
                target_valid_name = self._valid_name

            evaluation_result = self._find_evaluation_result(target_valid_name, env)
            if evaluation_result is None:
                raise ValueError(
                    'The entry associated with the validation name "{}" and the metric name "{}" '
                    "is not found in the evaluation result list {}.".format(
                        target_valid_name, self._metric, str(env.evaluation_result_list)
                    )
                )

            valid_name, metric, current_score, is_higher_better = evaluation_result[:4]

            if is_higher_better:
                if self._trial.study.direction != optuna.study.StudyDirection.MAXIMIZE:
                    raise ValueError(
                        "The intermediate values are inconsistent with the objective values"
                        "in terms of study directions. Please specify a metric to be minimized"
                        "for LightGBMPruningCallback."
                    )
            else:
                if self._trial.study.direction != optuna.study.StudyDirection.MINIMIZE:
                    raise ValueError(
                        "The intermediate values are inconsistent with the objective values"
                        "in terms of study directions. Please specify a metric to be"
                        "maximized for LightGBMPruningCallback."
                    )

            self._trial.report(current_score, step=env.iteration)

            if self._trial.should_prune():
                message = "Trial was pruned at iteration {}.".format(env.iteration)
                raise optuna.TrialPruned(message)
