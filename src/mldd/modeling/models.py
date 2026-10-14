"""Thesis model registry: estimator factory & pipeline policy (Fase 4 / T4.1).

Reproduces the NB06/NB07 estimator zoo without hardcoding hyper-parameters:
every tunable value is suggested from the search spaces declared in
``configs/model/registry.yaml`` and validated by :class:`mldd.config.ModelConfig`.

*   Classification registry (NB06): XGB, RF, LGBM, LDA.
*   Regression registry (NB07): XGB, RF, HGB, SVR.
"""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBClassifier, XGBRegressor

from mldd.config import AlgorithmName
from mldd.constants import Task
from mldd.exceptions import TrainingError

_CLASSIFIERS: dict[AlgorithmName, type[Any]] = {
    "xgb": XGBClassifier,
    "rf": RandomForestClassifier,
    "lgbm": LGBMClassifier,
    "lda": LinearDiscriminantAnalysis,
}

_REGRESSORS: dict[AlgorithmName, type[Any]] = {
    "xgb": XGBRegressor,
    "rf": RandomForestRegressor,
    "hgb": HistGradientBoostingRegressor,
    "svr": SVR,
}


def build_estimator(
    task: Task,
    name: AlgorithmName,
    params: dict[str, Any],
    *,
    seed: int,
) -> Any:
    """Instantiate registry algorithm ``name`` for ``task`` with ``params``.

    ``random_state`` is injected from the global seed when the estimator
    supports it (XGB/RF/LGBM/HGB); LDA and SVR are deterministic and reject
    the parameter, so they are left untouched.

    Args:
        task: ML task selecting the classifier/regressor registry.
        name: Registry algorithm identifier (configs/model/registry.yaml).
        params: Merged ``fixed`` + Optuna-suggested hyper-parameters.
        seed: Global randomness seed (configs/config.yaml ``seed``).

    Returns:
        A fresh, unfitted estimator instance.

    Raises:
        TrainingError: The algorithm is not available for that task.
    """
    registry = _CLASSIFIERS if task is Task.CLASSIFICATION else _REGRESSORS
    try:
        estimator_cls = registry[name]
    except KeyError as exc:
        raise TrainingError(f"Model '{name}' is not available for task '{task}'") from exc
    estimator = estimator_cls(**params)
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=seed)
    return estimator


def build_pipeline(estimator: Any, *, scale: bool) -> Pipeline:
    """Wrap ``estimator`` in the notebook pipeline policy (NB06/NB07).

    ``scale=True`` reproduces NB06 (StandardScaler before every classifier)
    and NB07's SVR branch; ``scale=False`` feeds raw features to the tree
    models, as in NB07.
    """
    if not scale:
        return Pipeline([("model", estimator)])
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
