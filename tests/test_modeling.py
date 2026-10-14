"""Fase 4 tests: registry factory, pipeline policy, Optuna tuning engine.

The search spaces come from ``configs/model/registry.yaml`` via
``load_config(["model=registry"])`` — the same source of truth the trainer uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mldd.config import AlgorithmName, load_config
from mldd.constants import Task
from mldd.exceptions import TrainingError
from mldd.modeling.models import build_estimator, build_pipeline

_REGISTRY_MODEL = load_config(["model=registry"]).model


def _tiny_xy(task: Task, n_rows: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    """Deterministic toy dataset: 4 features, balanced binary or continuous y."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n_rows, 4)), columns=[f"f{i}" for i in range(4)])
    if task is Task.CLASSIFICATION:
        y: pd.Series = pd.Series([0, 1] * (n_rows // 2))
    else:
        y = pd.Series(rng.normal(size=n_rows))
    return X, y


# ------------------------------------------------------------------------------
# T4.1: estimator factory over the registry
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("task", "name"),
    [
        (Task.CLASSIFICATION, "xgb"),
        (Task.CLASSIFICATION, "rf"),
        (Task.CLASSIFICATION, "lgbm"),
        (Task.CLASSIFICATION, "lda"),
        (Task.REGRESSION, "xgb"),
        (Task.REGRESSION, "rf"),
        (Task.REGRESSION, "hgb"),
        (Task.REGRESSION, "svr"),
    ],
)
def test_build_estimator_instantiates_every_registry_algorithm(
    task: Task, name: AlgorithmName
) -> None:
    candidate = _REGISTRY_MODEL.candidate(task, name)
    estimator = build_estimator(task, name, dict(candidate.fixed), seed=42)
    assert estimator is not None
    if "random_state" in estimator.get_params():
        assert estimator.get_params()["random_state"] == 42


def test_build_estimator_rejects_algorithm_outside_task_registry() -> None:
    with pytest.raises(TrainingError, match="not available"):
        build_estimator(Task.CLASSIFICATION, "svr", {}, seed=42)


def test_deterministic_algorithms_do_not_receive_random_state() -> None:
    lda = build_estimator(Task.CLASSIFICATION, "lda", {"solver": "lsqr", "shrinkage": 0.1}, seed=42)
    assert "random_state" not in lda.get_params()
    svr = build_estimator(Task.REGRESSION, "svr", {"C": 1.0}, seed=42)
    assert "random_state" not in svr.get_params()


# ------------------------------------------------------------------------------
# T4.1: pipeline scale policy (NB06 scales all; NB07 scales only SVR)
# ------------------------------------------------------------------------------
def test_build_pipeline_scale_policy() -> None:
    estimator = build_estimator(
        Task.CLASSIFICATION, "lda", {"solver": "lsqr", "shrinkage": 0.1}, seed=42
    )
    scaled = build_pipeline(estimator, scale=True)
    assert "scaler" in scaled.named_steps and "model" in scaled.named_steps
    plain = build_pipeline(estimator, scale=False)
    assert "scaler" not in plain.named_steps
