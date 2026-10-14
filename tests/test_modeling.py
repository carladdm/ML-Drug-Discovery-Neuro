"""Fase 4 tests: registry factory, pipeline policy, Optuna tuning engine.

The search spaces come from ``configs/model/registry.yaml`` via
``load_config(["model=registry"])`` — the same source of truth the trainer uses.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna import Trial

from mldd.config import AlgorithmName, load_config
from mldd.constants import Task
from mldd.exceptions import TrainingError
from mldd.modeling.models import build_estimator, build_pipeline
from mldd.modeling.train import train_classification_model, train_regression_model
from mldd.modeling.tuning import TuningResult, select_best, suggest_params, tune_candidate

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


# ------------------------------------------------------------------------------
# T4.2: Optuna engine over the config-declared search spaces
# ------------------------------------------------------------------------------
def test_suggest_params_follows_declared_space() -> None:
    """Every declared param is suggested exactly once, within its bounds."""
    space = _REGISTRY_MODEL.candidate(Task.REGRESSION, "rf").space
    study = optuna.create_study()

    def _dummy(trial: Trial) -> float:
        params = suggest_params(trial, space)
        assert set(params) == set(space)
        assert space["n_estimators"].low <= params["n_estimators"] <= space["n_estimators"].high
        assert params["max_features"] in ("sqrt", "log2", None)
        return 0.0

    study.optimize(_dummy, n_trials=3)


def test_tune_candidate_is_deterministic_under_seed() -> None:
    """Same data + same seed => same best params and score (seeded TPE)."""
    X, y = _tiny_xy(Task.REGRESSION)
    training = load_config(
        ["training=regression", "training.n_trials=2", "training.cv_folds=3"]
    ).training
    candidate = _REGISTRY_MODEL.candidate(Task.REGRESSION, "rf")
    first = tune_candidate(
        X, y, task=Task.REGRESSION, name="rf", candidate=candidate, training=training, seed=42
    )
    second = tune_candidate(
        X, y, task=Task.REGRESSION, name="rf", candidate=candidate, training=training, seed=42
    )
    assert first.best_params == second.best_params
    assert first.best_score == pytest.approx(second.best_score)
    assert first.n_trials == 2


def test_select_best_picks_highest_cv_score() -> None:
    results = [
        TuningResult(algorithm="rf", best_params={}, best_score=0.1, n_trials=2),
        TuningResult(algorithm="xgb", best_params={}, best_score=0.9, n_trials=2),
    ]
    assert select_best(results).algorithm == "xgb"


def test_select_best_rejects_empty_list() -> None:
    with pytest.raises(TrainingError, match="empty"):
        select_best([])


# ------------------------------------------------------------------------------
# T4.3: registry-driven training workflows (end-to-end on toy data)
# ------------------------------------------------------------------------------
_FAST_PROTOCOL = ["training.n_trials=2", "training.cv_folds=3"]


def test_train_classification_end_to_end() -> None:
    """Full tune → select → refit → holdout workflow on toy data (I/O-free)."""
    X, y = _tiny_xy(Task.CLASSIFICATION)
    result = train_classification_model(X, y, overrides=[*_FAST_PROTOCOL, "training.stratify=true"])
    assert result.algorithm in ("xgb", "rf", "lgbm", "lda")
    assert -1.0 <= result.holdout_mcc <= 1.0
    assert 0.0 <= result.holdout_roc_auc <= 1.0
    assert 0.0 <= result.holdout_accuracy <= 1.0
    assert result.best_params  # best trial parameters of the winning study


def test_train_regression_end_to_end() -> None:
    """Full tune → select → refit → holdout workflow on toy data (I/O-free)."""
    X, y = _tiny_xy(Task.REGRESSION)
    result = train_regression_model(X, y, overrides=_FAST_PROTOCOL)
    assert result.algorithm in ("xgb", "rf", "hgb", "svr")
    assert result.holdout_rmse >= 0.0
    assert result.holdout_r2 <= 1.0
    assert result.best_params
