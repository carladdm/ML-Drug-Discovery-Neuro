"""Registry-driven training workflows (classification & regression).

Fase 4 / T4.3: every candidate of the task registry (NB06/NB07) is tuned
with a seeded Optuna study, the winner by cross-validated score is refitted
on the training split and scored on the holdout. Fase 5 / T5.1 wraps this
into the MLflow-tracked trainer and moves persistence out of this module
(debt B15).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

from mldd.config import MODELS_DIR, AlgorithmName, AppConfig, load_config
from mldd.constants import Task
from mldd.logging_utils import get_logger
from mldd.modeling.models import build_estimator, build_pipeline
from mldd.modeling.tuning import TuningResult, select_best, tune_candidate

logger: logging.Logger = get_logger(__name__)


@dataclass(frozen=True)
class ClassificationResult:
    """Fitted pipeline + cross-validated and holdout metrics (NB06 protocol)."""

    pipeline: Any
    algorithm: AlgorithmName
    best_params: dict[str, Any]
    cv_score: float
    holdout_mcc: float
    holdout_roc_auc: float
    holdout_accuracy: float


@dataclass(frozen=True)
class RegressionResult:
    """Fitted pipeline + cross-validated and holdout metrics (NB07 protocol)."""

    pipeline: Any
    algorithm: AlgorithmName
    best_params: dict[str, Any]
    cv_score: float
    holdout_r2: float
    holdout_rmse: float


def _tune_registry(
    X_train: pd.DataFrame, y_train: pd.Series, cfg: AppConfig, task: Task
) -> list[TuningResult]:
    """Tune every registry candidate of ``task`` on the training split."""
    results: list[TuningResult] = []
    for name in cfg.model.candidates[task]:
        logger.info("Tuning candidate '%s' (%d trials)...", name, cfg.training.n_trials)
        results.append(
            tune_candidate(
                X_train,
                y_train,
                task=task,
                name=name,
                candidate=cfg.model.candidate(task, name),
                training=cfg.training,
                seed=cfg.seed,
            )
        )
    return results


def _final_pipeline(cfg: AppConfig, task: Task, best: TuningResult) -> Any:
    """Build the winning pipeline (fixed + best params, scale policy)."""
    candidate = cfg.model.candidate(task, best.algorithm)
    estimator = build_estimator(
        task, best.algorithm, {**candidate.fixed, **best.best_params}, seed=cfg.seed
    )
    return build_pipeline(estimator, scale=candidate.scale)


def _persist(cfg: AppConfig, pipeline: Any, task: Task, algorithm: AlgorithmName) -> None:
    """Persist the fitted pipeline under models/ (debt B15: Fase 5 trainer)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{cfg.target.name}_{task.value}_{algorithm}.joblib"
    joblib.dump(pipeline, model_path)
    logger.info("Model persisted to %s", model_path)


def train_classification_model(
    X: pd.DataFrame, y: pd.Series, overrides: list[str] | None = None
) -> ClassificationResult:
    """Tune the classification registry, refit the winner, score the holdout.

    Reproduces NB06 at production quality: the holdout split honours the
    stratify policy from ``configs/training/classification.yaml``, every
    candidate is tuned with a seeded TPE study and the winner is the one
    with the best mean CV MCC.

    Args:
        X: Feature matrix (e.g. PaDEL PubChem fingerprints).
        y: Binary target (0 = inactive, 1 = active).
        overrides: Extra Hydra overrides (``target=ache``,
            ``training.n_trials=2``...).

    Returns:
        The fitted pipeline plus CV and holdout metrics (MCC, ROC-AUC,
        accuracy).

    Raises:
        TrainingError: Tuning failed or the classification registry is empty.
        ConfigError: Invalid configuration composition/validation.
    """
    cfg = load_config(["training=classification", *(overrides or [])])
    logger.info(
        "Splitting data into Train / Test (%.0f/%.0f)...",
        (1 - cfg.training.test_size) * 100,
        cfg.training.test_size * 100,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        random_state=cfg.seed,
        stratify=y if cfg.training.stratify else None,
    )
    best = select_best(_tune_registry(X_train, y_train, cfg, Task.CLASSIFICATION))
    logger.info(
        "Winning candidate: '%s' (CV %s=%.4f)",
        best.algorithm,
        cfg.training.scoring,
        best.best_score,
    )
    pipeline = _final_pipeline(cfg, Task.CLASSIFICATION, best)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    result = ClassificationResult(
        pipeline=pipeline,
        algorithm=best.algorithm,
        best_params=best.best_params,
        cv_score=best.best_score,
        holdout_mcc=float(matthews_corrcoef(y_test, preds)),
        holdout_roc_auc=float(roc_auc_score(y_test, probs)),
        holdout_accuracy=float(accuracy_score(y_test, preds)),
    )
    _persist(cfg, result.pipeline, Task.CLASSIFICATION, result.algorithm)
    logger.info(
        "Classification finished. Holdout MCC=%.4f | ROC-AUC=%.4f | Acc=%.4f",
        result.holdout_mcc,
        result.holdout_roc_auc,
        result.holdout_accuracy,
    )
    return result


def train_regression_model(
    X: pd.DataFrame, y: pd.Series, overrides: list[str] | None = None
) -> RegressionResult:
    """Tune the regression registry, refit the winner, score the holdout.

    Reproduces NB07 at production quality: no stratification (T1.3 policy),
    every candidate tuned with a seeded TPE study maximising R² over the
    configured KFold, winner refitted and scored with R² and RMSE on the
    holdout.

    Args:
        X: Feature matrix (e.g. PaDEL PubChem fingerprints).
        y: Continuous target (pChEMBL value).
        overrides: Extra Hydra overrides (``target=ache``,
            ``training.n_trials=2``...).

    Returns:
        The fitted pipeline plus CV and holdout metrics (R², RMSE).

    Raises:
        TrainingError: Tuning failed or the regression registry is empty.
        ConfigError: Invalid configuration composition/validation.
    """
    cfg = load_config(["training=regression", *(overrides or [])])
    logger.info(
        "Splitting regression data (%.0f/%.0f)...",
        (1 - cfg.training.test_size) * 100,
        cfg.training.test_size * 100,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.training.test_size, random_state=cfg.seed
    )
    best = select_best(_tune_registry(X_train, y_train, cfg, Task.REGRESSION))
    logger.info(
        "Winning candidate: '%s' (CV %s=%.4f)",
        best.algorithm,
        cfg.training.scoring,
        best.best_score,
    )
    pipeline = _final_pipeline(cfg, Task.REGRESSION, best)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    result = RegressionResult(
        pipeline=pipeline,
        algorithm=best.algorithm,
        best_params=best.best_params,
        cv_score=best.best_score,
        holdout_r2=float(r2_score(y_test, preds)),
        holdout_rmse=float(root_mean_squared_error(y_test, preds)),
    )
    _persist(cfg, result.pipeline, Task.REGRESSION, result.algorithm)
    logger.info(
        "pChEMBL regression finished. Holdout R2=%.4f | RMSE=%.4f",
        result.holdout_r2,
        result.holdout_rmse,
    )
    return result
