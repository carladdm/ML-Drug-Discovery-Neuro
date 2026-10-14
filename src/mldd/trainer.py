"""MLflow-tracked training boundary (DoD #3: training I/O lives here).

Loads the processed model matrix produced by the featurization stage, runs
the registry-driven workflows in :mod:`mldd.modeling.train` and persists
both artifacts of a run: the fitted pipeline (joblib under ``models/``) and
the MLflow record (params, metrics, tags, model artifact) under the
tracking configuration of ``configs/config.yaml``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import joblib
import mlflow
import pandas as pd

from mldd.config import BASE_DIR, AlgorithmName, AppConfig
from mldd.constants import (
    BIOACTIVITY_CLASS,
    CURATED_COLUMNS,
    PCHEMBL_VALUE,
    SMILES_CLEAN,
    TARGET_ID,
    Task,
)
from mldd.exceptions import TrainingError
from mldd.logging_utils import get_logger
from mldd.modeling.train import fit_classification_model, fit_regression_model

logger: logging.Logger = get_logger(__name__)

#: Metadata/target columns of the model matrix; everything else is a feature.
_METADATA_COLUMNS: Final[frozenset[str]] = frozenset(
    {BIOACTIVITY_CLASS, PCHEMBL_VALUE, SMILES_CLEAN, TARGET_ID, *CURATED_COLUMNS}
)


@dataclass(frozen=True)
class TrainingRun:
    """Summary of one tracked training run."""

    target: str
    task: Task
    algorithm: AlgorithmName
    best_params: dict[str, Any]
    metrics: dict[str, float]
    model_path: Path
    mlflow_run_id: str


def load_model_matrix(path: Path) -> pd.DataFrame:
    """Load the processed model matrix (features + target columns).

    Args:
        path: CSV written by the featurization stage (NB04/NB05 matrix).

    Returns:
        The full matrix as read from disk.

    Raises:
        TrainingError: The file is missing or lacks a target column
            (``bioactivity_class`` or ``pchembl_value``).
    """
    if not path.is_file():
        raise TrainingError(f"Model matrix not found: {path}")
    df = pd.read_csv(path)
    missing = [column for column in (BIOACTIVITY_CLASS, PCHEMBL_VALUE) if column not in df.columns]
    if missing:
        raise TrainingError(f"Model matrix is missing target column(s): {missing}")
    logger.info("Loaded model matrix %s (%d rows, %d columns).", path, *df.shape)
    return df


def split_features_and_target(df: pd.DataFrame, task: Task) -> tuple[pd.DataFrame, pd.Series]:
    """Split the model matrix into X (features) and y (task target).

    Args:
        df: Full model matrix (metadata + features + targets).
        task: Selects ``bioactivity_class`` (classification) or
            ``pchembl_value`` (regression) as the target column.

    Returns:
        The feature matrix (every non-metadata column) and the target vector.

    Raises:
        TrainingError: No feature columns remain, or the target has nulls.
    """
    target_column = BIOACTIVITY_CLASS if task is Task.CLASSIFICATION else PCHEMBL_VALUE
    feature_columns = [column for column in df.columns if column not in _METADATA_COLUMNS]
    if not feature_columns:
        raise TrainingError("Model matrix has no feature columns.")
    y = df[target_column]
    if y.isna().any():
        raise TrainingError(f"Target column '{target_column}' contains nulls.")
    return df[feature_columns].copy(), y


#: Tracking URIs backed by a SQLite file whose relative path must not
#: depend on the process CWD (mlflow ≥3 deprecated the filesystem store).
_SQLITE_PREFIX: Final = "sqlite:///"


def _resolve_tracking_uri(uri: str) -> str:
    """Resolve relative tracking URIs against the repo root (never CWD)."""
    if uri.startswith(_SQLITE_PREFIX):
        db_path = Path(uri[len(_SQLITE_PREFIX) :])
        if not db_path.is_absolute():
            db_path = (BASE_DIR / db_path).resolve()
        return f"{_SQLITE_PREFIX}{db_path}"
    if "://" in uri or Path(uri).is_absolute():
        return uri
    return str((BASE_DIR / uri).resolve())


def _track(
    cfg: AppConfig,
    data_path: Path,
    *,
    algorithm: AlgorithmName,
    best_params: dict[str, Any],
    metrics: dict[str, float],
    model_path: Path,
) -> str:
    """Log params/metrics/artifacts of one training run to MLflow.

    Returns:
        The MLflow run id.
    """
    mlflow.set_tracking_uri(_resolve_tracking_uri(cfg.tracking.tracking_uri))
    mlflow.set_experiment(cfg.tracking.experiment_name)
    run_name = f"{cfg.target.name}_{cfg.training.task.value}_{algorithm}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "target": cfg.target.dataset,
                "task": cfg.training.task.value,
                "algorithm": algorithm,
                "seed": str(cfg.seed),
                "data_path": str(data_path),
            }
        )
        mlflow.log_params({name: str(value) for name, value in best_params.items()})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path))
        run_id = str(run.info.run_id)
    logger.info("MLflow run %s logged under experiment '%s'.", run_id, cfg.tracking.experiment_name)
    return run_id


def run_training(cfg: AppConfig, data_path: Path) -> TrainingRun:
    """Train one target/task end-to-end with MLflow tracking.

    Loads the model matrix, tunes the task registry, refits the winner and
    persists the joblib pipeline under ``models/`` plus the MLflow run.

    Args:
        cfg: Validated configuration (task, target, registry, tracking).
        data_path: Processed model matrix CSV for this target.

    Returns:
        Summary of the run (winner, metrics, artifact paths, MLflow id).

    Raises:
        TrainingError: Missing/invalid model matrix or failed training.
    """
    df = load_model_matrix(data_path)
    X, y = split_features_and_target(df, cfg.training.task)

    if cfg.training.task is Task.CLASSIFICATION:
        tuned_cls = fit_classification_model(X, y, cfg)
        pipeline, algorithm, best_params = (
            tuned_cls.pipeline,
            tuned_cls.algorithm,
            tuned_cls.best_params,
        )
        metrics = {
            "cv_score": tuned_cls.cv_score,
            "holdout_mcc": tuned_cls.holdout_mcc,
            "holdout_roc_auc": tuned_cls.holdout_roc_auc,
            "holdout_accuracy": tuned_cls.holdout_accuracy,
        }
    else:
        tuned_reg = fit_regression_model(X, y, cfg)
        pipeline, algorithm, best_params = (
            tuned_reg.pipeline,
            tuned_reg.algorithm,
            tuned_reg.best_params,
        )
        metrics = {
            "cv_score": tuned_reg.cv_score,
            "holdout_r2": tuned_reg.holdout_r2,
            "holdout_rmse": tuned_reg.holdout_rmse,
        }

    cfg.models_path.mkdir(parents=True, exist_ok=True)
    model_path = cfg.models_path / f"{cfg.target.name}_{cfg.training.task.value}_{algorithm}.joblib"
    joblib.dump(pipeline, model_path)
    logger.info("Fitted pipeline persisted to %s.", model_path)

    run_id = _track(
        cfg,
        data_path,
        algorithm=algorithm,
        best_params=best_params,
        metrics=metrics,
        model_path=model_path,
    )
    return TrainingRun(
        target=cfg.target.name,
        task=cfg.training.task,
        algorithm=algorithm,
        best_params=best_params,
        metrics=metrics,
        model_path=model_path,
        mlflow_run_id=run_id,
    )
