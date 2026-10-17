"""Fase 5 / T5.1 tests: model-matrix loading and the MLflow trainer."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from mldd.config import AppConfig, PathsConfig, TrackingConfig, load_config
from mldd.constants import Task
from mldd.exceptions import TrainingError
from mldd.modeling.predict import TrainedModel
from mldd.trainer import load_model_matrix, run_training, split_features_and_target


def _tiny_matrix(task: Task, n_rows: int = 60) -> pd.DataFrame:
    """Deterministic toy model matrix in the processed-CSV contract."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"PubchemFP{i}": rng.integers(0, 2, size=n_rows) for i in range(10)})
    df["pchembl_value"] = rng.normal(6.0, 1.0, size=n_rows)
    df["bioactivity_class"] = [0, 1] * (n_rows // 2)
    df["molecule_chembl_id"] = [f"MOL{i}" for i in range(n_rows)]
    return df


def _fast_cfg(task: Task, tmp_path: Path) -> AppConfig:
    """Config with a shrunken protocol and isolated models/ + SQLite tracking."""
    cfg = load_config([f"training={task.value}", "training.n_trials=2", "training.cv_folds=3"])
    return cfg.model_copy(
        update={
            "paths": PathsConfig(models_dir=str(tmp_path / "models")),
            "tracking": TrackingConfig(
                tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", experiment_name="mldd-test"
            ),
        }
    )


# ------------------------------------------------------------------------------
# Matrix loading & X/y split (trainer boundary contract)
# ------------------------------------------------------------------------------
def test_load_model_matrix_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TrainingError, match="not found"):
        load_model_matrix(tmp_path / "nope.csv")


def test_load_model_matrix_requires_target_columns(tmp_path: Path) -> None:
    path = tmp_path / "features_only.csv"
    pd.DataFrame({"PubchemFP1": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(TrainingError, match="target column"):
        load_model_matrix(path)


def test_split_features_and_target_uses_task_target() -> None:
    df = _tiny_matrix(Task.REGRESSION)
    X, y = split_features_and_target(df, Task.REGRESSION)
    assert list(X.columns) == [f"PubchemFP{i}" for i in range(10)]
    assert y.name == "pchembl_value"
    X_cls, y_cls = split_features_and_target(df, Task.CLASSIFICATION)
    assert y_cls.name == "bioactivity_class"
    assert len(X_cls) == len(y_cls) == len(df)


def test_split_features_and_target_rejects_null_target() -> None:
    df = _tiny_matrix(Task.REGRESSION)
    df.loc[0, "pchembl_value"] = np.nan
    with pytest.raises(TrainingError, match="nulls"):
        split_features_and_target(df, Task.REGRESSION)


# ------------------------------------------------------------------------------
# T5.1: end-to-end run_training with MLflow tracking (isolated file store)
# ------------------------------------------------------------------------------
def test_run_training_classification_end_to_end(tmp_path: Path) -> None:
    data_path = tmp_path / "matrix.csv"
    _tiny_matrix(Task.CLASSIFICATION).to_csv(data_path, index=False)
    cfg = _fast_cfg(Task.CLASSIFICATION, tmp_path)

    run = run_training(cfg, data_path)

    assert run.algorithm in ("xgb", "rf", "lgbm", "lda")
    assert run.model_path.is_file()
    bundle = joblib.load(run.model_path)
    assert isinstance(bundle, TrainedModel) and len(bundle.feature_columns) == 10
    assert bundle.task is Task.CLASSIFICATION
    assert {"cv_score", "holdout_mcc", "holdout_roc_auc", "holdout_accuracy"} <= set(run.metrics)
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    logged = client.get_run(run.mlflow_run_id)
    assert "holdout_mcc" in logged.data.metrics
    assert logged.data.params  # winner hyper-parameters were logged


def test_run_training_regression_end_to_end(tmp_path: Path) -> None:
    data_path = tmp_path / "matrix.csv"
    _tiny_matrix(Task.REGRESSION).to_csv(data_path, index=False)
    cfg = _fast_cfg(Task.REGRESSION, tmp_path)

    run = run_training(cfg, data_path)

    assert run.algorithm in ("xgb", "rf", "hgb", "svr")
    assert run.model_path.is_file()
    bundle = joblib.load(run.model_path)
    assert isinstance(bundle, TrainedModel) and len(bundle.feature_columns) == 10
    assert bundle.task is Task.REGRESSION
    assert {"cv_score", "holdout_r2", "holdout_rmse"} <= set(run.metrics)
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    logged = client.get_run(run.mlflow_run_id)
    assert "holdout_r2" in logged.data.metrics
    assert logged.data.tags["task"] == "regression"
