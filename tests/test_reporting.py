"""Fase 7 / T7 tests: report figures (evaluation, comparison, XAI)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from mldd.config import (
    AlgorithmName,
    AppConfig,
    PathsConfig,
    ReportConfig,
    TrackingConfig,
    load_config,
)
from mldd.constants import Task
from mldd.exceptions import ReportError
from mldd.modeling.predict import TrainedModel
from mldd.reporting import (
    _best_cv_scores,
    _xai_figures,
    plot_classification_eval,
    plot_regression_scatter,
    plot_shap_summary,
    plot_top_features,
    resolve_model_path,
    run_report,
)

_FEATURES = [f"PubchemFP{index}" for index in range(1, 6)]


def _fast_cfg(task: Task, tmp_path: Path) -> AppConfig:
    cfg = load_config([f"training={task.value}"])
    return cfg.model_copy(
        update={
            "paths": PathsConfig(
                data_dir=str(tmp_path / "data"),
                models_dir=str(tmp_path / "models"),
                reports_dir=str(tmp_path / "figures"),
            ),
            "tracking": TrackingConfig(
                tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
                experiment_name="report-test",
            ),
        }
    )


def _matrix(task: Task, n_rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    df = pd.DataFrame({column: rng.integers(0, 2, size=n_rows) for column in _FEATURES})
    df["pchembl_value"] = 5.0 + df[_FEATURES].sum(axis=1) * 0.4
    df["bioactivity_class"] = (df["pchembl_value"] >= 6.0).astype(int)
    return df


def _bundle(task: Task, *, forest: bool = False) -> TrainedModel:
    df = _matrix(task)
    X = df[_FEATURES]
    algorithm: AlgorithmName = "lda"
    if task is Task.CLASSIFICATION:
        y: pd.Series = df["bioactivity_class"]
        estimator = LogisticRegression(random_state=0)
    elif forest:
        y = df["pchembl_value"]
        estimator = RandomForestRegressor(n_estimators=10, random_state=0)
        algorithm = "rf"
    else:
        y = df["pchembl_value"]
        estimator = LinearRegression()
        algorithm = "svr"
    estimator.fit(X, y)
    return TrainedModel(
        pipeline=Pipeline([("model", estimator)]),
        feature_columns=_FEATURES,
        target="TOY",
        task=task,
        algorithm=algorithm,
    )


def _save_bundle(cfg: AppConfig, bundle: TrainedModel) -> Path:
    cfg.models_path.mkdir(parents=True, exist_ok=True)
    path = cfg.models_path / f"{cfg.target.dataset}_{bundle.task.value}_{bundle.algorithm}.joblib"
    joblib.dump(bundle, path)
    return path


def _log_run(
    cfg: AppConfig, *, algorithm: str, score: float, target: str | None = None, task: str
) -> None:
    mlflow.set_tracking_uri(cfg.tracking.tracking_uri)
    mlflow.set_experiment(cfg.tracking.experiment_name)
    with mlflow.start_run():
        mlflow.set_tags(
            {
                "target": target or cfg.target.dataset,
                "task": task,
                "algorithm": algorithm,
            }
        )
        mlflow.log_metrics({"cv_score": score})


# ------------------------------------------------------------------------------
# Model resolution
# ------------------------------------------------------------------------------
def test_resolve_model_path_missing_raises(tmp_path: Path) -> None:
    cfg = _fast_cfg(Task.REGRESSION, tmp_path)
    with pytest.raises(ReportError, match="No model bundle"):
        resolve_model_path(cfg)


def test_resolve_model_path_picks_newest(tmp_path: Path) -> None:
    cfg = _fast_cfg(Task.REGRESSION, tmp_path)
    cfg.models_path.mkdir(parents=True)
    older = cfg.models_path / "MAO-B_regression_svr.joblib"
    newer = cfg.models_path / "MAO-B_regression_rf.joblib"
    joblib.dump({"v": 1}, older)
    joblib.dump({"v": 2}, newer)
    os.utime(older, (1_000_000_000, 1_000_000_000))
    assert resolve_model_path(cfg) == newer


# ------------------------------------------------------------------------------
# Figure builders
# ------------------------------------------------------------------------------
def test_plot_regression_scatter_writes_figure(tmp_path: Path) -> None:
    y_true = pd.Series([5.0, 6.0, 7.0, 8.0])
    path = plot_regression_scatter(y_true, np.array([5.1, 5.9, 7.2, 7.8]), tmp_path / "s.png")
    assert path.is_file() and path.stat().st_size > 0


def test_plot_classification_eval_writes_figure(tmp_path: Path) -> None:
    y_true = pd.Series([0, 1, 1, 0, 1])
    path = plot_classification_eval(
        y_true,
        np.array([0, 1, 1, 0, 0]),
        np.array([0.2, 0.8, 0.7, 0.4, 0.45]),
        tmp_path / "e.png",
    )
    assert path.is_file() and path.stat().st_size > 0


def test_plot_top_features_respects_top_n(tmp_path: Path) -> None:
    importances = np.arange(5, dtype=float)
    path = plot_top_features(_FEATURES, importances, tmp_path / "t.png", top_n=3)
    assert path.is_file() and path.stat().st_size > 0


def test_plot_shap_summary_tree_model_writes_figure(tmp_path: Path) -> None:
    bundle = _bundle(Task.REGRESSION, forest=True)
    X = _matrix(Task.REGRESSION)[_FEATURES].head(20)
    path = plot_shap_summary(bundle, X, tmp_path / "shap.png", max_display=5)
    assert path.is_file() and path.stat().st_size > 0


def test_xai_figures_caps_shap_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Regression: reporting.shap_max_samples must bound TreeExplainer's input.

    The cap existed in ReportConfig but was never applied, so `mldd report`
    computed SHAP values over the entire holdout (minutes-to-hours on real
    matrices) instead of the configured sample.
    """
    cfg = _fast_cfg(Task.REGRESSION, tmp_path).model_copy(
        update={"reporting": ReportConfig(shap_max_samples=7, top_features=5)}
    )
    cfg.reports_path.mkdir(parents=True, exist_ok=True)  # run_report does this
    bundle = _bundle(Task.REGRESSION, forest=True)
    X = _matrix(Task.REGRESSION, n_rows=20)[_FEATURES]

    observed: dict[str, int] = {}

    class _FakeTreeExplainer:
        def __init__(self, estimator: Any) -> None:
            self.estimator = estimator

        def shap_values(self, features: Any) -> np.ndarray:
            rows, cols = np.asarray(features).shape
            observed["rows"] = rows
            return np.zeros((rows, cols))

    def _fake_summary_plot(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(shap, "TreeExplainer", _FakeTreeExplainer)
    monkeypatch.setattr(shap, "summary_plot", _fake_summary_plot)

    written = _xai_figures(cfg, bundle, X)

    assert observed["rows"] == 7  # capped, not the 20 rows of X
    assert len(written) == 2  # top-features + shap summary


# ------------------------------------------------------------------------------
# MLflow-backed comparison scores
# ------------------------------------------------------------------------------
def test_best_cv_scores_keeps_max_per_algorithm(tmp_path: Path) -> None:
    cfg = _fast_cfg(Task.REGRESSION, tmp_path)
    _log_run(cfg, algorithm="rf", score=0.50, task="regression")
    _log_run(cfg, algorithm="xgb", score=0.60, task="regression")
    _log_run(cfg, algorithm="xgb", score=0.55, task="regression")
    # excluded: different task tag
    _log_run(cfg, algorithm="xgb", score=0.99, task="classification")

    assert _best_cv_scores(cfg) == {"rf": 0.50, "xgb": 0.60}


def test_best_cv_scores_empty_without_experiment(tmp_path: Path) -> None:
    cfg = _fast_cfg(Task.REGRESSION, tmp_path)
    assert _best_cv_scores(cfg) == {}


# ------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------
def test_run_report_regression_end_to_end(tmp_path: Path) -> None:
    cfg = _fast_cfg(Task.REGRESSION, tmp_path)
    cfg.matrix_path.parent.mkdir(parents=True, exist_ok=True)
    _matrix(Task.REGRESSION).to_csv(cfg.matrix_path, index=False)
    _save_bundle(cfg, _bundle(Task.REGRESSION))
    _log_run(cfg, algorithm="svr", score=0.61, task="regression")

    figures = run_report(cfg)

    assert {figure.name for figure in figures} == {
        "MAO-B_regression_predicted_vs_experimental.png",
        "MAO-B_regression_models_comparison.png",
        "MAO-B_regression_top_features.png",  # LinearRegression: SHAP skipped
    }
    assert all(figure.is_file() and figure.stat().st_size > 0 for figure in figures)


def test_run_report_classification_end_to_end(tmp_path: Path) -> None:
    cfg = _fast_cfg(Task.CLASSIFICATION, tmp_path)
    cfg.matrix_path.parent.mkdir(parents=True, exist_ok=True)
    _matrix(Task.CLASSIFICATION).to_csv(cfg.matrix_path, index=False)
    _save_bundle(cfg, _bundle(Task.CLASSIFICATION))
    _log_run(cfg, algorithm="lda", score=0.72, task="classification")

    figures = run_report(cfg)

    assert {figure.name for figure in figures} == {
        "MAO-B_classification_evaluation.png",
        "MAO-B_classification_models_comparison.png",
        "MAO-B_classification_top_features.png",
    }


def test_run_report_skips_comparison_without_mlflow_runs(tmp_path: Path) -> None:
    cfg = _fast_cfg(Task.REGRESSION, tmp_path)
    cfg.matrix_path.parent.mkdir(parents=True, exist_ok=True)
    _matrix(Task.REGRESSION).to_csv(cfg.matrix_path, index=False)
    _save_bundle(cfg, _bundle(Task.REGRESSION))

    figures = run_report(cfg)

    names = {figure.name for figure in figures}
    assert "MAO-B_regression_models_comparison.png" not in names
    assert "MAO-B_regression_predicted_vs_experimental.png" in names
