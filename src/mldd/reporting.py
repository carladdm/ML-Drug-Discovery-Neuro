"""Evaluation & XAI report generation (Fase 7 / T7; DoD #3: I/O lives here).

Regenerates the thesis-style core figures for one target/task from the
artifacts the production pipeline already produces:

* **models comparison** â€” best CV score per registry algorithm, read back
  from the MLflow experiment (tags: target, task, algorithm);
* **holdout evaluation** â€” predicted-vs-experimental scatter (regression)
  or confusion matrix + ROC curve (classification), reproduced with the
  exact training holdout split (same seed and stratify policy);
* **XAI** â€” top-N global feature importance and a SHAP summary
  (TreeExplainer) for tree ensembles; linear models (LDA/SVR) fall back
  to |coefficients| for importance and skip SHAP with a logged warning.

Figures land in ``paths.reports_dir`` with thesis-style names.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")  # batch generation must never require a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    auc,
    confusion_matrix,
    matthews_corrcoef,
    r2_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mldd.config import AppConfig
from mldd.constants import PCHEMBL_VALUE, Task
from mldd.exceptions import ReportError
from mldd.logging_utils import get_logger
from mldd.modeling.predict import TrainedModel, align_features
from mldd.predictor import load_model
from mldd.trainer import _resolve_tracking_uri, load_model_matrix, split_features_and_target

logger: logging.Logger = get_logger(__name__)

#: Figure naming stems (thesis-style), appended to ``{dataset}_{task}_``.
_MODELS_COMPARISON: Final = "models_comparison"
_EVALUATION: Final = "evaluation"
_SCATTER: Final = "predicted_vs_experimental"
_TOP_FEATURES: Final = "top_features"
_SHAP_SUMMARY: Final = "shap_summary"


def resolve_model_path(cfg: AppConfig) -> Path:
    """Locate the trainer artifact for this target/task (newest wins).

    Args:
        cfg: Validated configuration (target, task, models directory).

    Returns:
        Path of the most recent ``{target}_{task}_{algorithm}.joblib``.

    Raises:
        ReportError: No bundle matches â€” run ``mldd train`` first.
    """
    pattern = f"{cfg.target.dataset}_{cfg.training.task.value}_*.joblib"
    matches = [path for path in cfg.models_path.glob(pattern) if path.is_file()]
    if not matches:
        raise ReportError(
            f"No model bundle for '{cfg.target.dataset}' [{cfg.training.task.value}] "
            f"under {cfg.models_path}; run `mldd train target=...` first."
        )
    newest = max(matches, key=lambda path: path.stat().st_mtime)
    if len(matches) > 1:
        logger.warning(
            "%d bundles match '%s'; using the newest: %s.", len(matches), pattern, newest
        )
    return newest


def _holdout_split(
    cfg: AppConfig, X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reproduce the training holdout split exactly (same seed/policy)."""
    split: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        random_state=cfg.seed,
        stratify=y if cfg.training.stratify else None,
    )
    return split


def _global_importance(estimator: Any) -> np.ndarray | None:
    """Global feature importances: impurity/gain or |coefficients|.

    Args:
        estimator: The bundle's final estimator.

    Returns:
        One non-negative importance per feature, or ``None`` when the
        estimator exposes neither attribute.
    """
    importances = getattr(estimator, "feature_importances_", None)
    if importances is not None:
        return np.asarray(importances, dtype=float)
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        return None
    array = np.asarray(coef, dtype=float)
    return np.abs(array if array.ndim == 1 else array.mean(axis=0))


def _best_cv_scores(cfg: AppConfig) -> dict[str, float]:
    """Best recorded CV score per algorithm for this target/task (MLflow)."""
    client = MlflowClient(tracking_uri=_resolve_tracking_uri(cfg.tracking.tracking_uri))
    experiment = client.get_experiment_by_name(cfg.tracking.experiment_name)
    if experiment is None:
        return {}
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.target = '{cfg.target.dataset}' AND tags.task = "
            f"'{cfg.training.task.value}' AND attributes.status = 'FINISHED'"
        ),
    )
    best: dict[str, float] = {}
    for run in runs:
        algorithm = run.data.tags.get("algorithm")
        score = run.data.metrics.get("cv_score")
        if algorithm is None or score is None:
            continue
        best[algorithm] = max(best.get(algorithm, float("-inf")), float(score))
    return best


# ------------------------------------------------------------------------------
# Figure builders (each writes exactly one PNG)
# ------------------------------------------------------------------------------
def plot_models_comparison(cfg: AppConfig, scores: dict[str, float], path: Path) -> Path:
    """Bar chart of the best CV score per registry algorithm."""
    algorithms = sorted(scores)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(algorithms, [scores[name] for name in algorithms], color="steelblue")
    ax.set_ylabel(f"Best CV {cfg.training.scoring}")
    ax.set_title(f"{cfg.target.dataset} [{cfg.training.task.value}] â€” registry comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s.", path)
    return path


def plot_regression_scatter(y_true: pd.Series, y_pred: np.ndarray, path: Path) -> Path:
    """Predicted-vs-experimental scatter for the regression holdout."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="none")
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    ax.plot([low, high], [low, high], "k--", linewidth=1)
    ax.set_xlabel(f"Experimental {PCHEMBL_VALUE}")
    ax.set_ylabel(f"Predicted {PCHEMBL_VALUE}")
    ax.set_title(f"Holdout (R^2={r2_score(y_true, y_pred):.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s.", path)
    return path


def plot_classification_eval(
    y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray, path: Path
) -> Path:
    """Confusion matrix + ROC curve for the classification holdout."""
    matrix = confusion_matrix(y_true, y_pred)
    false_positive, true_positive, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(false_positive, true_positive)
    mcc = matthews_corrcoef(y_true, y_pred)

    fig, (ax_matrix, ax_roc) = plt.subplots(1, 2, figsize=(11, 5))
    image = ax_matrix.imshow(matrix, cmap="Blues")
    for (row, col), value in np.ndenumerate(matrix):
        ax_matrix.text(col, row, str(value), ha="center", va="center")
    ax_matrix.set_xlabel("Predicted")
    ax_matrix.set_ylabel("Actual")
    ax_matrix.set_title(f"Confusion matrix (MCC={mcc:.3f})")
    fig.colorbar(image, ax=ax_matrix)

    ax_roc.plot(false_positive, true_positive, label=f"ROC AUC={roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curve")
    ax_roc.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s.", path)
    return path


def plot_top_features(
    feature_names: list[str], importances: np.ndarray, path: Path, *, top_n: int
) -> Path:
    """Horizontal bar chart of the top-N global feature importances."""
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[index] for index in order][::-1],
        importances[order][::-1],
        color="darkorange",
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(order)} influential features")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s.", path)
    return path


def plot_shap_summary(
    bundle: TrainedModel, X: pd.DataFrame, path: Path, *, max_display: int
) -> Path:
    """SHAP summary plot of the bundle's tree estimator (TreeExplainer).

    The SHAP model is the final estimator; the pipeline's preprocessing
    steps (e.g. StandardScaler) are applied manually so SHAP sees the
    exact feature space the estimator was fitted on. ``X`` is the sample
    to explain — the caller bounds it via ``reporting.shap_max_samples``.
    """
    import shap  # heavy import: only pay for it when XAI runs

    estimator = bundle.pipeline.steps[-1][1]
    pre_steps = bundle.pipeline.steps[:-1]
    features = X.to_numpy()
    if pre_steps:
        features = np.asarray(Pipeline(pre_steps).transform(X))
    values: Any = shap.TreeExplainer(estimator).shap_values(features)
    if isinstance(values, list):
        values = values[-1]  # active class under older multi-output APIs
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[..., -1]
    shap.summary_plot(
        array,
        features=features,
        feature_names=list(X.columns),
        show=False,
        max_display=max_display,
    )
    figure = plt.gcf()
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    logger.info("Wrote %s.", path)
    return path


# ------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------
def _xai_figures(cfg: AppConfig, bundle: TrainedModel, X_eval: pd.DataFrame) -> list[Path]:
    """Top-N importance + SHAP summary, skipping unsupported estimators."""
    estimator = bundle.pipeline.steps[-1][1]
    stem = f"{cfg.target.dataset}_{bundle.task.value}"
    written: list[Path] = []

    importances = _global_importance(estimator)
    if importances is not None:
        written.append(
            plot_top_features(
                list(X_eval.columns),
                importances,
                cfg.reports_path / f"{stem}_{_TOP_FEATURES}.png",
                top_n=cfg.reporting.top_features,
            )
        )
    else:
        logger.warning("Estimator exposes no importances; skipping top-features figure.")

    try:
        # Bound the explainer cost: reporting.shap_max_samples caps the rows
        # (seeded sample, so figures are reproducible across runs).
        shap_rows = min(cfg.reporting.shap_max_samples, len(X_eval))
        written.append(
            plot_shap_summary(
                bundle,
                X_eval.sample(n=shap_rows, random_state=cfg.seed),
                cfg.reports_path / f"{stem}_{_SHAP_SUMMARY}.png",
                max_display=cfg.reporting.top_features,
            )
        )
    except Exception as exc:  # non-tree estimators (LDA/SVR) cannot TreeExplainer
        logger.warning("SHAP summary skipped (%s): %s", type(exc).__name__, exc)
    if not written:
        logger.warning(
            "No XAI figures for the '%s' model (unsupported estimator).", bundle.algorithm
        )
    return written


def run_report(cfg: AppConfig) -> list[Path]:
    """Generate the core figures (evaluation + models comparison + XAI).

    Args:
        cfg: Validated configuration (target, task, reporting knobs).

    Returns:
        The figure paths written under ``paths.reports_dir``.

    Raises:
        ReportError: No model bundle exists for this target/task.
        TrainingError: Missing/invalid model matrix.
        FeaturizationError: Feature schema mismatch between bundle and matrix.
    """
    bundle = load_model(resolve_model_path(cfg))
    df = load_model_matrix(cfg.matrix_path)
    X, y = split_features_and_target(df, cfg.training.task)
    X = align_features(X, bundle.feature_columns)

    _, X_test, _, y_test = _holdout_split(cfg, X, y)
    cfg.reports_path.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.target.dataset}_{bundle.task.value}"
    written: list[Path] = []

    if bundle.task is Task.CLASSIFICATION:
        predictions = bundle.pipeline.predict(X_test)
        probabilities = np.asarray(bundle.pipeline.predict_proba(X_test))[:, 1]
        written.append(
            plot_classification_eval(
                y_test,
                predictions,
                probabilities,
                cfg.reports_path / f"{stem}_{_EVALUATION}.png",
            )
        )
    else:
        written.append(
            plot_regression_scatter(
                y_test,
                np.asarray(bundle.pipeline.predict(X_test)),
                cfg.reports_path / f"{stem}_{_SCATTER}.png",
            )
        )

    scores = _best_cv_scores(cfg)
    if scores:
        written.append(
            plot_models_comparison(
                cfg, scores, cfg.reports_path / f"{stem}_{_MODELS_COMPARISON}.png"
            )
        )
    else:
        logger.warning(
            "No completed MLflow runs for '%s' [%s]; skipping the comparison figure.",
            cfg.target.dataset,
            cfg.training.task.value,
        )

    written.extend(_xai_figures(cfg, bundle, X_test))
    logger.info(
        "Generated %d figure(s) for %s [%s].",
        len(written),
        cfg.target.dataset,
        cfg.training.task.value,
    )
    return written
