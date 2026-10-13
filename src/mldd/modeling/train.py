"""Dual training workflows (classification & regression).

.. note::
    Fase 4 / T4.1-T4.3 replaces the static RandomForest baselines with the
    thesis model registry (XGB/RF/LGBM/HGB/SVR/LDA) and Optuna tuning;
    Fase 5 / T5.1 wraps everything into the MLflow-tracked trainer.
"""

from __future__ import annotations

import logging

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

from mldd.config import MODELS_DIR, load_config
from mldd.logging_utils import get_logger

logger: logging.Logger = get_logger(__name__)


def train_classification_model(
    X: pd.DataFrame, y: pd.Series
) -> tuple[RandomForestClassifier, float]:
    """Train a Random Forest classifier to sort molecules into Active / Inactive."""
    cfg = load_config(["training=classification"])
    logger.info(
        "Splitting data into stratified Train / Test (%.0f/%.0f)...",
        (1 - cfg.training.test_size) * 100,
        cfg.training.test_size * 100,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.training.test_size, random_state=cfg.seed, stratify=y
    )

    logger.info("Training RandomForestClassifier...")
    clf = RandomForestClassifier(**cfg.model.params, random_state=cfg.seed)
    clf.fit(X_train, y_train)

    # Evaluate via probability for the ROC curve / AUC
    y_proba = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    logger.info("Classification finished. Test ROC-AUC Score: %.4f", roc_auc)

    # Persist the model under models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "rf_bioactivity_classifier.joblib"
    joblib.dump(clf, model_path)

    return clf, roc_auc


def train_regression_model(
    X: pd.DataFrame, y: pd.Series
) -> tuple[RandomForestRegressor, float, float]:
    """Train a Random Forest regressor to predict the continuous pIC50 value."""
    cfg = load_config(["training=regression"])
    logger.info(
        "Splitting regression data (%.0f/%.0f)...",
        (1 - cfg.training.test_size) * 100,
        cfg.training.test_size * 100,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.training.test_size, random_state=cfg.seed
    )

    logger.info("Training RandomForestRegressor...")
    reg = RandomForestRegressor(**cfg.model.params, random_state=cfg.seed)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    logger.info("pIC50 regression finished. Test R2: %.4f | RMSE: %.4f", r2, rmse)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "rf_pic50_regressor.joblib"
    joblib.dump(reg, model_path)

    return reg, r2, rmse
