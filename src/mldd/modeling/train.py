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

from mldd.config import MODEL_PARAMS, MODELS_DIR, SEED
from mldd.logging_utils import get_logger

logger: logging.Logger = get_logger(__name__)


def train_classification_model(
    X: pd.DataFrame, y: pd.Series
) -> tuple[RandomForestClassifier, float]:
    """Train a Random Forest classifier to sort molecules into Active / Inactive."""
    logger.info("Splitting data into stratified Train / Test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    logger.info("Training RandomForestClassifier...")
    clf = RandomForestClassifier(**MODEL_PARAMS)
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
    logger.info("Splitting regression data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    logger.info("Training RandomForestRegressor...")
    reg = RandomForestRegressor(**MODEL_PARAMS)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    logger.info("pIC50 regression finished. Test R2: %.4f | RMSE: %.4f", r2, rmse)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "rf_pic50_regressor.joblib"
    joblib.dump(reg, model_path)

    return reg, r2, rmse
