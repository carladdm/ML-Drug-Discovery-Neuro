"""Fase 6 / T6.1 tests: inference bundle, input contract and pure prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError
from sklearn.linear_model import LinearRegression, LogisticRegression

from mldd.config import AlgorithmName
from mldd.constants import Task
from mldd.exceptions import FeaturizationError
from mldd.modeling.predict import (
    CANDIDATE_SCHEMA,
    PREDICTED_CLASS,
    PREDICTED_PCHEMBL,
    PROBABILITY,
    TrainedModel,
    align_features,
    predict_records,
)

_FEATURES = [f"PubchemFP{index}" for index in range(1, 6)]


def _fitted_bundle(task: Task) -> TrainedModel:
    """Deterministic toy bundle: 5 fingerprint bits -> task target."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.integers(0, 2, size=(40, 5)), columns=_FEATURES)
    algorithm: AlgorithmName = "lda"
    if task is Task.CLASSIFICATION:
        y: pd.Series = pd.Series((X.sum(axis=1) >= 3).astype(int))
        estimator = LogisticRegression(random_state=0)
    else:
        y = pd.Series(4.0 + 0.5 * X.sum(axis=1))
        estimator = LinearRegression()
        algorithm = "svr"
    estimator.fit(X, y)
    return TrainedModel(
        pipeline=estimator,
        feature_columns=_FEATURES,
        target="TOY",
        task=task,
        algorithm=algorithm,
    )


def _fingerprints(n_rows: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {f"PubchemFP{index}": rng.integers(0, 2, size=n_rows) for index in range(881)}
    )


# ------------------------------------------------------------------------------
# Feature alignment to the training schema
# ------------------------------------------------------------------------------
def test_align_features_selects_and_reorders() -> None:
    raw_full = _fingerprints()
    shuffled = raw_full[["PubchemFP5", "PubchemFP1", "PubchemFP3", "PubchemFP2", "PubchemFP4"]]
    aligned = align_features(shuffled, _FEATURES)
    assert list(aligned.columns) == _FEATURES
    # values follow the column mapping, not the incoming frame order
    pd.testing.assert_series_equal(
        aligned["PubchemFP5"].reset_index(drop=True), raw_full["PubchemFP5"]
    )


def test_align_features_rejects_missing_training_columns() -> None:
    with pytest.raises(FeaturizationError, match="training column"):
        align_features(_fingerprints()[["PubchemFP1"]], _FEATURES)


# ------------------------------------------------------------------------------
# Pandera input contract
# ------------------------------------------------------------------------------
def test_candidate_schema_requires_smiles() -> None:
    with pytest.raises(SchemaError):
        CANDIDATE_SCHEMA.validate(pd.DataFrame({"canonical_smiles": [None, "CCO"]}))


def test_candidate_schema_allows_extra_columns_and_missing_ids() -> None:
    frame = pd.DataFrame({"canonical_smiles": ["CCO"], "vendor": ["ACME"]})
    validated = CANDIDATE_SCHEMA.validate(frame)
    assert list(validated.columns) == ["canonical_smiles", "vendor"]


def test_candidate_schema_coerces_numeric_ids_to_strings() -> None:
    frame = pd.DataFrame({"canonical_smiles": ["CCO"], "molecule_chembl_id": [12345]})
    validated = CANDIDATE_SCHEMA.validate(frame)
    assert validated["molecule_chembl_id"].tolist() == ["12345"]


# ------------------------------------------------------------------------------
# Pure prediction
# ------------------------------------------------------------------------------
def test_predict_records_classification_outputs_class_and_probability() -> None:
    bundle = _fitted_bundle(Task.CLASSIFICATION)
    out = predict_records(bundle, _fingerprints(), active_threshold=6.0)
    assert list(out.columns) == [PREDICTED_CLASS, PROBABILITY]
    assert set(out[PREDICTED_CLASS].unique()) <= {0, 1}
    assert out[PROBABILITY].between(0.0, 1.0).all()
    assert len(out) == 10


def test_predict_records_regression_outputs_pchembl_and_threshold_class() -> None:
    bundle = _fitted_bundle(Task.REGRESSION)
    out = predict_records(bundle, _fingerprints(), active_threshold=6.0)
    assert list(out.columns) == [PREDICTED_PCHEMBL, PREDICTED_CLASS]
    expected = (out[PREDICTED_PCHEMBL] >= 6.0).astype(int)
    pd.testing.assert_series_equal(
        out[PREDICTED_CLASS].rename(None), expected.rename(None), check_names=False
    )


def test_predict_records_requires_training_columns() -> None:
    bundle = _fitted_bundle(Task.REGRESSION)
    with pytest.raises(FeaturizationError, match="training column"):
        predict_records(bundle, _fingerprints()[["PubchemFP1", "PubchemFP2"]], active_threshold=6.0)
