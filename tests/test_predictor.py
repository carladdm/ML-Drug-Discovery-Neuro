"""Fase 6 / T6.2 tests: bundle loading and batch prediction I/O."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from mldd.config import AlgorithmName, AppConfig, load_config
from mldd.constants import Task
from mldd.exceptions import ModelNotFoundError, PredictionError
from mldd.modeling.predict import (
    PREDICTED_CLASS,
    PREDICTED_PCHEMBL,
    PROBABILITY,
    TrainedModel,
)
from mldd.predictor import load_model, run_prediction

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


def _write_model(tmp_path: Path, bundle: TrainedModel) -> Path:
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    return path


def _write_candidates(tmp_path: Path) -> Path:
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "molecule_chembl_id": ["MOL1", "MOL2", "MOL3"],
            "canonical_smiles": ["CCO", "CCN", "CCC"],
            "vendor": ["A", "B", "C"],
        }
    ).to_csv(path, index=False)
    return path


def _mock_fingerprints(monkeypatch: pytest.MonkeyPatch, bundle: TrainedModel) -> None:
    """Fake PaDEL: training bits plus an irrelevant extra bit (raw 881-style)."""
    frame = pd.DataFrame({column: [0, 1, 1] for column in bundle.feature_columns})
    frame["PubchemFP999"] = [1, 0, 0]
    monkeypatch.setattr(
        "mldd.predictor.compute_pubchem_fingerprints",
        lambda smiles, ids, feature_config: frame,
    )


# ------------------------------------------------------------------------------
# Bundle loading
# ------------------------------------------------------------------------------
def test_load_model_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match="not found"):
        load_model(tmp_path / "nope.joblib")


def test_load_model_rejects_bare_artifact(tmp_path: Path) -> None:
    path = tmp_path / "legacy.joblib"
    joblib.dump({"pipeline": object()}, path)
    with pytest.raises(PredictionError, match="not a TrainedModel"):
        load_model(path)


def test_load_model_roundtrip(tmp_path: Path) -> None:
    bundle = _fitted_bundle(Task.CLASSIFICATION)
    loaded = load_model(_write_model(tmp_path, bundle))
    assert loaded.feature_columns == _FEATURES
    assert loaded.task is Task.CLASSIFICATION
    assert loaded.algorithm == "lda"
    assert loaded.target == "TOY"


# ------------------------------------------------------------------------------
# Batch prediction I/O
# ------------------------------------------------------------------------------
def test_run_prediction_classification_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _fitted_bundle(Task.CLASSIFICATION)
    model_path = _write_model(tmp_path, bundle)
    input_csv = _write_candidates(tmp_path)
    output_csv = tmp_path / "predictions.csv"
    _mock_fingerprints(monkeypatch, bundle)

    table = run_prediction(
        load_config(["training=classification"]), model_path, input_csv, output_csv
    )

    assert output_csv.is_file()
    assert len(table) == 3
    assert {PREDICTED_CLASS, PROBABILITY} <= set(table.columns)
    assert "vendor" in table.columns  # extra input columns pass through
    assert set(table[PREDICTED_CLASS].unique()) <= {0, 1}
    assert len(pd.read_csv(output_csv)) == 3


def test_run_prediction_regression_uses_config_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _fitted_bundle(Task.REGRESSION)
    model_path = _write_model(tmp_path, bundle)
    input_csv = _write_candidates(tmp_path)
    output_csv = tmp_path / "predictions.csv"
    _mock_fingerprints(monkeypatch, bundle)

    cfg: AppConfig = load_config(["training=regression"])
    table = run_prediction(cfg, model_path, input_csv, output_csv)

    assert PREDICTED_PCHEMBL in table.columns
    expected = (table[PREDICTED_PCHEMBL] >= cfg.data.active_threshold).astype(int)
    assert (table[PREDICTED_CLASS] == expected).all()


def test_run_prediction_missing_candidates_file(tmp_path: Path) -> None:
    bundle = _fitted_bundle(Task.CLASSIFICATION)
    model_path = _write_model(tmp_path, bundle)
    with pytest.raises(PredictionError, match="Candidates file not found"):
        run_prediction(load_config([]), model_path, tmp_path / "missing.csv", tmp_path / "out.csv")
