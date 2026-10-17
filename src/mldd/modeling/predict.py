"""Inference engine on new SMILES candidates (Fase 6 / T6.1).

Pure module: the :class:`TrainedModel` bundle contract, the Pandera input
contract and vectorized prediction functions — no filesystem access. The
PaDEL featurization and joblib persistence live at the I/O boundary
(:mod:`mldd.predictor`), which injects ``data.active_threshold`` from
``configs/`` for the regression activity call.

Why the bundle carries the feature schema: the training matrix is already
variance-filtered (configured post-merge threshold), so a fresh 881-bit
PaDEL matrix must be re-aligned to the exact training columns before
inference — that mapping is target-specific and travels with the fitted
pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from pandera.pandas import Column, DataFrameSchema

from mldd.config import AlgorithmName
from mldd.constants import MOLECULE_ID, SMILES, Task
from mldd.exceptions import FeaturizationError
from mldd.logging_utils import get_logger

logger: logging.Logger = get_logger(__name__)

# --- prediction output contract ----------------------------------------------
PREDICTED_CLASS: Final = "predicted_class"
PREDICTED_PCHEMBL: Final = "predicted_pchembl"
PROBABILITY: Final = "probability"

#: Input contract of a candidates file: raw SMILES plus optional ChEMBL ids.
#: Extra columns are allowed (and passed through to the output untouched).
CANDIDATE_SCHEMA: Final = DataFrameSchema(
    {
        SMILES: Column(str, nullable=False),
        MOLECULE_ID: Column(str, required=False, nullable=True),
    },
    strict=False,
    coerce=True,
)


@dataclass(frozen=True)
class TrainedModel:
    """Serializable inference bundle persisted by the trainer.

    Attributes:
        pipeline: Fitted sklearn pipeline (features -> predictions).
        feature_columns: Exact training feature schema, in order.
        target: Human-readable target name (e.g. ``MAO-B``).
        task: Prediction task of the bundle.
        algorithm: Registry algorithm that won the tuning run.
    """

    pipeline: Any
    feature_columns: list[str]
    target: str
    task: Task
    algorithm: AlgorithmName


def align_features(fingerprints: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Select and reorder a fresh fingerprint matrix to the training schema.

    Args:
        fingerprints: Raw 881-bit ``PubchemFP*`` matrix for the candidates.
        feature_columns: The bundle's training feature schema.

    Returns:
        The matrix restricted to ``feature_columns`` in that exact order
        (copy; the caller's frame is never mutated).

    Raises:
        FeaturizationError: The fingerprint matrix lacks a training column,
            i.e. the featurization contract was violated.
    """
    missing = [column for column in feature_columns if column not in fingerprints.columns]
    if missing:
        raise FeaturizationError(
            f"Fingerprint matrix is missing {len(missing)} training column(s), "
            f"e.g. {missing[:3]}; was the candidate matrix featurized with PaDEL?"
        )
    return fingerprints.loc[:, feature_columns]


def predict_records(
    bundle: TrainedModel, fingerprints: pd.DataFrame, active_threshold: float
) -> pd.DataFrame:
    """Run a fitted bundle over a fresh fingerprint matrix.

    Args:
        bundle: Fitted pipeline plus its training feature schema.
        fingerprints: Raw 881-bit PaDEL matrix, one row per candidate.
        active_threshold: pChEMBL cutoff for the regression activity call
            (``data.active_threshold`` from ``configs/``; injected here,
            never hardcoded).

    Returns:
        One row per candidate with task-specific columns: classification
        yields ``predicted_class`` (0/1) and ``probability`` (P(active));
        regression yields ``predicted_pchembl`` and ``predicted_class``
        (threshold rule mirroring ``dataset.label_activity``: active when
        ``pChEMBL >= active_threshold``).

    Raises:
        FeaturizationError: The matrix lacks the bundle's training columns.
    """
    X = align_features(fingerprints, bundle.feature_columns)
    if bundle.task is Task.CLASSIFICATION:
        classes = np.asarray(bundle.pipeline.predict(X), dtype=int)
        probabilities = np.asarray(bundle.pipeline.predict_proba(X))[:, 1].astype(float)
        return pd.DataFrame({PREDICTED_CLASS: classes, PROBABILITY: probabilities}, index=X.index)
    values = np.asarray(bundle.pipeline.predict(X), dtype=float)
    return pd.DataFrame(
        {
            PREDICTED_PCHEMBL: values,
            PREDICTED_CLASS: (values >= active_threshold).astype(int),
        },
        index=X.index,
    )
