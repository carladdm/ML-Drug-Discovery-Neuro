"""Batch inference boundary (DoD #3: prediction I/O lives here).

Loads the :class:`~mldd.modeling.predict.TrainedModel` bundle persisted by
the trainer, validates the candidates file against the Pandera input
contract, featurizes the SMILES with PaDEL (:mod:`mldd.io`), aligns the
881-bit matrix to the training feature schema and writes the prediction
table.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from pandera.errors import SchemaError

from mldd.config import AppConfig
from mldd.constants import MOLECULE_ID, SMILES
from mldd.exceptions import ModelNotFoundError, PredictionError
from mldd.io import compute_pubchem_fingerprints, load_raw_csv, save_raw_csv
from mldd.logging_utils import get_logger
from mldd.modeling.predict import CANDIDATE_SCHEMA, TrainedModel, predict_records

logger: logging.Logger = get_logger(__name__)


def load_model(model_path: Path) -> TrainedModel:
    """Load a trained inference bundle from disk.

    Args:
        model_path: Path of the ``.joblib`` artifact written by the trainer.

    Returns:
        The deserialized :class:`TrainedModel`.

    Raises:
        ModelNotFoundError: The artifact does not exist.
        PredictionError: The artifact exists but is not a ``TrainedModel``
            bundle (e.g. a bare pipeline persisted before Fase 6 — retrain).
    """
    if not model_path.is_file():
        raise ModelNotFoundError(f"Model artifact not found: {model_path}")
    artifact = joblib.load(model_path)
    if not isinstance(artifact, TrainedModel):
        raise PredictionError(
            f"Artifact {model_path} is a bare {type(artifact).__name__}, not a "
            "TrainedModel bundle; retrain with `mldd train` to regenerate it."
        )
    logger.info(
        "Loaded %s model '%s' (%d features) from %s.",
        artifact.task.value,
        artifact.algorithm,
        len(artifact.feature_columns),
        model_path,
    )
    return artifact


def _candidate_ids(candidates: pd.DataFrame) -> pd.Series:
    """Accessions for PaDEL alignment: ChEMBL ids when usable, else synthetic.

    Args:
        candidates: Validated candidate frame.

    Returns:
        Unique string accessions (one per row) — only used as the ``Name``
        written into PaDEL's ``.smi``; output rows always keep the original
        input columns.
    """
    if MOLECULE_ID in candidates.columns:
        ids = candidates[MOLECULE_ID].dropna().astype(str)
        if len(ids) == len(candidates) and ids.is_unique:
            return ids
    return pd.Series(
        [f"CMPD{position}" for position in range(len(candidates))], index=candidates.index
    )


def run_prediction(
    cfg: AppConfig, model_path: Path, input_csv: Path, output_csv: Path
) -> pd.DataFrame:
    """Batch inference: candidates CSV -> fingerprints -> prediction table.

    Args:
        cfg: Validated configuration (PaDEL settings and the
            ``data.active_threshold`` regression activity call).
        model_path: ``TrainedModel`` bundle written by the trainer.
        input_csv: Candidates file — ``canonical_smiles`` required,
            ``molecule_chembl_id`` optional, extra columns pass through.
        output_csv: Destination of the prediction table.

    Returns:
        The prediction table (input columns + task prediction columns).

    Raises:
        ModelNotFoundError: The model artifact is missing.
        PredictionError: The candidates file is missing or violates the
            input contract.
        FeaturizationError: PaDEL featurization failed (propagated).
    """
    bundle = load_model(model_path)
    if not input_csv.is_file():
        raise PredictionError(f"Candidates file not found: {input_csv}")
    candidates = load_raw_csv(input_csv)
    try:
        validated = CANDIDATE_SCHEMA.validate(candidates)
    except SchemaError as exc:
        raise PredictionError(
            f"Candidates file {input_csv} violates the input contract: {exc}"
        ) from exc

    fingerprints = compute_pubchem_fingerprints(
        validated[SMILES], _candidate_ids(validated), cfg.features
    )
    predictions = predict_records(bundle, fingerprints, cfg.data.active_threshold)
    output = pd.concat(
        [validated.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1
    )
    save_raw_csv(output, output_csv)
    logger.info(
        "Predicted %d candidates with the '%s' model -> %s.",
        len(output),
        bundle.algorithm,
        output_csv,
    )
    return output
