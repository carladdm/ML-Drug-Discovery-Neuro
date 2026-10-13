"""Molecular standardization, descriptors and filtering (NB03/NB04/NB05).

Pure functions only (DoD #3): no I/O lives here. Writing the temporary ``.smi``
file and running the PaDEL Java subprocess is ``mldd.io``'s job; the CLI wires
both worlds together. Every function takes a dataframe plus validated config
objects and returns a new dataframe.

The pipeline mirrors the notebooks end-to-end::

    prepare_physchem_features   -> smiles_clean + MW/LogP/HBD/HBA (NB03 step 1)
    remove_zscore_outliers       -> |z| > threshold, single pass (NB03 step 4)
    [io.compute_pubchem_fingerprints] -> PubChem 881-bit matrix (NB04)
    filter_low_variance (pre)    -> drop zero-variance bits (NB04)
    merge_fingerprints_and_targets -> X + bioactivity_class + pchembl (NB04)
    filter_low_variance (post)   -> second pass at training time (NB05)

Immutable domain rules (AGENTS.md / MEMORY.md section 2):
    * Structures are standardized with RDKit (LargestFragmentChooser +
      Uncharger); descriptors are computed on the standardized SMILES.
    * Outliers are removed with the Z-score over the five physicochemical
      descriptors, in a SINGLE pass over the original mean/std (values are not
      recomputed after removals).
    * Variance thresholds and the Z-score cut-off come from
      ``configs/features/physchem.yaml`` — never from literals in this module.
"""

from __future__ import annotations

import logging

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.feature_selection import VarianceThreshold

from mldd.config import FeatureConfig
from mldd.constants import (
    LOGP,
    MW,
    NUM_H_ACCEPTORS,
    NUM_H_DONORS,
    SMILES,
    SMILES_CLEAN,
    STANDARD_UNITS,
    STANDARD_VALUE,
    ZSCORE_DESCRIPTORS,
)
from mldd.exceptions import FeaturizationError
from mldd.logging_utils import get_logger

logger: logging.Logger = get_logger(__name__)


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], context: str) -> None:
    """Raise :class:`FeaturizationError` if ``df`` is missing any required column.

    Args:
        df: Frame to inspect.
        required: Column names the stage needs.
        context: Human-readable stage name for the error message.

    Raises:
        FeaturizationError: At least one column is absent.
    """
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise FeaturizationError(
            f"{context} requires columns {sorted(missing)}; got {list(df.columns)}"
        )


def _lipinski_of(smiles: str | None) -> tuple[float, float, int, int] | None:
    """Compute the four Lipinski descriptors for one SMILES string.

    Args:
        smiles: Standardized SMILES string, or ``None``.

    Returns:
        ``(MW, LogP, NumHDonors, NumHAcceptors)``, or ``None`` when the
        structure cannot be parsed by RDKit.
    """
    if not isinstance(smiles, str):  # None / pandas NaN from a missing value
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except (ValueError, RuntimeError):
        return None
    if mol is None:
        return None
    return (
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
    )


def standardize_smiles(smiles: str | None) -> str | None:
    """Clean one structure: largest fragment + charge neutralization (NB03).

    Args:
        smiles: Raw SMILES string (possibly a salt or charged species).

    Returns:
        The canonical SMILES of the neutralized largest fragment, or ``None``
        when RDKit cannot parse the input.
    """
    if not isinstance(smiles, str):  # None / pandas NaN from a missing value
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        largest = rdMolStandardize.LargestFragmentChooser().choose(mol)
        neutral = rdMolStandardize.Uncharger().uncharge(largest)
        canonical: str = Chem.MolToSmiles(neutral)
        return canonical
    except (ValueError, RuntimeError):
        return None


def add_standardized_smiles(
    df: pd.DataFrame, *, smiles_col: str = SMILES, out_col: str = SMILES_CLEAN
) -> pd.DataFrame:
    """Standardize every structure into ``smiles_clean`` (NB03 cell 32).

    Args:
        df: Input dataframe containing a SMILES column.
        smiles_col: Name of the raw SMILES column to standardize.
        out_col: Name of the output column for the standardized SMILES.

    Returns:
        A copy of ``df`` with the ``smiles_clean`` column appended; rows whose
        structure cannot be standardized keep a null entry.

    Raises:
        FeaturizationError: A required column is missing.
    """
    _require_columns(df, (smiles_col,), "add_standardized_smiles")
    out = df.copy()
    out[out_col] = [standardize_smiles(smiles) for smiles in out[smiles_col]]
    logger.info("Standardized %d structures into '%s'.", len(out), out_col)
    return out


def compute_lipinski_descriptors(
    df: pd.DataFrame, *, smiles_col: str = SMILES_CLEAN
) -> pd.DataFrame:
    """Compute MW, LogP, HBD and HBA over the standardized structures (NB03).

    Unlike the notebook (which appends NaN rows for unparseable structures),
    rows that cannot be featurized are dropped: a NaN descriptor must never
    reach a model matrix. In the thesis datasets this never fires — curated
    structures always parse.

    Args:
        df: Input dataframe containing a SMILES column.
        smiles_col: Name of the SMILES column to featurize
            (``smiles_clean`` by default, per NB03).

    Returns:
        A copy of ``df`` with the four ``LIPINSKI_DESCRIPTORS`` columns, without
        unparseable rows and with a reset index.

    Raises:
        FeaturizationError: A required column is missing.
    """
    _require_columns(df, (smiles_col,), "compute_lipinski_descriptors")
    computed = [_lipinski_of(smiles) for smiles in df[smiles_col]]
    keep_mask = [descriptors is not None for descriptors in computed]
    if not all(keep_mask):
        logger.info(
            "Dropping %d rows with unparseable '%s' structures.",
            keep_mask.count(False),
            smiles_col,
        )

    out = df[keep_mask].copy()
    values = [descriptors for descriptors in computed if descriptors is not None]
    out[MW] = [descriptors[0] for descriptors in values]
    out[LOGP] = [descriptors[1] for descriptors in values]
    out[NUM_H_DONORS] = [descriptors[2] for descriptors in values]
    out[NUM_H_ACCEPTORS] = [descriptors[3] for descriptors in values]
    logger.info("Lipinski descriptors computed for %d/%d molecules.", len(out), len(df))
    return out.reset_index(drop=True)


def prepare_physchem_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize structures, compute Lipinski and drop curation artifacts.

    This is Notebook 03 step 1 end-to-end: ``canonical_smiles`` becomes
    ``smiles_clean``, the four descriptors are appended, and the raw-measurement
    columns (``standard_value``/``standard_units``) are dropped — they were
    only needed to rescue pChEMBL during curation.

    Args:
        df: Curated records (``CURATED_COLUMNS`` plus ``bioactivity_class``).

    Returns:
        The physicochemical feature frame ready for outlier removal.

    Raises:
        FeaturizationError: A required column is missing.
    """
    _require_columns(df, (SMILES,), "prepare_physchem_features")
    standardized = add_standardized_smiles(df)
    with_descriptors = compute_lipinski_descriptors(standardized)
    artifacts = [
        column for column in (STANDARD_VALUE, STANDARD_UNITS) if column in with_descriptors
    ]
    if artifacts:
        with_descriptors = with_descriptors.drop(columns=artifacts)
        logger.info("Dropped curation artifact columns: %s.", artifacts)
    return with_descriptors.reset_index(drop=True)


def remove_zscore_outliers(df: pd.DataFrame, feature_config: FeatureConfig) -> pd.DataFrame:
    """Drop extreme rows with the Z-score over five descriptors (NB03 cell 30).

    The notebook computes the mean/std of each descriptor ONCE over the whole
    frame and OR-combines the ``|z| > threshold`` masks into a single pass:
    statistics are never recomputed after removals, so a row can only be an
    outlier relative to the *original* distribution. Descriptors with zero
    variance are skipped (their z-score is undefined). ``pchembl_value`` takes
    part alongside the four Lipinski descriptors, exactly as in the thesis.

    Args:
        df: Frame with ``ZSCORE_DESCRIPTORS`` columns present.
        feature_config: Supplies ``zscore_threshold`` (from configs/, never
            hardcoded here).

    Returns:
        The frame without outlier rows, with a reset index and unchanged
        columns.

    Raises:
        FeaturizationError: A required descriptor column is missing.
    """
    _require_columns(df, ZSCORE_DESCRIPTORS, "remove_zscore_outliers")
    outlier_mask = pd.Series(False, index=df.index)
    for descriptor in ZSCORE_DESCRIPTORS:
        values = pd.to_numeric(df[descriptor], errors="coerce")
        std = values.std()
        if std > 0:
            zscore = (values - values.mean()) / std
            outlier_mask = outlier_mask | (zscore.abs() > feature_config.zscore_threshold)
        else:
            logger.info("Descriptor '%s' has zero variance; skipping Z-score.", descriptor)

    kept = df[~outlier_mask].copy().reset_index(drop=True)
    logger.info(
        "Z-score outliers removed: %d of %d rows (|z| > %s).",
        len(df) - len(kept),
        len(df),
        feature_config.zscore_threshold,
    )
    return kept


def filter_low_variance(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop features whose variance does not exceed ``threshold`` (NB04/NB05).

    This is the sklearn ``VarianceThreshold`` used verbatim by the notebooks:
    a feature survives only when its variance is *strictly greater* than the
    threshold. The pre-process pass uses ``variance_threshold_pre`` on the raw
    PaDEL bits (NB04); the second pass uses ``variance_threshold_post`` on the
    merged matrix (NB05, ``0.8 * (1 - 0.8)``). Both values live in
    ``configs/features/physchem.yaml``.

    Args:
        df: Feature matrix (fingerprint bits or merged X+targets).
        threshold: Minimum exclusive variance, from the composed config.

    Returns:
        The filtered matrix with only the surviving feature columns, in order.

    Raises:
        FeaturizationError: The matrix is empty, or the threshold would remove
            every feature (a misconfigured threshold, never a silent empty X).
    """
    if df.empty:
        raise FeaturizationError("Variance filter received an empty feature matrix.")
    selector = VarianceThreshold(threshold=threshold)
    try:
        filtered = selector.fit_transform(df)
    except ValueError as exc:  # sklearn: no feature meets the threshold
        raise FeaturizationError(
            f"Variance threshold {threshold} removed every feature: {exc}"
        ) from exc

    kept = list(df.columns[selector.get_support(indices=True)])
    logger.info(
        "Variance filter (>%s): kept %d of %d features (%d removed).",
        threshold,
        len(kept),
        df.shape[1],
        df.shape[1] - len(kept),
    )
    return pd.DataFrame(filtered, columns=kept)


def merge_fingerprints_and_targets(
    fingerprints: pd.DataFrame, targets: pd.DataFrame
) -> pd.DataFrame:
    """Horizontally merge the feature matrix with the target columns (NB04 step 5).

    Both frames are joined positionally, so their indexes are reset first —
    relying on shared labels is what silently misaligns X and y in the legacy
    notebooks.

    Args:
        fingerprints: The (filtered) fingerprint/descriptor matrix.
        targets: The target columns (``bioactivity_class``, ``pchembl_value``).

    Returns:
        The merged modelling matrix: features first, then targets, row-aligned.

    Raises:
        FeaturizationError: The two frames have different lengths.
    """
    if len(fingerprints) != len(targets):
        raise FeaturizationError(
            f"Feature/target length mismatch: {len(fingerprints)} vs {len(targets)}."
        )
    merged = pd.concat(
        [fingerprints.reset_index(drop=True), targets.reset_index(drop=True)], axis=1
    )
    logger.info("Merged feature matrix: %d rows x %d columns.", len(merged), merged.shape[1])
    return merged


__all__ = [
    "add_standardized_smiles",
    "compute_lipinski_descriptors",
    "filter_low_variance",
    "merge_fingerprints_and_targets",
    "prepare_physchem_features",
    "remove_zscore_outliers",
    "standardize_smiles",
]
