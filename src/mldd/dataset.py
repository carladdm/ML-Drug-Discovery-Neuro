"""Bioactivity curation: pure functions implementing the thesis science (NB01/NB02).

No I/O lives here (DoD #3): fetching raw records is ``mldd.io``'s job, CSV
writes are too, and the CLI wires them together. Every function below takes a
dataframe plus validated config objects and returns a new dataframe.

The pipeline mirrors Notebook 02 end-to-end:
    drop_missing_structures -> rescue_pchembl -> deduplicate_by_inchikey
    -> select_curated_columns -> label_activity

Immutable domain rules (AGENTS.md / MEMORY.md section 2):
    * pChEMBL rescue only for exact measurements (``standard_relation == '='``)
      and only unit-aware: nM -> ``9 - log10(v)``, uM-class -> ``6 - log10(v)``.
      Ignoring units shifts values by three orders of magnitude.
    * Deduplicate by InChIKey keeping the consensus mean, and drop molecules
      whose per-InChIKey pChEMBL spread exceeds ``dedup_max_spread``.
    * Classification threshold: active >= ``active_threshold``, inactive below
      it. There is NO gray zone: values in (5, 6) are labelled inactive, never
      dropped.
    * LRRK2 WT/G2019S split via strict regex over ``assay_description``.
"""

from __future__ import annotations

import logging
import re
from typing import NamedTuple

import numpy as np
import pandas as pd
from rdkit import Chem

from mldd.config import DataConfig, VariantSplitConfig
from mldd.constants import (
    ASSAY_DESCRIPTION,
    BIOACTIVITY_CLASS,
    CURATED_COLUMNS,
    INCHI_KEY,
    PCHEMBL_VALUE,
    SMILES,
    STANDARD_RELATION,
    STANDARD_UNITS,
    STANDARD_VALUE,
    VARIANT_SEQUENCE,
)
from mldd.exceptions import CurationError
from mldd.logging_utils import get_logger

logger: logging.Logger = get_logger(__name__)

#: Temporary column used to carry the InChIKey consensus mean across the merge.
_CONSENSUS_COLUMN = "_consensus_mean"


class VariantFrames(NamedTuple):
    """The two datasets derived from the LRRK2 parent target."""

    wild_type: pd.DataFrame
    mutant: pd.DataFrame


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], context: str) -> None:
    """Raise :class:`CurationError` if ``df`` is missing any required column.

    Args:
        df: Frame to inspect.
        required: Column names the stage needs.
        context: Human-readable stage name for the error message.

    Raises:
        CurationError: At least one column is absent.
    """
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise CurationError(f"{context} requires columns {sorted(missing)}; got {list(df.columns)}")


def _inchikey(smiles: str) -> str | None:
    """Compute the InChIKey for one SMILES string (NB02 canonicalization).

    Args:
        smiles: The SMILES string to canonicalize and hash.

    Returns:
        The InChIKey, or ``None`` when RDKit cannot parse the molecule.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol, canonical=True)
        key: str | None = Chem.MolToInchiKey(Chem.MolFromSmiles(canonical))
        return key or None
    except (ValueError, RuntimeError):
        return None


def drop_missing_structures(df: pd.DataFrame) -> pd.DataFrame:
    """Drop records lacking a measurement or a structure (NB02 step 1).

    Args:
        df: Raw ChEMBL records.

    Returns:
        A frame without nulls in ``standard_value`` or ``canonical_smiles``.

    Raises:
        CurationError: A required column is missing.
    """
    _require_columns(df, (STANDARD_VALUE, SMILES), "drop_missing_structures")
    cleaned = df.dropna(subset=[STANDARD_VALUE, SMILES]).copy()
    logger.info("Records after dropping null value/structure: %d", len(cleaned))
    return cleaned


def rescue_pchembl(df: pd.DataFrame, data_config: DataConfig) -> pd.DataFrame:
    """Fill ``pchembl_value`` from ``standard_value`` where it is absent (NB02).

    Existing non-null ``pchembl_value`` entries are kept verbatim. Missing
    entries are computed as ``offset - log10(standard_value)`` using the unit
    offset from the config, and only when the measurement is exact
    (``standard_relation == '='``) with a recognised unit and a positive value.
    Everything else is dropped: a bound ('<', '>') is not an exact measurement.

    Args:
        df: Records with (optionally) ``pchembl_value``.
        data_config: Unit offsets and relation gate.

    Returns:
        Records with a non-null ``pchembl_value``.

    Raises:
        CurationError: A required column is missing.
    """
    _require_columns(df, (STANDARD_VALUE, STANDARD_UNITS, STANDARD_RELATION), "rescue_pchembl")
    out = df.copy()

    value = pd.to_numeric(out[STANDARD_VALUE], errors="coerce")
    units = out[STANDARD_UNITS].astype("string").str.strip()
    offsets = units.map(data_config.rescue_unit_offsets)

    exact = out[STANDARD_RELATION] == data_config.standard_relation
    eligible = exact & value.notna() & (value > 0) & offsets.notna()

    rescued = pd.Series(np.nan, index=out.index, dtype="float64")
    rescued[eligible] = offsets[eligible] - np.log10(value[eligible])

    existing = (
        pd.to_numeric(out[PCHEMBL_VALUE], errors="coerce")
        if PCHEMBL_VALUE in out.columns
        else pd.Series(np.nan, index=out.index, dtype="float64")
    )
    out[PCHEMBL_VALUE] = existing.where(existing.notna(), rescued)

    cleaned = out.dropna(subset=[PCHEMBL_VALUE]).copy()
    logger.info("Records after pChEMBL rescue: %d (from %d)", len(cleaned), len(out))
    return cleaned


def deduplicate_by_inchikey(df: pd.DataFrame, data_config: DataConfig) -> pd.DataFrame:
    """Collapse replicate measurements per InChIKey (NB02 step 3).

    Molecules are keyed by InChIKey (RDKit canonicalisation), so salts and
    differing tautomer spellings of the same compound merge together. Each
    surviving molecule takes the consensus mean of its measurements; groups
    whose pChEMBL spread exceeds ``dedup_max_spread`` are inconsistent assays
    and are dropped entirely.

    Args:
        df: Records with a non-null ``pchembl_value``.
        data_config: Curation rules (``dedup_max_spread``).

    Returns:
        One record per InChIKey with the consensus pChEMBL value.

    Raises:
        CurationError: A required column is missing.
    """
    _require_columns(df, (SMILES, PCHEMBL_VALUE), "deduplicate_by_inchikey")
    out = df.copy()
    out[INCHI_KEY] = [_inchikey(smiles) for smiles in out[SMILES]]
    out = out.dropna(subset=[INCHI_KEY])

    stats = out.groupby(INCHI_KEY)[PCHEMBL_VALUE].agg(["mean", "min", "max"])
    spread = stats["max"] - stats["min"]
    valid_keys = spread[spread <= data_config.dedup_max_spread].index
    logger.info(
        "InChIKey groups: %d total, %d dropped for spread > %s",
        len(stats),
        len(stats) - len(valid_keys),
        data_config.dedup_max_spread,
    )

    deduped = out.drop_duplicates(subset=[INCHI_KEY], keep="first")
    deduped = deduped[deduped[INCHI_KEY].isin(valid_keys)].copy()
    deduped[PCHEMBL_VALUE] = deduped[INCHI_KEY].map(stats["mean"])
    logger.info("Records after InChIKey deduplication: %d (from %d)", len(deduped), len(out))
    return deduped.reset_index(drop=True)


def select_curated_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the curated data contract columns (NB02 step 4).

    Args:
        df: Curated records.

    Returns:
        Only the ``CURATED_COLUMNS`` of the contract.

    Raises:
        CurationError: A required column is missing.
    """
    _require_columns(df, CURATED_COLUMNS, "select_curated_columns")
    return df[list(CURATED_COLUMNS)].copy()


def label_activity(df: pd.DataFrame, data_config: DataConfig) -> pd.DataFrame:
    """Label actives/inactives at the configured threshold (NB02 step 5).

    Actives are ``pchembl_value >= active_threshold``; everything below is
    inactive. There is no gray zone: intermediate values are labelled, never
    dropped (this is the rule the legacy draft violated).

    Args:
        df: Records with a non-null ``pchembl_value``.
        data_config: Curation rules (``active_threshold``).

    Returns:
        Records with an integer ``bioactivity_class`` column (1 active, 0 inactive).

    Raises:
        CurationError: A required column is missing, or ``pchembl_value`` still
            has nulls (the caller skipped :func:`rescue_pchembl`).
    """
    _require_columns(df, (PCHEMBL_VALUE,), "label_activity")
    if df[PCHEMBL_VALUE].isna().any():
        raise CurationError("label_activity requires a non-null pchembl_value; run rescue_pchembl")

    out = df.copy()
    out[BIOACTIVITY_CLASS] = (out[PCHEMBL_VALUE] >= data_config.active_threshold).astype(int)
    logger.info(
        "Labelled %d actives / %d inactives at threshold %s",
        int(out[BIOACTIVITY_CLASS].sum()),
        int((out[BIOACTIVITY_CLASS] == 0).sum()),
        data_config.active_threshold,
    )
    return out


def curate_bioactivity(df: pd.DataFrame, data_config: DataConfig) -> pd.DataFrame:
    """Run the full NB02 curation chain on one target's raw records.

    Args:
        df: Raw ChEMBL records for a single dataset.
        data_config: Curation rules from the composed config.

    Returns:
        The curated dataframe (``CURATED_COLUMNS`` plus ``bioactivity_class``).

    Raises:
        CurationError: A required column is missing at any stage.
    """
    logger.info("Curating %d raw records...", len(df))
    step1 = drop_missing_structures(df)
    step2 = rescue_pchembl(step1, data_config)
    step3 = deduplicate_by_inchikey(step2, data_config)
    step4 = select_curated_columns(step3)
    step5 = label_activity(step4, data_config)
    logger.info("Curation finished: %d curated records", len(step5))
    return step5


def split_lrrk2_variants(df: pd.DataFrame, variant_split: VariantSplitConfig) -> VariantFrames:
    """Split LRRK2 records into Wild Type / G2019S mutant sets (NB01 cell 18).

    A record is mutant when the assay carries an explicit variant sequence, or
    when its (uppercased) ``assay_description`` matches the inclusion regex
    without matching the exclusion regex — the exclusion list avoids reading a
    selectivity-against-mutant assay as a mutant-assay measurement.

    Args:
        df: Raw LRRK2 records (parent target).
        variant_split: Configured regexes and output labels.

    Returns:
        A :class:`VariantFrames` namedtuple ``(wild_type, mutant)``.

    Raises:
        CurationError: A required column is missing.
    """
    _require_columns(df, (ASSAY_DESCRIPTION,), "split_lrrk2_variants")
    inclusion = re.compile(variant_split.inclusion_pattern, re.IGNORECASE)
    exclusion = re.compile(variant_split.exclusion_pattern, re.IGNORECASE)

    description = df[ASSAY_DESCRIPTION].astype("string").fillna("")
    included = description.str.contains(inclusion).fillna(False).astype(bool)
    excluded = description.str.contains(exclusion).fillna(False).astype(bool)

    if VARIANT_SEQUENCE in df.columns:
        has_variant_sequence = df[VARIANT_SEQUENCE].notna()
    else:
        has_variant_sequence = pd.Series(False, index=df.index)

    is_mutant = has_variant_sequence | (included & ~excluded)
    frames = VariantFrames(
        wild_type=df[~is_mutant].copy(),
        mutant=df[is_mutant].copy(),
    )
    logger.info(
        "LRRK2 split: %d wild-type / %d mutant records", len(frames.wild_type), len(frames.mutant)
    )
    return frames


__all__ = [
    "VariantFrames",
    "curate_bioactivity",
    "deduplicate_by_inchikey",
    "drop_missing_structures",
    "label_activity",
    "rescue_pchembl",
    "select_curated_columns",
    "split_lrrk2_variants",
]
