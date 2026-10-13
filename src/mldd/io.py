"""ChEMBL & PaDEL I/O boundary: raw activity ingestion and fingerprint runs.

Network calls, file writes and the PaDEL Java subprocess live here (DoD #3:
I/O only in ``cli``/``io``/``trainer``/``predictor``). All curation and
descriptor *science* is pure and lives in ``mldd.dataset``/``mldd.features``;
this module only fetches raw records, persists them and shells out to PaDEL.

The ChEMBL fetch is wrapped in a tenacity retry policy to survive occasional
rate-limiting and transient network drops (NB01 re-ran cells manually when the
API hiccuped; here that resilience is automatic).
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import padelpy
import pandas as pd
from chembl_webresource_client.new_client import new_client
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mldd.config import DataConfig, FeatureConfig, TargetConfig
from mldd.constants import (
    MOLECULE_ID,
    PUBCHEM_FP_COLUMNS,
    PUBCHEM_FP_NAME_COLUMN,
    SMILES,
    TARGET_ID,
)
from mldd.exceptions import FeaturizationError, IngestionError
from mldd.logging_utils import get_logger

logger: logging.Logger = get_logger(__name__)

#: Upper bound for a single target response; anything above is a filter
#: failure (e.g. ignored ``assay_type``) rather than real data.
_MAX_RECORDS_GUARD = 1_000_000

#: Retry policy for the ChEMBL transport (module-level so tests can relax it).
_RETRY_STOP = stop_after_attempt(5)
_RETRY_WAIT = wait_exponential(multiplier=2, min=2, max=60)


class ActivityQuery(Protocol):
    """Structural type of a chained ChEMBL activity query (real or test double)."""

    def filter(self, **kwargs: Any) -> ActivityQuery: ...

    def __iter__(self) -> Iterator[dict[str, Any]]: ...


class ActivityClient(Protocol):
    """Structural type of the ChEMBL ``activity`` client entry point."""

    def filter(self, **kwargs: Any) -> ActivityQuery: ...


def _query_activities(
    client: ActivityClient, target_chembl_id: str, data_config: DataConfig
) -> list[dict[str, Any]]:
    """Run the thesis acquisition query (NB01) with transient-failure retries.

    Applies ``standard_type`` (IC50), the configured ``assay_type`` set
    (Binding/Functional) and the exact ``standard_relation`` ('=') so only
    exact measurements enter the pipeline — bounds ('<', '>') never do.

    Only the transport is retried; emptiness/size validation lives in
    :func:`fetch_raw_activities` so non-transient problems are not retried.

    Returns:
        The raw activity records in ChEMBL API shape.

    Raises:
        IngestionError: Transport failed after all retry attempts.
    """
    retrying = Retrying(
        retry=retry_if_exception_type(Exception),
        stop=_RETRY_STOP,
        wait=_RETRY_WAIT,
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )

    def _call() -> list[dict[str, Any]]:
        query = (
            client.filter(target_chembl_id=target_chembl_id)
            .filter(standard_type=data_config.standard_type)
            .filter(assay_type__in=list(data_config.assay_types))
            .filter(standard_relation=data_config.standard_relation)
        )
        return list(query)

    try:
        return retrying(_call)
    except Exception as exc:
        raise IngestionError(f"ChEMBL query failed for {target_chembl_id}: {exc}") from exc


def fetch_raw_activities(
    target_config: TargetConfig, data_config: DataConfig, *, client: ActivityClient | None = None
) -> pd.DataFrame:
    """Download the raw bioactivity records for one configured target.

    Args:
        target_config: The target to fetch (ChEMBL accession, organism, type).
        data_config: Acquisition filters (standard type, assay types, relation).
        client: Optional client override for tests; defaults to the official
            ChEMBL client.

    Returns:
        Raw ChEMBL records plus a ``target_chembl_id`` column.

    Raises:
        IngestionError: Fetching failed after retries, returned no records, or
            yielded an implausibly large response.
    """
    activity: ActivityClient = new_client.activity if client is None else client
    logger.info(
        "Fetching %s records for %s (%s)...",
        data_config.standard_type,
        target_config.dataset,
        target_config.chembl_id,
    )
    records = _query_activities(activity, target_config.chembl_id, data_config)

    if not records:
        raise IngestionError(
            f"No {data_config.standard_type} records for {target_config.chembl_id}."
        )
    if len(records) > _MAX_RECORDS_GUARD:
        raise IngestionError(
            f"Implausible record count ({len(records)}) for {target_config.chembl_id}; "
            "the API filters likely did not apply."
        )

    df = pd.DataFrame(records)
    df[TARGET_ID] = target_config.chembl_id
    logger.info("Fetched %d raw records for %s.", len(df), target_config.dataset)
    return df


def save_raw_csv(df: pd.DataFrame, path: Path) -> Path:
    """Persist a dataframe to CSV, creating parent directories.

    Args:
        df: The dataframe to write.
        path: Destination file path.

    Returns:
        The path written (for logging / manifest use).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote %d rows to %s.", len(df), path)
    return path


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Load a raw CSV from disk.

    Args:
        path: Source file path.

    Returns:
        The loaded dataframe.

    Raises:
        IngestionError: The file does not exist.
    """
    if not path.is_file():
        raise IngestionError(f"Raw data file not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s.", len(df), path)
    return df


def compute_pubchem_fingerprints(
    smiles: pd.Series, molecule_ids: pd.Series, feature_config: FeatureConfig
) -> pd.DataFrame:
    """Compute the PubChem 881-bit fingerprint matrix via PaDEL (NB04).

    Writes the molecules to a temporary ``.smi`` file, shells out to the
    PaDEL-Descriptor bundled with ``padelpy`` and returns the bit matrix. The
    run reproduces the notebook's science: ``removesalt`` and
    ``standardizenitro`` are enabled and ``retainorder`` guarantees the output
    rows keep the input order, so the matrix aligns positionally with the
    caller's dataframe. NB04 feeds the raw ``canonical_smiles`` here — PaDEL
    re-standardizes internally, and the notebook's ``PubchemFingerprinter.xml``
    produces the exact same bits as the bundled ``-fingerprints`` flag
    (verified bit-for-bit on 50 real MAO-B molecules).

    Args:
        smiles: SMILES to featurize, one per row.
        molecule_ids: Molecule accessions written as the ``.smi`` names.
        feature_config: Featurization settings (only the subprocess cap is
            consumed here; the bit layout is a fixed PubChem contract).

    Returns:
        The 881-bit ``PubchemFP*`` matrix, one row per input molecule, in input
        order, without the ``Name`` column.

    Raises:
        FeaturizationError: Empty or ragged input, a null SMILES (which PaDEL
            would silently drop, misaligning the matrix), a failed/absent PaDEL
            run, or an output that violates the fingerprint column contract.
    """
    if len(smiles) != len(molecule_ids):
        raise FeaturizationError(
            f"SMILES/ID length mismatch: {len(smiles)} vs {len(molecule_ids)}."
        )
    if smiles.empty:
        raise FeaturizationError("No molecules to featurize.")
    if smiles.isna().any():
        raise FeaturizationError(
            "Null SMILES would be dropped by PaDEL and silently misalign the "
            "fingerprint matrix; drop them before featurizing."
        )

    with tempfile.TemporaryDirectory(prefix="mldd_padel_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        smi_file = tmp_path / "molecules.smi"
        pd.DataFrame({SMILES: smiles.to_numpy(), MOLECULE_ID: molecule_ids.to_numpy()}).to_csv(
            smi_file, sep="\t", index=False, header=False
        )

        output_file = tmp_path / "pubchem_fp.csv"
        logger.info("Running PaDEL PubChem fingerprinting for %d molecules...", len(smiles))
        try:
            padelpy.padeldescriptor(
                mol_dir=str(smi_file),
                d_file=str(output_file),
                fingerprints=True,
                removesalt=True,
                standardizenitro=True,
                retainorder=True,
                sp_timeout=feature_config.padel_timeout,
            )
        except (ReferenceError, RuntimeError) as exc:
            raise FeaturizationError(f"PaDEL fingerprint run failed: {exc}") from exc

        if not output_file.is_file():
            raise FeaturizationError("PaDEL completed without producing a descriptor file.")
        raw = pd.read_csv(output_file)

    missing = [column for column in PUBCHEM_FP_COLUMNS if column not in raw.columns]
    if missing or PUBCHEM_FP_NAME_COLUMN not in raw.columns:
        raise FeaturizationError(
            f"PaDEL output violates the PubChem contract; missing columns: {missing[:5]}"
        )
    if len(raw) != len(smiles):
        raise FeaturizationError(
            f"PaDEL returned {len(raw)} rows for {len(smiles)} molecules; "
            "the fingerprint matrix would misalign with the dataset."
        )

    logger.info(
        "Computed %d PubChem fingerprint bits for %d molecules.", len(PUBCHEM_FP_COLUMNS), len(raw)
    )
    return pd.DataFrame(
        raw[list(PUBCHEM_FP_COLUMNS)].to_numpy(),
        columns=list(PUBCHEM_FP_COLUMNS),
    )
