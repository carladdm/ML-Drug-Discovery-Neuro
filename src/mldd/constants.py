"""Canonical constants: task types and dataframe column contracts.

The target registry (ChEMBL accessions, variant-split regex) no longer lives
here: since Fase 1 it is owned by ``configs/target/*.yaml`` and exposed via
``mldd.config.load_config`` (single source of truth, AGENTS.md).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class Task(StrEnum):
    """ML task flavours supported by the pipeline."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


# ------------------------------------------------------------------------------
# Dataframe column names (data contract across pipeline stages)
# ------------------------------------------------------------------------------
MOLECULE_ID: Final = "molecule_chembl_id"
TARGET_ID: Final = "target_chembl_id"
SMILES: Final = "canonical_smiles"
SMILES_CLEAN: Final = "smiles_clean"
STANDARD_VALUE: Final = "standard_value"
STANDARD_UNITS: Final = "standard_units"
STANDARD_RELATION: Final = "standard_relation"
PCHEMBL_VALUE: Final = "pchembl_value"
BIOACTIVITY_CLASS: Final = "bioactivity_class"

# Lipinski physicochemical descriptors
MW: Final = "MW"
LOGP: Final = "LogP"
NUM_H_DONORS: Final = "NumHDonors"
NUM_H_ACCEPTORS: Final = "NumHAcceptors"
LIPINSKI_DESCRIPTORS: Final[tuple[str, ...]] = (MW, LOGP, NUM_H_DONORS, NUM_H_ACCEPTORS)

#: Descriptors scanned for extreme outliers (NB03 remove_outliers_and_visualize)
ZSCORE_DESCRIPTORS: Final[tuple[str, ...]] = (
    PCHEMBL_VALUE,
    MW,
    LOGP,
    NUM_H_DONORS,
    NUM_H_ACCEPTORS,
)

# PubChem fingerprint contract (PaDEL-Descriptor, PubChemFP0..PubChemFP880)
PUBCHEM_FP_BITS: Final = 881
PUBCHEM_FP_NAME_COLUMN: Final = "Name"
PUBCHEM_FP_PREFIX: Final = "PubchemFP"
PUBCHEM_FP_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"{PUBCHEM_FP_PREFIX}{index}" for index in range(PUBCHEM_FP_BITS)
)

# ChEMBL raw-record columns consumed by the curation stage (NB01/NB02)
ASSAY_DESCRIPTION: Final = "assay_description"
VARIANT_SEQUENCE: Final = "variant_sequence"
VARIANT_TYPE: Final = "variant_type"
INCHI_KEY: Final = "inchi_key"

# Columns kept by the curation stage (Notebook 02 contract)
CURATED_COLUMNS: Final[tuple[str, ...]] = (
    MOLECULE_ID,
    SMILES,
    PCHEMBL_VALUE,
    STANDARD_VALUE,
    STANDARD_UNITS,
)
