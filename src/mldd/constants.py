"""Canonical constants: target registry, task types and dataframe column names.

Single source of truth for the four neurodegenerative ChEMBL targets of the
thesis and the six derived datasets (LRRK2 appears unified, WT-only and
G2019S-mutant). ChEMBL accessions verified against Notebook 01 outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class Task(StrEnum):
    """ML task flavours supported by the pipeline."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass(frozen=True)
class TargetSpec:
    """Static description of a ChEMBL target dataset.

    Attributes:
        dataset: Dataset/file identifier used across the pipeline (e.g. ``AChE``).
        chembl_id: ChEMBL target accession (e.g. ``CHEMBL220``).
        name: Human-readable protein name.
        is_variant: True for datasets derived by splitting a parent target
            (LRRK2 WT and LRRK2 G2019S both derive from CHEMBL1075104).
    """

    dataset: str
    chembl_id: str
    name: str
    is_variant: bool = False


ACHE: Final = TargetSpec("AChE", "CHEMBL220", "Acetylcholinesterase")
MAOB: Final = TargetSpec("MAO-B", "CHEMBL2039", "Monoamine oxidase B")
GSK3B: Final = TargetSpec("GSK3B", "CHEMBL262", "Glycogen synthase kinase-3 beta")
LRRK2: Final = TargetSpec("LRRK2", "CHEMBL1075104", "Leucine-rich repeat kinase 2")
LRRK2_WT: Final = TargetSpec(
    "LRRK2_WT", "CHEMBL1075104", "Leucine-rich repeat kinase 2 (Wild Type)", is_variant=True
)
LRRK2_G2019S: Final = TargetSpec(
    "LRRK2_G2019S", "CHEMBL1075104", "Leucine-rich repeat kinase 2 (G2019S mutant)", is_variant=True
)

#: Every dataset the pipeline can produce (6 datasets, 4 unique ChEMBL targets).
ALL_TARGETS: Final[tuple[TargetSpec, ...]] = (ACHE, MAOB, GSK3B, LRRK2, LRRK2_WT, LRRK2_G2019S)

#: Fast lookup dataset name -> spec.
TARGET_BY_DATASET: Final[dict[str, TargetSpec]] = {t.dataset: t for t in ALL_TARGETS}

#: Unique ChEMBL accessions across all datasets.
UNIQUE_CHEMBL_IDS: Final[frozenset[str]] = frozenset(t.chembl_id for t in ALL_TARGETS)

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

# Columns kept by the curation stage (Notebook 02 contract)
CURATED_COLUMNS: Final[tuple[str, ...]] = (
    MOLECULE_ID,
    SMILES,
    PCHEMBL_VALUE,
    STANDARD_VALUE,
    STANDARD_UNITS,
)
