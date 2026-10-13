"""Guard-rail tests: the target registry must match the thesis (Notebook 01).

These tests fail if someone reintroduces the incorrect ChEMBL accessions that
existed in the early draft (CHEMBL1908/CHEMBL204/CHEMBL203).
"""

from __future__ import annotations

from mldd.constants import (
    ALL_TARGETS,
    TARGET_BY_DATASET,
    UNIQUE_CHEMBL_IDS,
    Task,
)


def test_target_ids_match_thesis() -> None:
    """ChEMBL accessions verified against Notebook 01 outputs and README."""
    assert TARGET_BY_DATASET["AChE"].chembl_id == "CHEMBL220"
    assert TARGET_BY_DATASET["MAO-B"].chembl_id == "CHEMBL2039"
    assert TARGET_BY_DATASET["GSK3B"].chembl_id == "CHEMBL262"
    assert TARGET_BY_DATASET["LRRK2"].chembl_id == "CHEMBL1075104"


def test_six_datasets_four_unique_targets() -> None:
    """The pipeline produces 6 datasets derived from 4 unique ChEMBL targets."""
    assert len(ALL_TARGETS) == 6
    assert len(UNIQUE_CHEMBL_IDS) == 4


def test_lrrk2_variants_share_parent_target() -> None:
    """LRRK2, LRRK2_WT and LRRK2_G2019S derive from the same parent target."""
    lrrk2_ids = {TARGET_BY_DATASET[d].chembl_id for d in ("LRRK2", "LRRK2_WT", "LRRK2_G2019S")}
    assert lrrk2_ids == {"CHEMBL1075104"}


def test_variant_flags() -> None:
    """Only the split LRRK2 datasets are marked as variants."""
    variant_datasets = {t.dataset for t in ALL_TARGETS if t.is_variant}
    assert variant_datasets == {"LRRK2_WT", "LRRK2_G2019S"}


def test_task_enum_values() -> None:
    assert Task.CLASSIFICATION.value == "classification"
    assert Task.REGRESSION.value == "regression"
