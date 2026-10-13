"""Scientific-contract tests: the curation rules from AGENTS.md / MEMORY.md.

These were frozen as *strict xfails* while the legacy draft in ``mldd.dataset``
still implemented the incorrect academic logic. Fase 2 rewrote the module as
pure functions, so the markers are gone and the rules are now enforced: the
suite fails if anyone reintroduces the gray zone, the nM-only formula or the
missing relation gate.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mldd.config import load_config
from mldd.constants import (
    BIOACTIVITY_CLASS,
    MOLECULE_ID,
    PCHEMBL_VALUE,
    SMILES,
    STANDARD_RELATION,
    STANDARD_UNITS,
    STANDARD_VALUE,
    TARGET_ID,
)
from mldd.dataset import curate_bioactivity

_DATA_CONFIG = load_config(["target=maob"]).data


def _record(pic50: float) -> dict[str, object]:
    """Build a ChEMBL-like record whose exact nM IC50 yields ``pic50``."""
    return {
        MOLECULE_ID: "TEST_MOL",
        TARGET_ID: "CHEMBL2039",
        SMILES: "CCO",
        STANDARD_VALUE: 10 ** (9 - pic50),
        STANDARD_UNITS: "nM",
        STANDARD_RELATION: "=",
    }


def _run(records: list[dict[str, object]]) -> pd.DataFrame:
    return curate_bioactivity(pd.DataFrame(records), _DATA_CONFIG)


def test_threshold_exact_six_is_active() -> None:
    """pChEMBL == 6.0 is ACTIVE (inclusive boundary)."""
    result = _run([_record(6.0)])
    assert result[BIOACTIVITY_CLASS].tolist() == [1]
    assert result[PCHEMBL_VALUE].iloc[0] == pytest.approx(6.0)


def test_threshold_five_is_inactive() -> None:
    """pChEMBL == 5.0 is INACTIVE."""
    result = _run([_record(5.0)])
    assert result[BIOACTIVITY_CLASS].tolist() == [0]


@pytest.mark.parametrize("pic50", [5.5, 5.99])
def test_gray_zone_must_be_labeled_inactive(pic50: float) -> None:
    """Values in (5, 6) are INACTIVE, never dropped (MEMORY.md section 2)."""
    result = _run([_record(pic50)])
    assert result[BIOACTIVITY_CLASS].tolist() == [0]


def test_micromolar_value_is_rescued_with_six_minus_log10() -> None:
    """1 uM IC50 -> pChEMBL 6.0 (active); the nM-only formula would yield 9.0."""
    record = {
        MOLECULE_ID: "TEST_MOL",
        TARGET_ID: "CHEMBL2039",
        SMILES: "CCO",
        STANDARD_VALUE: 1.0,
        STANDARD_UNITS: "uM",
        STANDARD_RELATION: "=",
    }
    result = _run([record])
    assert result[PCHEMBL_VALUE].iloc[0] == pytest.approx(6.0)
    assert result[BIOACTIVITY_CLASS].tolist() == [1]


def test_non_equality_relation_is_excluded() -> None:
    """A '<'/'>' bound is not an exact measurement: the record must be dropped."""
    record = _record(7.0)
    record[STANDARD_RELATION] = ">"
    result = _run([record])
    assert result.empty
