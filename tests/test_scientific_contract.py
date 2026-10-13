"""Scientific-contract freeze tests (TDD anchor for Fase 2 / T2.2-T2.3).

AGENTS.md and MEMORY.md define the immutable rules: active pChEMBL >= 6,
inactive < 6 with **no gray zone** (5-6 values are NOT excluded), unit-aware
pChEMBL rescue (nM -> 9-log10(v), uM -> 6-log10(v)) gated on
``standard_relation == '='``.

The legacy draft in ``mldd.dataset`` still implements the incorrect academic
logic, so the rules it violates are frozen here as *strict xfails*: the suite
stays green today, but the moment Fase 2 rewrites the module these tests
XPASS and ``strict=True`` turns the suite red — remove the marker then, never
the test itself.
"""

from __future__ import annotations

import pandas as pd
import pytest

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
from mldd.dataset import MultiTargetBioactivityIngestor

_INGESTOR = MultiTargetBioactivityIngestor([])


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
    return _INGESTOR.preprocess_and_calculate_pic50(pd.DataFrame(records))


def test_threshold_exact_six_is_active() -> None:
    """pChEMBL == 6.0 is ACTIVE (inclusive boundary)."""
    result = _run([_record(6.0)])
    assert result[BIOACTIVITY_CLASS].tolist() == [1]
    assert result[PCHEMBL_VALUE].iloc[0] == pytest.approx(6.0)


def test_threshold_five_is_inactive() -> None:
    """pChEMBL == 5.0 is INACTIVE (satisfies both rules; kept un-marked)."""
    result = _run([_record(5.0)])
    assert result[BIOACTIVITY_CLASS].tolist() == [0]


@pytest.mark.xfail(
    reason="Fase 2/T2.3: legacy dataset.py creates a 5-6 gray zone and DROPS those rows; "
    "AGENTS.md: active >=6, inactive <6, no gray zone.",
    strict=True,
)
@pytest.mark.parametrize("pic50", [5.5, 5.99])
def test_gray_zone_must_be_labeled_inactive(pic50: float) -> None:
    """Values in (5, 6) are INACTIVE, never dropped (MEMORY.md section 2)."""
    result = _run([_record(pic50)])
    assert result[BIOACTIVITY_CLASS].tolist() == [0]


@pytest.mark.xfail(
    reason="Fase 2/T2.3: legacy dataset.py assumes nM and ignores standard_units "
    "(error of 3 orders of magnitude on uM records).",
    strict=True,
)
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


@pytest.mark.xfail(
    reason="Fase 2/T2.3: legacy dataset.py never filters standard_relation, but only "
    "'=' holds an exact value; bounds ('<', '>') must be excluded.",
    strict=True,
)
def test_non_equality_relation_is_excluded() -> None:
    """A '<'/'>' bound is not an exact measurement: the record must be dropped."""
    record = _record(7.0)
    record[STANDARD_RELATION] = ">"
    result = _run([record])
    assert result.empty
