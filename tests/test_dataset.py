"""Unit tests for the pure curation functions (Fase 2 / T2.2-T2.3, NB02).

Each test pins one thesis rule: unit-aware rescue, InChIKey consensus
deduplication with a spread guard, and the no-gray-zone labelling.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mldd.config import VariantSplitConfig, load_config
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
from mldd.dataset import (
    curate_bioactivity,
    deduplicate_by_inchikey,
    drop_missing_structures,
    label_activity,
    rescue_pchembl,
    select_curated_columns,
    split_lrrk2_variants,
)
from mldd.exceptions import CurationError

_DATA_CONFIG = load_config(["target=maob"]).data
#: Two ethanol spellings that RDKit canonicalizes to the same InChIKey.
_ETHANOL_SMILES = "CCO"
_ETHANOL_CANONICAL = "OCC"


def _record(
    value: float | None = 1000.0,
    units: str = "nM",
    relation: str = "=",
    smiles: str | None = _ETHANOL_SMILES,
    mol_id: str = "MOL_1",
    pchembl: float | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        MOLECULE_ID: mol_id,
        TARGET_ID: "TEST",
        SMILES: smiles,
        STANDARD_VALUE: value,
        STANDARD_UNITS: units,
        STANDARD_RELATION: relation,
    }
    if pchembl is not None:
        record[PCHEMBL_VALUE] = pchembl
    return record


def _frame(records: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(records)


# ------------------------------------------------------------------------------
# drop_missing_structures
# ------------------------------------------------------------------------------
def test_drop_missing_structures_removes_null_value_and_smiles() -> None:
    df = _frame([_record(), _record(smiles=None), _record(value=None)])
    result = drop_missing_structures(df)
    assert len(result) == 1


def test_drop_missing_structures_requires_columns() -> None:
    with pytest.raises(CurationError, match="drop_missing_structures"):
        drop_missing_structures(pd.DataFrame({MOLECULE_ID: ["X"]}))


# ------------------------------------------------------------------------------
# rescue_pchembl
# ------------------------------------------------------------------------------
def test_rescue_uses_nine_minus_log10_for_nanomolar() -> None:
    result = rescue_pchembl(_frame([_record(value=1000.0, units="nM")]), _DATA_CONFIG)
    assert result[PCHEMBL_VALUE].iloc[0] == pytest.approx(6.0)


@pytest.mark.parametrize("units", ["uM", "µM", "micromolar"])
def test_rescue_uses_six_minus_log10_for_micromolar_units(units: str) -> None:
    """1 uM -> 6.0; the nM-only formula would give 9.0 (3 orders of magnitude off)."""
    result = rescue_pchembl(_frame([_record(value=1.0, units=units)]), _DATA_CONFIG)
    assert result[PCHEMBL_VALUE].iloc[0] == pytest.approx(6.0)


def test_rescue_keeps_existing_pchembl_value_verbatim() -> None:
    """When ChEMBL already provides pchembl_value it is authoritative."""
    result = rescue_pchembl(_frame([_record(value=1.0, units="uM", pchembl=7.25)]), _DATA_CONFIG)
    assert result[PCHEMBL_VALUE].iloc[0] == pytest.approx(7.25)


def test_rescue_drops_non_equality_relation() -> None:
    result = rescue_pchembl(_frame([_record(relation="<")]), _DATA_CONFIG)
    assert result.empty


def test_rescue_drops_unknown_unit() -> None:
    result = rescue_pchembl(_frame([_record(units="mg/mL")]), _DATA_CONFIG)
    assert result.empty


@pytest.mark.parametrize("value", [0.0, -5.0])
def test_rescue_drops_non_positive_values(value: float) -> None:
    """log10 of a non-positive value is undefined; the record must go."""
    result = rescue_pchembl(_frame([_record(value=value)]), _DATA_CONFIG)
    assert result.empty


# ------------------------------------------------------------------------------
# deduplicate_by_inchikey
# ------------------------------------------------------------------------------
def test_dedup_collapses_replicates_to_consensus_mean() -> None:
    """Same InChIKey (canonical spellings) -> one record with the mean pChEMBL."""
    df = _frame(
        [
            _record(pchembl=6.0, mol_id="A"),
            _record(smiles=_ETHANOL_CANONICAL, pchembl=7.0, mol_id="B"),
        ]
    )
    result = deduplicate_by_inchikey(df, _DATA_CONFIG)
    assert len(result) == 1
    assert result[PCHEMBL_VALUE].iloc[0] == pytest.approx(6.5)


def test_dedup_drops_groups_exceeding_spread_threshold() -> None:
    """A pChEMBL spread > dedup_max_spread means inconsistent assays: drop all."""
    df = _frame(
        [
            _record(pchembl=5.0, mol_id="A"),
            _record(smiles=_ETHANOL_CANONICAL, pchembl=7.5, mol_id="B"),
        ]
    )
    assert deduplicate_by_inchikey(df, _DATA_CONFIG).empty


def test_dedup_keeps_consistent_groups_and_distinct_molecules() -> None:
    df = _frame(
        [
            _record(pchembl=6.0, mol_id="A"),
            _record(smiles=_ETHANOL_CANONICAL, pchembl=6.2, mol_id="B"),
            _record(smiles="CCN", pchembl=4.0, mol_id="C"),
        ]
    )
    result = deduplicate_by_inchikey(df, _DATA_CONFIG)
    assert len(result) == 2


def test_dedup_drops_unparsable_smiles() -> None:
    df = _frame([_record(smiles="not-a-molecule", pchembl=6.0)])
    assert deduplicate_by_inchikey(df, _DATA_CONFIG).empty


# ------------------------------------------------------------------------------
# label_activity
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(("pic50", "expected"), [(6.0, 1), (5.999, 0), (0.0, 0)])
def test_labelling_has_no_gray_zone(pic50: float, expected: int) -> None:
    """Anything below the threshold is inactive; nothing is dropped."""
    df = _frame([_record(pchembl=pic50)])
    result = label_activity(df, _DATA_CONFIG)
    assert result[BIOACTIVITY_CLASS].tolist() == [expected]


def test_label_activity_rejects_null_pchembl() -> None:
    """Labelling before rescue would silently mark rows inactive: refuse instead."""
    df = _frame([_record()]).assign(**{PCHEMBL_VALUE: [float("nan")]})
    with pytest.raises(CurationError, match="rescue_pchembl"):
        label_activity(df, _DATA_CONFIG)


# ------------------------------------------------------------------------------
# select_curated_columns
# ------------------------------------------------------------------------------
def test_select_curated_columns_keeps_only_the_contract() -> None:
    result = select_curated_columns(_frame([_record(pchembl=6.0)]))
    assert list(result.columns) == [
        MOLECULE_ID,
        SMILES,
        PCHEMBL_VALUE,
        STANDARD_VALUE,
        STANDARD_UNITS,
    ]


# ------------------------------------------------------------------------------
# curate_bioactivity (end-to-end on a synthetic frame)
# ------------------------------------------------------------------------------
def test_curate_end_to_end_produces_contract_columns() -> None:
    df = _frame(
        [
            _record(value=100.0, pchembl=7.0),  # exact nM, ChEMBL pchembl present
            _record(smiles="CCN", value=1.0, units="uM"),  # rescued -> 6.0 active
            _record(smiles="CCC", value=1.0, relation="<"),  # bound -> dropped
            _record(smiles="CCCC", pchembl=5.5),  # gray-zone value -> inactive
        ]
    )
    result = curate_bioactivity(df, _DATA_CONFIG)
    assert list(result.columns) == [
        MOLECULE_ID,
        SMILES,
        PCHEMBL_VALUE,
        STANDARD_VALUE,
        STANDARD_UNITS,
        BIOACTIVITY_CLASS,
    ]
    assert len(result) == 3
    assert set(result[BIOACTIVITY_CLASS]) == {0, 1}


# ------------------------------------------------------------------------------
# split_lrrk2_variants (NB01 cell 18)
# ------------------------------------------------------------------------------
def _variant_split() -> VariantSplitConfig:
    """Return the LRRK2 split rules from the composed config (asserted present)."""
    split = load_config(["target=lrrk2"]).target.variant_split
    assert split is not None, "the lrrk2 target must define variant_split"
    return split


def _variant_frame(
    descriptions: list[str], sequences: list[str | None] | None = None
) -> pd.DataFrame:
    df = pd.DataFrame({"assay_description": descriptions})
    if sequences is not None:
        df["variant_sequence"] = pd.Series(sequences, dtype="string")
    return df


def test_split_marks_inclusion_keyword_as_mutant() -> None:
    frames = split_lrrk2_variants(
        _variant_frame(["IC50 on the G2019S LRRK2 mutant"]), _variant_split()
    )
    assert len(frames.mutant) == 1
    assert frames.wild_type.empty


def test_split_respects_exclusion_keyword() -> None:
    """'Selectivity over mutant' assays measure the WT, not the mutant."""
    frames = split_lrrk2_variants(
        _variant_frame(["Selectivity over the G2019S mutant isoform"]), _variant_split()
    )
    assert len(frames.wild_type) == 1
    assert frames.mutant.empty


def test_split_defaults_to_wild_type_without_keywords() -> None:
    frames = split_lrrk2_variants(_variant_frame(["IC50 determination"]), _variant_split())
    assert len(frames.wild_type) == 1


def test_split_marks_explicit_variant_sequence_as_mutant() -> None:
    frames = split_lrrk2_variants(
        _variant_frame(["IC50 determination", "IC50 determination"], sequences=[None, "MAAG..."]),
        _variant_split(),
    )
    assert len(frames.mutant) == 1
    assert len(frames.wild_type) == 1


def test_split_requires_assay_description() -> None:
    with pytest.raises(CurationError, match="split_lrrk2_variants"):
        split_lrrk2_variants(pd.DataFrame({"molecule_chembl_id": ["X"]}), _variant_split())
