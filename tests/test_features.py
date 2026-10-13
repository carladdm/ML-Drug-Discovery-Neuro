"""Feature-layer tests (Fase 3): descriptors, outlier removal and variance filters.

The science is frozen by the notebooks:
- NB03: largest-fragment + uncharger standardization (``smiles_clean``),
  MW/LogP/HBD/HBA descriptors on standardized structures, and the single-pass
  Z-score outlier filter (|z| > threshold over 5 descriptors).
- NB04/NB05: variance filters (0 on raw bits, 0.16 after the X+Y merge).

Edge cases (salts, charged groups, unparseable structures) follow the
notebooks' own test molecules.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mldd.config import load_config
from mldd.constants import (
    LOGP,
    MW,
    NUM_H_ACCEPTORS,
    NUM_H_DONORS,
    PCHEMBL_VALUE,
    SMILES,
    SMILES_CLEAN,
    STANDARD_UNITS,
    STANDARD_VALUE,
)
from mldd.exceptions import FeaturizationError
from mldd.features import (
    add_standardized_smiles,
    compute_lipinski_descriptors,
    filter_low_variance,
    merge_fingerprints_and_targets,
    prepare_physchem_features,
    remove_zscore_outliers,
    standardize_smiles,
)

_FEATURES = load_config().features

# NB03 cell 7 (notebook test molecules)
_SALT = "CN.[Cl-]"  # methylamine hydrochloride
_CHARGED = "CC(=O)[O-]"  # acetate


# ------------------------------------------------------------------------------
# T3.1: structure standardization (largest fragment + uncharger)
# ------------------------------------------------------------------------------
def test_standardize_removes_salt_and_keeps_largest_fragment() -> None:
    assert standardize_smiles(_SALT) == "CN"


def test_standardize_neutralizes_charges() -> None:
    assert standardize_smiles(_CHARGED) == "CC(=O)O"


def test_standardize_keeps_neutral_molecules_verbatim() -> None:
    assert standardize_smiles("CCO") == "CCO"


def test_standardize_returns_none_for_invalid_or_missing_input() -> None:
    assert standardize_smiles(None) is None
    assert standardize_smiles("this_is_not_a_smiles") is None


def test_add_standardized_smiles_adds_column_preserving_rows() -> None:
    df = pd.DataFrame({SMILES: [_SALT, "CCO"], "molecule_chembl_id": ["M1", "M2"]})
    out = add_standardized_smiles(df)
    assert list(out[SMILES_CLEAN]) == ["CN", "CCO"]
    assert list(out["molecule_chembl_id"]) == ["M1", "M2"]  # rows preserved
    assert SMILES_CLEAN not in df.columns  # input not mutated


def test_add_standardized_smiles_requires_smiles_column() -> None:
    with pytest.raises(FeaturizationError, match=SMILES):
        add_standardized_smiles(pd.DataFrame({"other": [1]}))


# ------------------------------------------------------------------------------
# T3.1: Lipinski descriptors on the standardized structures
# ------------------------------------------------------------------------------
def test_lipinski_values_match_reference_molecule() -> None:
    """Ethanol: MW 46.07 g/mol, LogP ~0 (Crippen), 1 donor, 1 acceptor."""
    df = pd.DataFrame({SMILES_CLEAN: ["CCO"]})
    out = compute_lipinski_descriptors(df)
    assert out[MW].iloc[0] == pytest.approx(46.069, abs=0.01)
    assert out[LOGP].iloc[0] == pytest.approx(-0.0014, abs=0.01)
    assert out[NUM_H_DONORS].iloc[0] == 1
    assert out[NUM_H_ACCEPTORS].iloc[0] == 1


def test_lipinski_drops_unparseable_rows() -> None:
    """NB03 appends NaN for invalid structures; production drops them (no NaN
    may reach a model matrix) and logs the count."""
    df = pd.DataFrame({SMILES_CLEAN: ["CCO", None, "bogus"]})
    out = compute_lipinski_descriptors(df)
    assert len(out) == 1
    assert out[NUM_H_DONORS].iloc[0] == 1


def test_lipinski_uses_the_configured_smiles_column() -> None:
    df = pd.DataFrame({SMILES: ["CCO"]})
    out = compute_lipinski_descriptors(df, smiles_col=SMILES)
    assert len(out) == 1


def test_lipinski_requires_the_smiles_column() -> None:
    with pytest.raises(FeaturizationError, match=SMILES_CLEAN):
        compute_lipinski_descriptors(pd.DataFrame({"other": [1]}))


# ------------------------------------------------------------------------------
# T3.1: composite NB03 step 1 (standardize -> descriptors -> drop artifacts)
# ------------------------------------------------------------------------------
def test_prepare_physchem_features_mirrors_nb03_step_1() -> None:
    df = pd.DataFrame(
        {
            "molecule_chembl_id": ["M1", "M2"],
            SMILES: [_SALT, "CCO"],
            "pchembl_value": [7.1, 5.2],
            STANDARD_VALUE: [79.4, 6309.6],
            STANDARD_UNITS: ["nM", "nM"],
        }
    )
    out = prepare_physchem_features(df)

    assert list(out[SMILES_CLEAN]) == ["CN", "CCO"]
    for column in (MW, LOGP, NUM_H_DONORS, NUM_H_ACCEPTORS):
        assert column in out.columns
    # Curation artifacts are dropped, as NB03 does before saving the dataset
    assert STANDARD_VALUE not in out.columns
    assert STANDARD_UNITS not in out.columns
    # Measurement columns survive for the modelling stage
    assert "pchembl_value" in out.columns
    assert out.index.tolist() == [0, 1]  # index reset after any row drop


def test_prepare_physchem_features_requires_smiles_column() -> None:
    with pytest.raises(FeaturizationError, match=SMILES):
        prepare_physchem_features(pd.DataFrame({"other": [1]}))


# ------------------------------------------------------------------------------
# T3.3: single-pass Z-score outlier removal (NB03 cell 30)
# ------------------------------------------------------------------------------
def _zscore_frame(values: list[float]) -> pd.DataFrame:
    """Frame whose ``pchembl_value`` carries ``values`` and descriptors are constant."""
    n = len(values)
    return pd.DataFrame(
        {
            "molecule_chembl_id": [f"M{index}" for index in range(n)],
            PCHEMBL_VALUE: values,
            MW: [300.0] * n,
            LOGP: [2.5] * n,
            NUM_H_DONORS: [1] * n,
            NUM_H_ACCEPTORS: [4] * n,
        }
    )


def test_zscore_removes_extreme_rows_single_pass() -> None:
    """Statistics are computed once: the second extreme row is judged against
    the ORIGINAL mean/std, so an iterative filter would over-remove."""
    df = _zscore_frame([0.0] * 17 + [10.0, 3.5])

    out = remove_zscore_outliers(df, _FEATURES)

    assert len(out) == 18  # only the |z| > 3 row goes
    assert 3.5 in out[PCHEMBL_VALUE].tolist()
    assert 10.0 not in out[PCHEMBL_VALUE].tolist()
    assert out.index.tolist() == list(range(18))  # index reset


def test_zscore_uses_config_threshold_not_a_literal() -> None:
    """The cut-off comes from configs/ (zscore_threshold: 3.0)."""
    df = _zscore_frame([0.0] * 9 + [2.6])  # |z| < 3 -> kept
    assert len(remove_zscore_outliers(df, _FEATURES)) == 10

    relaxed = _FEATURES.model_copy(update={"zscore_threshold": 2.0})
    assert len(remove_zscore_outliers(df, relaxed)) == 9  # now an outlier


def test_zscore_skips_zero_variance_descriptors() -> None:
    """A constant descriptor never flags rows (std == 0 guard, NB03)."""
    df = _zscore_frame([5.0, 5.5, 6.0, 6.5, 7.0])
    out = remove_zscore_outliers(df, _FEATURES)
    assert len(out) == 5  # nothing extreme, constant descriptors skipped


def test_zscore_requires_descriptor_columns() -> None:
    with pytest.raises(FeaturizationError, match=PCHEMBL_VALUE):
        remove_zscore_outliers(pd.DataFrame({"other": [1]}), _FEATURES)


# ------------------------------------------------------------------------------
# T3.3: variance filters (NB04 pre-process, NB05 post-merge)
# ------------------------------------------------------------------------------
def test_variance_pre_filter_drops_only_constant_columns() -> None:
    """threshold=0.0 (NB04): a feature survives iff it varies at all."""
    df = pd.DataFrame(
        {
            "PubchemFP0": [1, 1, 1, 1],  # constant -> dropped
            "PubchemFP1": [0, 1, 0, 1],  # varying -> kept
            "PubchemFP2": [0, 0, 0, 1],  # varying -> kept
        }
    )
    out = filter_low_variance(df, _FEATURES.variance_threshold_pre)
    assert list(out.columns) == ["PubchemFP1", "PubchemFP2"]


def test_variance_post_filter_matches_nb05_bernoulli_boundary() -> None:
    """threshold=0.16 (NB05): keeps features strictly above 0.8 * (1 - 0.8)."""
    df = pd.DataFrame(
        {
            "rare_bit": [1] * 16 + [0] * 4,  # var == 0.16 exactly -> dropped
            "skewed_bit": [1] * 17 + [0] * 3,  # var == 0.1311 -> dropped
            "balanced_bit": [1] * 10 + [0] * 10,  # var == 0.25 -> kept
        }
    )
    out = filter_low_variance(df, _FEATURES.variance_threshold_post)
    assert list(out.columns) == ["balanced_bit"]


def test_variance_filter_rejects_empty_or_fully_constant_matrix() -> None:
    with pytest.raises(FeaturizationError, match="empty"):
        filter_low_variance(pd.DataFrame(), _FEATURES.variance_threshold_pre)
    with pytest.raises(FeaturizationError, match="every feature"):
        filter_low_variance(pd.DataFrame({"a": [1, 1], "b": [2, 2]}), 0.5)


# ------------------------------------------------------------------------------
# T3.3: X + y merge (NB04 step 5)
# ------------------------------------------------------------------------------
def test_merge_aligns_positionally_ignoring_indexes() -> None:
    """NB04 concatenates X and y positionally; shared index labels are a trap."""
    fingerprints = pd.DataFrame({"PubchemFP0": [0, 1, 0]}, index=[10, 11, 12])
    targets = pd.DataFrame(
        {"bioactivity_class": [1, 0, 1], PCHEMBL_VALUE: [7.1, 5.2, 6.4]}, index=[0, 1, 2]
    )
    out = merge_fingerprints_and_targets(fingerprints, targets)
    assert out.iloc[0].tolist() == [0, 1, 7.1]  # position 0, not index label
    assert list(out.columns) == ["PubchemFP0", "bioactivity_class", PCHEMBL_VALUE]


def test_merge_rejects_length_mismatch() -> None:
    with pytest.raises(FeaturizationError, match="mismatch"):
        merge_fingerprints_and_targets(
            pd.DataFrame({"PubchemFP0": [0, 1]}), pd.DataFrame({"y": [1]})
        )
