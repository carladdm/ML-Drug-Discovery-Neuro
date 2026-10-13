"""Feature-layer tests (Fase 3): RDKit standardization and Lipinski descriptors (T3.1).

The science is frozen by Notebook 03: largest-fragment + uncharger
standardization (``smiles_clean``) and MW/LogP/HBD/HBA descriptors computed on
the standardized structures. Edge cases (salts, charged groups, unparseable
structures) follow the notebook's own test molecules.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mldd.constants import (
    LOGP,
    MW,
    NUM_H_ACCEPTORS,
    NUM_H_DONORS,
    SMILES,
    SMILES_CLEAN,
    STANDARD_UNITS,
    STANDARD_VALUE,
)
from mldd.exceptions import FeaturizationError
from mldd.features import (
    add_standardized_smiles,
    compute_lipinski_descriptors,
    prepare_physchem_features,
    standardize_smiles,
)

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
