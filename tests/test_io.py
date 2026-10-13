"""Unit tests for the ChEMBL and PaDEL I/O boundaries (Fase 2 / T2.1, Fase 3 / T3.2).

No network and no Java in the unit tests: the ChEMBL client is replaced by an
in-memory double that records the filter chain, and the PaDEL subprocess is
replaced by a fake that writes PaDEL-shaped CSVs, so the acquisition filters
(thesis NB01) and the PubChem fingerprint contract (NB04) are asserted exactly.
The real PaDEL run lives behind the ``padel`` marker.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import padelpy
import pandas as pd
import pytest
from tenacity import stop_after_attempt, wait_exponential

import mldd.io as io
from mldd.config import load_config
from mldd.constants import PUBCHEM_FP_BITS, PUBCHEM_FP_COLUMNS, PUBCHEM_FP_NAME_COLUMN, TARGET_ID
from mldd.exceptions import FeaturizationError, IngestionError

_CONFIG = load_config(["target=maob"])
_TARGET = _CONFIG.target
_DATA = _CONFIG.data
_FEATURES = _CONFIG.features


class _FakeQuery:
    """Records chained filters and yields a canned record list."""

    def __init__(self, records: list[dict[str, Any]], calls: list[dict[str, Any]]) -> None:
        self._records = records
        self._calls = calls

    def filter(self, **kwargs: Any) -> _FakeQuery:
        self._calls.append(kwargs)
        return self

    def __iter__(self) -> Any:
        return iter(self._records)


class _FakeClient:
    """Stands in for ``new_client.activity``."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.calls: list[dict[str, Any]] = []

    def filter(self, **kwargs: Any) -> _FakeQuery:
        self.calls.append(kwargs)
        return _FakeQuery(self.records, self.calls)


def test_fetch_applies_thesis_acquisition_filters() -> None:
    """NB01 filters: target id, standard type, assay types B/F and relation '='."""
    client = _FakeClient([{"molecule_chembl_id": "M1"}])
    df = io.fetch_raw_activities(_TARGET, _DATA, client=client)
    assert client.calls == [
        {"target_chembl_id": _TARGET.chembl_id},
        {"standard_type": _DATA.standard_type},
        {"assay_type__in": list(_DATA.assay_types)},
        {"standard_relation": _DATA.standard_relation},
    ]
    assert list(df[TARGET_ID]) == [_TARGET.chembl_id]


def test_fetch_empty_response_raises() -> None:
    with pytest.raises(IngestionError, match="No IC50 records"):
        io.fetch_raw_activities(_TARGET, _DATA, client=_FakeClient([]))


def test_fetch_implausible_volume_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A filter that silently fails would return the whole table: refuse it."""
    monkeypatch.setattr(io, "_MAX_RECORDS_GUARD", 1)
    with pytest.raises(IngestionError, match="Implausible"):
        io.fetch_raw_activities(_TARGET, _DATA, client=_FakeClient([{"a": 1}, {"a": 2}]))


def test_transport_failure_is_retried_then_translated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Network errors are retried (no sleeping in tests) and surfaced as IngestionError."""
    monkeypatch.setattr(io, "_RETRY_STOP", stop_after_attempt(2))
    monkeypatch.setattr(io, "_RETRY_WAIT", wait_exponential(multiplier=0, max=0))

    class _BrokenClient:
        def filter(self, **kwargs: Any) -> Any:
            raise ConnectionError("ChEMBL is down")

    with pytest.raises(IngestionError, match="ChEMBL query failed"):
        io.fetch_raw_activities(_TARGET, _DATA, client=_BrokenClient())


def test_save_and_load_raw_csv_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "nested" / "raw.csv"
    io.save_raw_csv(df, path)
    pd.testing.assert_frame_equal(io.load_raw_csv(path), df)


def test_load_raw_csv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="not found"):
        io.load_raw_csv(tmp_path / "missing.csv")


# ------------------------------------------------------------------------------
# T3.2: PubChem fingerprint boundary (PaDEL subprocess, no Java in unit tests)
# ------------------------------------------------------------------------------
def _fake_pubchem_csv(path: Path, n_rows: int, *, drop_columns: tuple[str, ...] = ()) -> None:
    """Write a PaDEL-shaped CSV honouring (or breaking) the PubChem contract."""
    columns = [PUBCHEM_FP_NAME_COLUMN, *(c for c in PUBCHEM_FP_COLUMNS if c not in drop_columns)]
    frame = pd.DataFrame(
        [[f"M{index}", *([0] * (len(columns) - 1))] for index in range(n_rows)], columns=columns
    )
    frame.to_csv(path, index=False)


def _fake_named_csv(path: Path, names: list[str]) -> None:
    """Write a contract-valid PaDEL CSV whose rows carry the given accessions."""
    frame = pd.DataFrame(
        [[name, *([0] * PUBCHEM_FP_BITS)] for name in names],
        columns=[PUBCHEM_FP_NAME_COLUMN, *PUBCHEM_FP_COLUMNS],
    )
    frame.to_csv(path, index=False)


def _smiles_frame(n: int) -> tuple[pd.Series, pd.Series]:
    return pd.Series(["CCO"] * n), pd.Series([f"M{index}" for index in range(n)])


def test_pubchem_fingerprints_returns_contract_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper must drop Name, keep the 881 bits and align with the input."""
    observed: dict[str, Any] = {}

    def _fake_padeldescriptor(**kwargs: Any) -> None:
        observed.update(kwargs)
        smi_path = Path(kwargs["mol_dir"])
        # The .smi file must be tab-separated, headerless, SMILES + id (NB04)
        smi = pd.read_csv(smi_path, sep="\t", header=None)
        assert smi.shape == (3, 2)
        _fake_pubchem_csv(Path(kwargs["d_file"]), len(smi))

    monkeypatch.setattr(padelpy, "padeldescriptor", _fake_padeldescriptor)
    smiles, ids = _smiles_frame(3)
    out = io.compute_pubchem_fingerprints(smiles, ids, _FEATURES)

    assert list(out.columns) == list(PUBCHEM_FP_COLUMNS)
    assert out.shape == (3, PUBCHEM_FP_BITS)
    # Thesis flags: salts removed, nitro standardized, input order retained
    assert observed["removesalt"] is True
    assert observed["standardizenitro"] is True
    assert observed["fingerprints"] is True
    assert observed["retainorder"] is True
    assert observed["sp_timeout"] == _FEATURES.padel_timeout


def test_pubchem_fingerprints_rejects_null_smiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """A null SMILES would be dropped by PaDEL, silently shifting every row."""
    monkeypatch.setattr(padelpy, "padeldescriptor", lambda **_: None)
    smiles, ids = _smiles_frame(2)
    smiles.iloc[1] = None
    with pytest.raises(FeaturizationError, match="misalign"):
        io.compute_pubchem_fingerprints(smiles, ids, _FEATURES)


def test_pubchem_fingerprints_rejects_ragged_or_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(padelpy, "padeldescriptor", lambda **_: None)
    smiles, ids = _smiles_frame(2)
    with pytest.raises(FeaturizationError, match="mismatch"):
        io.compute_pubchem_fingerprints(smiles, ids.iloc[:1], _FEATURES)
    with pytest.raises(FeaturizationError, match="No molecules"):
        io.compute_pubchem_fingerprints(smiles.iloc[:0], ids.iloc[:0], _FEATURES)


def test_pubchem_fingerprints_translates_padel_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _broken(**_: Any) -> None:
        raise RuntimeError("java exploded")

    monkeypatch.setattr(padelpy, "padeldescriptor", _broken)
    smiles, ids = _smiles_frame(1)
    with pytest.raises(FeaturizationError, match="failed"):
        io.compute_pubchem_fingerprints(smiles, ids, _FEATURES)


def test_pubchem_fingerprints_rejects_contract_violations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truncated or short PaDEL output must never reach the training stage."""
    smiles, ids = _smiles_frame(2)

    def _missing_columns(**kwargs: Any) -> None:
        _fake_pubchem_csv(Path(kwargs["d_file"]), 2, drop_columns=(PUBCHEM_FP_COLUMNS[5],))

    def _short_output(**kwargs: Any) -> None:
        _fake_pubchem_csv(Path(kwargs["d_file"]), 1)

    def _no_output(**_: Any) -> None:
        return None

    for fake, match in (
        (_missing_columns, "contract"),
        (_short_output, "misalign"),
        (_no_output, "descriptor file"),
    ):
        monkeypatch.setattr(padelpy, "padeldescriptor", fake)
        with pytest.raises(FeaturizationError, match=match):
            io.compute_pubchem_fingerprints(smiles, ids, _FEATURES)


def test_pubchem_fingerprints_realigns_reordered_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """PaDEL runs multi-threaded and may emit rows out of order: the matrix is
    realigned by the Name column, never trusted to positional order."""
    rows = [("M2", 1), ("M0", 0), ("M1", 1)]  # deliberately shuffled

    def _reordered(**kwargs: Any) -> None:
        frame = pd.DataFrame(
            [[name, pattern, *([0] * (PUBCHEM_FP_BITS - 1))] for name, pattern in rows],
            columns=[PUBCHEM_FP_NAME_COLUMN, *PUBCHEM_FP_COLUMNS],
        )
        frame.to_csv(kwargs["d_file"], index=False)

    monkeypatch.setattr(padelpy, "padeldescriptor", _reordered)
    smiles, ids = _smiles_frame(3)  # M0, M1, M2
    out = io.compute_pubchem_fingerprints(smiles, ids, _FEATURES)

    assert list(out[PUBCHEM_FP_COLUMNS[0]]) == [0, 1, 1]  # input order: M0, M1, M2


def test_pubchem_fingerprints_rejects_name_contract_violations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicated, dropped or duplicated-input accessions can never align X/y."""
    smiles, ids = _smiles_frame(2)

    def _duplicated_names(**kwargs: Any) -> None:
        _fake_named_csv(Path(kwargs["d_file"]), ["M0", "M0"])

    def _unknown_name(**kwargs: Any) -> None:
        _fake_named_csv(Path(kwargs["d_file"]), ["M0", "M9"])  # M1 never emitted

    for fake, match in (
        (_duplicated_names, "duplicated"),
        (_unknown_name, "dropped 1 molecule"),
    ):
        monkeypatch.setattr(padelpy, "padeldescriptor", fake)
        with pytest.raises(FeaturizationError, match=match):
            io.compute_pubchem_fingerprints(smiles, ids, _FEATURES)

    with pytest.raises(FeaturizationError, match="unique"):
        io.compute_pubchem_fingerprints(smiles, pd.Series(["M0", "M0"]), _FEATURES)


@pytest.mark.padel
@pytest.mark.skipif(shutil.which("java") is None, reason="requires a local Java runtime")
def test_pubchem_fingerprints_real_padel_run() -> None:
    """Integration: the bundled PaDEL jar emits the PubChem 881-bit matrix."""
    smiles = pd.Series(["CCO", "CC(=O)Oc1ccccc1C(=O)O"])
    ids = pd.Series(["ETHANOL", "ASPIRIN"])
    out = io.compute_pubchem_fingerprints(smiles, ids, _FEATURES)

    assert out.shape == (2, PUBCHEM_FP_BITS)
    assert out.isin([0, 1]).all().all()
