"""Unit tests for the ChEMBL I/O boundary (Fase 2 / T2.1) using a fake client.

No network access: the official client is replaced by an in-memory double that
records the filter chain, so the acquisition filters (thesis NB01) are asserted
exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from tenacity import stop_after_attempt, wait_exponential

import mldd.io as io
from mldd.config import load_config
from mldd.constants import TARGET_ID
from mldd.exceptions import IngestionError

_CONFIG = load_config(["target=maob"])
_TARGET = _CONFIG.target
_DATA = _CONFIG.data


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
