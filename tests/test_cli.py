"""Fase 5 / T5.2 tests: CLI wiring (Hydra overrides -> I/O boundaries).

The I/O itself is tested at its own boundary (``test_io``/``test_trainer``);
here every dependency is faked so the assertions cover command wiring,
config composition and the artifact path conventions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from mldd.cli import app
from mldd.config import AppConfig, load_config
from mldd.constants import (
    BIOACTIVITY_CLASS,
    MOLECULE_ID,
    PCHEMBL_VALUE,
    SMILES,
    Task,
)
from mldd.trainer import TrainingRun

_runner = CliRunner()


def test_cli_exposes_pipeline_commands() -> None:
    result = _runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "featurize", "train", "version"):
        assert command in result.stdout


def test_train_command_composes_overrides_and_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[AppConfig, Path]] = []

    def _fake_run_training(cfg: AppConfig, data_path: Path) -> TrainingRun:
        calls.append((cfg, data_path))
        return TrainingRun(
            target=cfg.target.name,
            task=cfg.training.task,
            algorithm="xgb",
            best_params={},
            metrics={"cv_score": 0.5},
            model_path=Path("model.joblib"),
            mlflow_run_id="run-123",
        )

    monkeypatch.setattr("mldd.trainer.run_training", _fake_run_training)
    result = _runner.invoke(app, ["train", "target=maob", "task=regression"])
    assert result.exit_code == 0, result.output

    cfg, data_path = calls[0]
    assert cfg.target.dataset == "MAO-B"
    assert cfg.training.task is Task.REGRESSION
    assert data_path == cfg.matrix_path
    assert "winner=xgb" in result.stdout


def test_train_command_fails_loudly_on_bad_target() -> None:
    result = _runner.invoke(app, ["train", "target=does_not_exist"])
    assert result.exit_code != 0


def test_ingest_command_saves_raw_and_curated(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_df = pd.DataFrame({"molecule_chembl_id": ["MOL1"], SMILES: ["CCO"]})
    saved: list[Path] = []

    def _fake_save(df: pd.DataFrame, path: Path) -> Path:
        saved.append(path)
        return path

    monkeypatch.setattr("mldd.io.fetch_raw_activities", lambda target, data: raw_df.copy())
    monkeypatch.setattr("mldd.dataset.curate_bioactivity", lambda df, data: df.copy())
    monkeypatch.setattr("mldd.io.save_raw_csv", _fake_save)

    result = _runner.invoke(app, ["ingest", "target=maob"])
    assert result.exit_code == 0, result.output

    cfg = load_config(["target=maob"])
    assert saved == [cfg.raw_path, cfg.curated_path]
    assert cfg.raw_path.name == "MAO-B_bioactivity_data.csv"
    assert cfg.curated_path.name == "MAO-B_bioactivity_data_curated.csv"


def test_ingest_lrrk2_writes_variant_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    curated_df = pd.DataFrame(
        {
            "assay_description": [
                "LRRK2 G2019S mutant inhibition assay",
                "LRRK2 wild type inhibition assay",
            ],
            SMILES: ["CCO", "CCN"],
        }
    )
    saved: list[Path] = []

    def _fake_save(df: pd.DataFrame, path: Path) -> Path:
        saved.append(path)
        return path

    monkeypatch.setattr("mldd.io.fetch_raw_activities", lambda target, data: curated_df.copy())
    monkeypatch.setattr("mldd.dataset.curate_bioactivity", lambda df, data: df.copy())
    monkeypatch.setattr("mldd.io.save_raw_csv", _fake_save)

    result = _runner.invoke(app, ["ingest", "target=lrrk2"])
    assert result.exit_code == 0, result.output

    variant_names = [path.name for path in saved[2:]]
    assert variant_names == [
        "LRRK2_WT_bioactivity_data_curated.csv",
        "LRRK2_G2019S_bioactivity_data_curated.csv",
    ]


def test_featurize_command_builds_model_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    curated = pd.DataFrame(
        {
            MOLECULE_ID: [f"MOL{i}" for i in range(6)],
            SMILES: ["CCO", "CCN", "CCC", "CCCO", "c1ccccc1", "CCCl"],
            PCHEMBL_VALUE: [6.1, 6.2, 6.3, 6.4, 6.5, 6.6],
            BIOACTIVITY_CLASS: [1, 0, 1, 0, 1, 0],
        }
    )
    fingerprints = pd.DataFrame({f"PubchemFP{i}": [0, 1, 0, 1, 1, 0] for i in range(1, 11)})
    saved: list[tuple[Path, pd.DataFrame]] = []

    def _fake_save(df: pd.DataFrame, path: Path) -> Path:
        saved.append((path, df))
        return path

    monkeypatch.setattr("mldd.io.load_raw_csv", lambda path: curated.copy())
    monkeypatch.setattr(
        "mldd.io.compute_pubchem_fingerprints",
        lambda smiles, ids, feature_config: fingerprints.copy(),
    )
    monkeypatch.setattr("mldd.io.save_raw_csv", _fake_save)

    result = _runner.invoke(app, ["featurize", "target=maob"])
    assert result.exit_code == 0, result.output

    path, matrix = saved[0]
    cfg = load_config(["target=maob"])
    assert path == cfg.matrix_path
    assert len(matrix) == 6
    assert BIOACTIVITY_CLASS in matrix.columns
    assert any(column.startswith("PubchemFP") for column in matrix.columns)
