"""Smoke tests: package imports, CLI entry point and path resolution."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import mldd
from mldd.cli import app
from mldd.config import BASE_DIR, DATA_DIR, MODELS_DIR, SEED
from mldd.exceptions import MLDDException
from mldd.logging_utils import get_logger


def test_version_exposed() -> None:
    assert mldd.__version__


def test_base_dir_points_to_repo_root() -> None:
    assert (BASE_DIR / "pyproject.toml").is_file()


def test_data_dirs_are_paths_under_base() -> None:
    for directory in (DATA_DIR, MODELS_DIR):
        assert isinstance(directory, Path)
        assert directory.is_relative_to(BASE_DIR)


def test_seed_is_42() -> None:
    assert SEED == 42


def test_exception_hierarchy() -> None:
    assert issubclass(ValueError, Exception)
    try:
        raise MLDDException("boom")
    except MLDDException as exc:
        assert "boom" in str(exc)


def test_logger_returns_named_logger() -> None:
    log = get_logger("mldd.test")
    assert log.name == "mldd.test"


def test_cli_app_exposes_version_command() -> None:
    """`mldd version` exits cleanly and prints the package version."""
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert mldd.__version__ in result.stdout
