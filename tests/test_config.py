"""Configuration layer tests (Fase 1 / T1.1-T1.3): composition and validation.

Covers the DoD: (a) all 6 targets x 2 tasks compose without touching code,
(b) invalid configs raise ConfigError with a clear message, (c) zero scientific
literals leak into src/ (they live only under configs/).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mldd.config import BASE_DIR, CONFIG_DIR, AppConfig, load_config
from mldd.constants import Task
from mldd.exceptions import ConfigError

_TARGET_OPTIONS = ["ache", "maob", "gsk3b", "lrrk2", "lrrk2_wt", "lrrk2_g2019s"]


# ------------------------------------------------------------------------------
# T1.2: composition matrix (6 targets x 2 tasks), no code changes required
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("target", _TARGET_OPTIONS)
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_compose_all_target_task_combos(target: str, task: str) -> None:
    cfg = load_config([f"target={target}", f"training={task}"])
    assert isinstance(cfg, AppConfig)
    assert cfg.target.dataset
    assert cfg.training.task is (
        Task.CLASSIFICATION if task == "classification" else Task.REGRESSION
    )


def test_default_config_is_valid() -> None:
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.target.chembl_id == "CHEMBL2039"  # default target=maob
    assert cfg.training.task is Task.REGRESSION  # default training=regression


# ------------------------------------------------------------------------------
# T1.3: stratify policy (documented decision)
# ------------------------------------------------------------------------------
def test_classification_stratifies_regression_does_not() -> None:
    assert load_config(["training=classification"]).training.stratify is True
    assert load_config(["training=regression"]).training.stratify is False


def test_regression_rejects_stratify_true() -> None:
    with pytest.raises(ConfigError, match="stratify"):
        load_config(["training=regression", "training.stratify=true"])


# ------------------------------------------------------------------------------
# Fase 4 / T4.1: model registry schema (candidates + Optuna search spaces)
# ------------------------------------------------------------------------------
def test_registry_config_composes_and_validates() -> None:
    cfg = load_config(["model=registry"])
    assert cfg.model.candidates[Task.CLASSIFICATION] == ["xgb", "rf", "lgbm", "lda"]
    assert cfg.model.candidates[Task.REGRESSION] == ["xgb", "rf", "hgb", "svr"]
    xgb_space = cfg.model.spaces[Task.CLASSIFICATION]["xgb"].space
    assert xgb_space["learning_rate"].log is True
    assert cfg.model.spaces[Task.REGRESSION]["svr"].scale is True


def test_registry_scoring_matches_task_protocol() -> None:
    assert load_config(["training=classification"]).training.scoring == "mcc"
    assert load_config(["training=regression"]).training.scoring == "r2"


def test_registry_candidate_without_space_fails() -> None:
    """A candidate must have a search space declared for its own task."""
    with pytest.raises(ConfigError, match="search space"):
        load_config(["model=registry", "model.candidates.classification=[svr]"])


def test_registry_log_space_requires_positive_low() -> None:
    with pytest.raises(ConfigError, match="log-scaled"):
        load_config(
            ["model=registry", "model.spaces.classification.xgb.space.learning_rate.low=-3"]
        )


def test_registry_categorical_requires_choices() -> None:
    with pytest.raises(ConfigError, match="choices"):
        load_config(["model=registry", "model.spaces.classification.lda.space.solver.choices=[]"])


def test_candidate_helper_rejects_unregistered_pair() -> None:
    cfg = load_config(["model=registry"])
    with pytest.raises(ConfigError, match="not registered"):
        cfg.model.candidate(Task.CLASSIFICATION, "svr")


# ------------------------------------------------------------------------------
# LRRK2 variant split (domain rule lives in config, validated as regex)
# ------------------------------------------------------------------------------
def test_lrrk2_variant_split_regexes_compile() -> None:
    split = load_config(["target=lrrk2"]).target.variant_split
    assert split is not None
    re.compile(split.inclusion_pattern)
    re.compile(split.exclusion_pattern)
    assert "G2019S" in split.inclusion_pattern


def test_derived_variants_have_no_split_of_their_own() -> None:
    for option in ("lrrk2_wt", "lrrk2_g2019s"):
        assert load_config([f"target={option}"]).target.variant_split is None


# ------------------------------------------------------------------------------
# Validation: bad configs fail loudly (ConfigError, never silent defaults)
# ------------------------------------------------------------------------------
def test_unknown_target_override_fails() -> None:
    with pytest.raises(ConfigError):
        load_config(["target=does_not_exist"])


def test_unknown_key_fails() -> None:
    """Two validation layers: Hydra rejects unknown overrides, Pydantic extra=forbid."""
    with pytest.raises(ConfigError, match="override"):
        load_config(["training.nonexistent_key=1"])
    with pytest.raises(ConfigError, match="Extra inputs"):
        # '+key' forces Hydra to append it; then the Pydantic schema forbids it.
        load_config(["+training.nonexistent_key=1"])


def test_out_of_range_threshold_fails() -> None:
    with pytest.raises(ConfigError):
        load_config(["data.active_threshold=99"])


def test_bad_task_value_fails() -> None:
    with pytest.raises(ConfigError):
        load_config(["training.task=quantum"])


def test_missing_config_dir_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(config_dir=tmp_path / "nope")


# ------------------------------------------------------------------------------
# DoD (T1.1): scientific literals live only in configs/, not in src/
# ------------------------------------------------------------------------------
#: Values that must never be hardcoded in src/ (they belong to configs/).
#: Note: specific ChEMBL accessions are banned too (AGENTS.md); the generic
#: word "CHEMBL" is allowed (e.g. the regex pattern in TargetConfig).
_BANNED = (
    "6.0",  # active threshold
    "0.16",  # variance threshold (post-merge)
    "1e-9",  # nM -> M conversion factor
    "n_estimators",  # hyper-parameters
    "CHEMBL220",  # AChE
    "CHEMBL2039",  # MAO-B
    "CHEMBL203",  # the classic trap id (substring of CHEMBL2039 anyway)
    "CHEMBL262",  # GSK3B
    "CHEMBL1075104",  # LRRK2
)
#: Modules exempt from the scan. Empty since Fase 2/T2.3 closed: every module
#: in src/ must keep the scientific literals in configs/ only.
_ALLOWLIST: set[str] = set()


def test_no_scientific_literals_in_src() -> None:
    offenders: list[str] = []
    for path in (BASE_DIR / "src" / "mldd").rglob("*.py"):
        if path.name in _ALLOWLIST:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            for token in _BANNED:
                if token in line:
                    offenders.append(f"{path.relative_to(BASE_DIR)}:{lineno}: {token}")
    assert not offenders, "Move these to configs/:\n" + "\n".join(offenders)


def test_configs_tree_is_complete() -> None:
    """Required groups exist: root, 6 targets, 2 training protocols."""
    assert (CONFIG_DIR / "config.yaml").is_file()
    assert len(list((CONFIG_DIR / "target").glob("*.yaml"))) == 6
    assert (CONFIG_DIR / "training" / "classification.yaml").is_file()
    assert (CONFIG_DIR / "training" / "regression.yaml").is_file()
    assert (CONFIG_DIR / "data" / "chembl.yaml").is_file()
    assert (CONFIG_DIR / "features" / "physchem.yaml").is_file()
    assert (CONFIG_DIR / "model" / "registry.yaml").is_file()
