"""Centralised configuration: Hydra composition + Pydantic v2 validation.

Single source of truth: the YAML files under ``configs/`` (AGENTS.md). This
module defines the *schema* (Pydantic models, ``extra="forbid"``) and the
:func:`load_config` loader that composes the Hydra tree (``target``,
``training``, ``data``, ``features``, ``model`` groups) and validates the
result. Invalid or missing keys raise :class:`mldd.exceptions.ConfigError`
with a precise message — never silently default scientific values in code.

Usage::

    from mldd.config import load_config

    cfg = load_config(["target=maob", "training=regression"])
    cfg.target.chembl_id        # the MAO-B accession defined in configs/target/maob.yaml
    cfg.training.stratify       # False (regression never stratifies, T1.3)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Final, Literal, Self

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from mldd.constants import Task
from mldd.exceptions import ConfigError

# ------------------------------------------------------------------------------
# Repo-relative paths (structural layout, not tunable parameters)
# ------------------------------------------------------------------------------
BASE_DIR: Final = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR: Final = BASE_DIR / "configs"
DATA_DIR: Final = BASE_DIR / "data"
RAW_DATA_DIR: Final = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final = DATA_DIR / "processed"
MODELS_DIR: Final = BASE_DIR / "models"
REPORTS_DIR: Final = BASE_DIR / "reports" / "figures"

# ------------------------------------------------------------------------------
# Registry vocabularies (Fase 4) — mirrored by configs/model/registry.yaml
# ------------------------------------------------------------------------------
AlgorithmName = Literal["xgb", "rf", "lgbm", "hgb", "svr", "lda"]
ScoringName = Literal["mcc", "r2"]


# ------------------------------------------------------------------------------
# Schema (T1.1) — mirrors configs/*.yaml; extra keys are rejected
# ------------------------------------------------------------------------------
class VariantSplitConfig(BaseModel):
    """Regex-based LRRK2 WT/G2019S split over ``assay_description`` (NB01)."""

    model_config = ConfigDict(extra="forbid")

    inclusion_pattern: str
    exclusion_pattern: str
    g2019s_label: str = "G2019S"
    wt_label: str = "WT"

    @field_validator("inclusion_pattern", "exclusion_pattern")
    @classmethod
    def _compilable(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc
        return value


class TargetConfig(BaseModel):
    """A ChEMBL target dataset (6 datasets, 4 unique accessions)."""

    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(min_length=1, description="Dataset id, e.g. 'MAO-B'.")
    chembl_id: str = Field(pattern=r"^CHEMBL\d+$", description="ChEMBL target accession.")
    name: str = Field(min_length=1)
    organism: str = "Homo sapiens"
    target_type: str = "SINGLE PROTEIN"
    variant_split: VariantSplitConfig | None = None
    derived_from: str | None = Field(default=None, description="Parent dataset for split variants.")


class DataConfig(BaseModel):
    """ChEMBL ingestion & curation rules (Notebook 01/02 science)."""

    model_config = ConfigDict(extra="forbid")

    standard_type: str = Field(min_length=1)
    assay_types: list[str] = Field(min_length=1)
    standard_relation: Literal["="]
    rescue_unit_offsets: dict[str, float] = Field(min_length=1)
    active_threshold: float = Field(ge=0.0, le=12.0)
    dedup_max_spread: float = Field(gt=0.0)


class FeatureConfig(BaseModel):
    """Molecular featurization & filtering (Notebook 03/04/05 science)."""

    model_config = ConfigDict(extra="forbid")

    featurizer: Literal["pubchem_fp", "morgan_fp"]
    variance_threshold_pre: float = Field(ge=0.0, le=1.0)
    variance_threshold_post: float = Field(ge=0.0, le=1.0)
    zscore_threshold: float = Field(gt=0.0)
    padel_timeout: int = Field(gt=0, description="Subprocess cap (s) for the PaDEL Java run.")


class SearchSpaceParamConfig(BaseModel):
    """One tunable hyper-parameter: how Optuna must suggest it (Fase 4 / T4.1).

    Numeric params declare ``low``/``high`` (plus optional log-uniform
    sampling); categorical params declare ``choices`` (``null`` allowed, e.g.
    ``max_features`` in NB07). Values live only in ``configs/model/*.yaml``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["int", "float", "categorical"]
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list[str | int | float | None] | None = None

    @model_validator(mode="after")
    def _bounds_match_type(self) -> Self:
        """Reject inconsistent declarations (missing bounds/choices, log<=0)."""
        if self.type == "categorical":
            if not self.choices:
                raise ValueError("categorical params require a non-empty 'choices' list")
            if self.low is not None or self.high is not None:
                raise ValueError("'low'/'high' are only valid for int/float params")
            if self.log:
                raise ValueError("'log' is only valid for int/float params")
        else:
            if self.low is None or self.high is None:
                raise ValueError(f"{self.type} params require both 'low' and 'high'")
            if self.low >= self.high:
                raise ValueError("'low' must be strictly below 'high'")
            if self.log and self.low <= 0:
                raise ValueError("log-scaled params require low > 0")
        return self


class CandidateConfig(BaseModel):
    """One registry candidate (Fase 4): estimator policy + Optuna search space.

    ``scale`` freezes the notebook pipeline policy (NB06 scales every
    classifier; NB07 scales only SVR); ``fixed`` holds operational params;
    ``random_state`` is injected at runtime from the global seed.
    """

    model_config = ConfigDict(extra="forbid")

    scale: bool = Field(description="Wrap the estimator in StandardScaler.")
    fixed: dict[str, Any] = Field(default_factory=dict)
    space: dict[str, SearchSpaceParamConfig] = Field(min_length=1)


class ModelConfig(BaseModel):
    """Thesis model registry (Fase 4 / T4.1): candidates + search spaces per task.

    Every task declares its candidate algorithms (NB06: XGB/RF/LGBM/LDA;
    NB07: XGB/RF/HGB/SVR) and a search space per candidate. The Phase-0
    RandomForest baseline (``baseline_rf.yaml``) was removed in T4.3.
    """

    model_config = ConfigDict(extra="forbid")

    candidates: dict[Task, list[AlgorithmName]] = Field(min_length=1)
    spaces: dict[Task, dict[AlgorithmName, CandidateConfig]] = Field(min_length=1)

    @model_validator(mode="after")
    def _candidates_have_spaces(self) -> Self:
        """Every declared candidate needs a search space for its own task."""
        if set(self.candidates) != set(self.spaces):
            raise ValueError("candidates and spaces must declare the same tasks")
        for task, names in self.candidates.items():
            if not names:
                raise ValueError(f"candidates for task '{task}' must not be empty")
            missing = [name for name in names if name not in self.spaces[task]]
            if missing:
                raise ValueError(
                    f"candidates {missing} of task '{task}' have no search space declared"
                )
        return self

    def candidate(self, task: Task, name: AlgorithmName) -> CandidateConfig:
        """Return the registry entry for algorithm ``name`` under ``task``.

        Raises:
            ConfigError: The algorithm is not registered for that task.
        """
        try:
            return self.spaces[task][name]
        except KeyError as exc:
            raise ConfigError(f"Model '{name}' is not registered for task '{task}'") from exc


class TrainingConfig(BaseModel):
    """Train/test & cross-validation protocol."""

    model_config = ConfigDict(extra="forbid")

    task: Task
    test_size: float = Field(gt=0.0, lt=1.0)
    stratify: bool
    cv_folds: int = Field(ge=2)
    shuffle_cv: bool
    n_trials: int = Field(ge=1)
    scoring: ScoringName

    @model_validator(mode="after")
    def _no_stratify_on_regression(self) -> Self:
        """T1.3 decision: regression must not stratify (sklearn rejects it)."""
        if self.task is Task.REGRESSION and self.stratify:
            raise ValueError("stratify=true is invalid for task=regression")
        return self


class PathsConfig(BaseModel):
    """Directory layout, resolved against ``BASE_DIR`` at runtime."""

    model_config = ConfigDict(extra="forbid")

    data_dir: str = "data"
    models_dir: str = "models"
    reports_dir: str = "reports/figures"


class TrackingConfig(BaseModel):
    """MLflow tracking (local file store; server-ready, Fase 5 / T5.1)."""

    model_config = ConfigDict(extra="forbid")

    tracking_uri: str = "mlruns"
    experiment_name: str = "mldd-bioactivity"


class AppConfig(BaseModel):
    """Root configuration produced by :func:`load_config`."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    target: TargetConfig
    training: TrainingConfig
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    # --- resolved paths ------------------------------------------------------
    @property
    def data_path(self) -> Path:
        return (BASE_DIR / self.paths.data_dir).resolve()

    @property
    def models_path(self) -> Path:
        return (BASE_DIR / self.paths.models_dir).resolve()

    @property
    def reports_path(self) -> Path:
        return (BASE_DIR / self.paths.reports_dir).resolve()


# ------------------------------------------------------------------------------
# Loader (T1.2) — compose + validate; no magic literals in src/
# ------------------------------------------------------------------------------
def load_config(overrides: list[str] | None = None, *, config_dir: Path | None = None) -> AppConfig:
    """Compose the Hydra config tree and validate it with Pydantic.

    Args:
        overrides: Hydra-style overrides, e.g. ``["target=maob", "training=regression"]``.
        config_dir: Alternative config root (defaults to the repo ``configs/``).

    Returns:
        A fully validated :class:`AppConfig`.

    Raises:
        ConfigError: Missing config dir, bad override/composition, or schema
            violation (unknown key, out-of-range threshold, wrong ChEMBL id...).
    """
    root = (config_dir or CONFIG_DIR).resolve()
    if not root.is_dir():
        raise ConfigError(f"Configuration directory not found: {root}")

    try:
        with initialize_config_dir(version_base="1.3", config_dir=str(root)):
            composed = compose(config_name="config", overrides=list(overrides or []))
        container = OmegaConf.to_container(composed, resolve=True)
    except Exception as exc:  # hydra/omegaconf composition errors
        raise ConfigError(f"Configuration composition failed: {exc}") from exc

    if not isinstance(container, dict):
        raise ConfigError(f"Composed configuration is not a mapping: {type(container).__name__}")

    try:
        return AppConfig.model_validate(container)
    except ValidationError as exc:
        raise ConfigError(f"Configuration validation failed:\n{exc}") from exc
