"""Optuna tuning engine over the config-declared search spaces (Fase 4 / T4.2).

Reproduces the NB06/NB07 protocol: a TPE sampler seeded with the global seed
runs ``n_trials`` trials maximising the task scoring metric (MCC for
classification, R² for regression) over a KFold cross-validation. The search
space is never hardcoded here — it is iterated generically from the
:class:`mldd.config.SearchSpaceParamConfig` entries of
``configs/model/registry.yaml`` (the literal scanner in ``tests/test_config.py``
enforces that no hyper-parameter name lives in ``src/``).

Decision (documented): NB07 ran an unseeded sampler and an unshuffled CV;
production unifies both tasks on the seeded, shuffled protocol of NB06 for
reproducibility (``shuffle_cv: true`` in ``configs/training/*.yaml``).
Cross-validation folds run sequentially while each estimator may use all
cores (``fixed: n_jobs: -1``), avoiding joblib nested-parallel oversubscription.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import optuna
import pandas as pd
from optuna import Trial
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import KFold, cross_val_score

from mldd.config import (
    AlgorithmName,
    CandidateConfig,
    ScoringName,
    SearchSpaceParamConfig,
    TrainingConfig,
)
from mldd.constants import Task
from mldd.exceptions import TrainingError
from mldd.logging_utils import get_logger
from mldd.modeling.models import build_estimator, build_pipeline

logger: logging.Logger = get_logger(__name__)

# Optuna logs every trial at INFO on stderr; silence it (no print()/noise in src).
optuna.logging.set_verbosity(optuna.logging.WARNING)

#: Scorer registry (NB06 maximises MCC, NB07 maximises R²; both "maximize").
_SCORERS: dict[ScoringName, Any] = {
    "mcc": make_scorer(matthews_corrcoef),
    "r2": "r2",
}


@dataclass(frozen=True)
class TuningResult:
    """Outcome of one Optuna study (one algorithm, one task)."""

    algorithm: AlgorithmName
    best_params: dict[str, Any]
    best_score: float
    n_trials: int


def suggest_params(trial: Trial, space: dict[str, SearchSpaceParamConfig]) -> dict[str, Any]:
    """Generic Optuna suggestions driven by a config search space.

    Args:
        trial: Active Optuna trial.
        space: Parameter declarations (int/float bounds or categorical choices).

    Returns:
        The suggested value for every declared parameter.

    Raises:
        TrainingError: A numeric param is missing bounds (schema should
            prevent this; guarded so mypy and runtime fail loudly).
    """
    suggested: dict[str, Any] = {}
    for name, spec in space.items():
        if spec.type == "categorical":
            # Optuna types choices with a constrained TypeVar; mixed lists
            # (e.g. [sqrt, log2, None]) are valid at runtime, so cast to Any.
            suggested[name] = trial.suggest_categorical(name, cast("list[Any]", spec.choices))
            continue
        if spec.low is None or spec.high is None:  # pragma: no cover - schema-validated
            raise TrainingError(f"Search space '{name}' is missing low/high bounds")
        if spec.type == "int":
            suggested[name] = trial.suggest_int(name, int(spec.low), int(spec.high), log=spec.log)
        else:
            suggested[name] = trial.suggest_float(
                name, float(spec.low), float(spec.high), log=spec.log
            )
    return suggested


def _objective(
    trial: Trial,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task: Task,
    name: AlgorithmName,
    candidate: CandidateConfig,
    seed: int,
    cv: KFold,
    scoring: ScoringName,
) -> float:
    """One trial: suggest params, cross-validate the pipeline, return the mean score."""
    params = {**candidate.fixed, **suggest_params(trial, candidate.space)}
    estimator = build_estimator(task, name, params, seed=seed)
    pipeline = build_pipeline(estimator, scale=candidate.scale)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=_SCORERS[scoring], n_jobs=1)
    return float(scores.mean())


def tune_candidate(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task: Task,
    name: AlgorithmName,
    candidate: CandidateConfig,
    training: TrainingConfig,
    seed: int,
) -> TuningResult:
    """Run one seeded TPE study for candidate ``name`` and return its best trial.

    Args:
        X: Training-split feature matrix.
        y: Training-split target vector.
        task: ML task (selects registry and CV strategy).
        name: Registry algorithm identifier.
        candidate: Registry entry (scale policy, fixed params, search space).
        training: Protocol (folds, shuffle, trials, scoring) from configs/.
        seed: Global randomness seed (sampler + CV splits + estimators).

    Returns:
        The best trial (params + mean CV score) for this algorithm.

    Raises:
        TrainingError: The Optuna study failed entirely (e.g. every trial
            raised during estimator fitting).
    """
    cv = KFold(n_splits=training.cv_folds, shuffle=training.shuffle_cv, random_state=seed)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    try:
        study.optimize(
            lambda trial: _objective(
                trial,
                X,
                y,
                task=task,
                name=name,
                candidate=candidate,
                seed=seed,
                cv=cv,
                scoring=training.scoring,
            ),
            n_trials=training.n_trials,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise TrainingError(f"Tuning of '{name}' failed: {exc}") from exc
    logger.info(
        "Tuned '%s': best %s=%.4f over %d trials",
        name,
        training.scoring,
        study.best_value,
        training.n_trials,
    )
    return TuningResult(
        algorithm=name,
        best_params=dict(study.best_params),
        best_score=float(study.best_value),
        n_trials=training.n_trials,
    )


def select_best(results: Sequence[TuningResult]) -> TuningResult:
    """Return the candidate with the highest cross-validated score.

    Raises:
        TrainingError: The results list is empty.
    """
    if not results:
        raise TrainingError("Cannot select a model from an empty tuning results list")
    return max(results, key=lambda result: result.best_score)
