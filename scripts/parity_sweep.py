"""Resumable parity sweep for T5.3: train every target x task.

Runs the production training workflow (:func:`mldd.trainer.run_training`)
for every target declared in ``configs/target/`` and both training
protocols, persisting results to ``reports/parity_results.json`` after
each run so the sweep survives interruption (re-running skips completed
entries). Target discovery is config-driven — no target list lives here.

Usage:
    python scripts/parity_sweep.py
"""

from __future__ import annotations

import json
import sys
from typing import TypedDict, cast

from mldd.config import BASE_DIR, AppConfig, load_config
from mldd.trainer import run_training

RESULTS_PATH = BASE_DIR / "reports" / "parity_results.json"
TASKS: tuple[str, ...] = ("regression", "classification")


class TaskResult(TypedDict):
    """Record of one completed training run (task-specific holdout keys)."""

    algorithm: str
    cv_score: float
    holdout: dict[str, float]
    mlflow_run_id: str


class TaskFailure(TypedDict):
    """Record of one failed run (the sweep keeps going)."""

    error: str


SweepResults = dict[str, dict[str, TaskResult | TaskFailure]]


def _discover_targets() -> list[str]:
    """Target keys from the Hydra config group (yaml stems, alphabetical)."""
    return sorted(path.stem for path in (BASE_DIR / "configs" / "target").glob("*.yaml"))


def _load_results() -> SweepResults:
    if not RESULTS_PATH.is_file():
        return {}
    return cast(SweepResults, json.loads(RESULTS_PATH.read_text(encoding="utf-8")))


def _save_results(results: SweepResults) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    results: SweepResults = _load_results()
    failures = 0
    for target in _discover_targets():
        for task in TASKS:
            if task in results.get(target, {}):
                print(f"[skip] {target}/{task}: already recorded", flush=True)
                continue
            cfg: AppConfig = load_config([f"target={target}", f"training={task}"])
            print(f"[run ] {target}/{task}: matrix={cfg.matrix_path.name}", flush=True)
            try:
                run = run_training(cfg, cfg.matrix_path)
            except Exception as exc:  # the sweep must survive a failed target
                failures += 1
                results.setdefault(target, {})[task] = {"error": f"{type(exc).__name__}: {exc}"}
                _save_results(results)
                print(f"[fail] {target}/{task}: {exc}", flush=True)
                continue
            holdout = {name: value for name, value in run.metrics.items() if name != "cv_score"}
            results.setdefault(target, {})[task] = {
                "algorithm": str(run.algorithm),
                "cv_score": float(run.metrics["cv_score"]),
                "holdout": holdout,
                "mlflow_run_id": run.mlflow_run_id,
            }
            _save_results(results)
            print(
                f"[done] {target}/{task}: {run.algorithm} cv={run.metrics['cv_score']:.4f}",
                flush=True,
            )
    print(f"Sweep complete: {len(results)} target(s) recorded, {failures} failure(s).", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
