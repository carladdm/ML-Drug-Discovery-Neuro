"""Command-line interface (Fase 5 / T5.2).

Every command composes the Hydra config from its variadic overrides
(``mldd train target=maob task=regression``) and delegates to the I/O
boundary modules: :mod:`mldd.io` for ingestion/featurization,
:mod:`mldd.trainer` for tracked training runs. ``typer.echo`` is the
user-facing output channel of the CLI (never ``print()`` in library code).
"""

from __future__ import annotations

from typing import Any, Final

import typer

from mldd import __version__, dataset, features, io, trainer
from mldd.config import AppConfig, load_config
from mldd.constants import BIOACTIVITY_CLASS, MOLECULE_ID, PCHEMBL_VALUE, SMILES
from mldd.logging_utils import configure_logging

#: Variadic Hydra overrides; module-level singleton keeps flake8-bugbear (B008) happy.
_OVERRIDES_ARGUMENT: Any = typer.Argument(None)
#: Documented CLI shorthand (AGENTS.md): ``task=regression`` == ``training=regression``.
_TASK_PREFIX: Final = "task="

app = typer.Typer(
    name="mldd",
    help="mldd — neurodegenerative bioactivity prediction pipeline.",
    add_completion=False,
)


def _compose_config(overrides: list[str] | None) -> AppConfig:
    """Compose the config tree, accepting ``task=`` as shorthand for ``training=``."""
    translated = [
        f"training={item[len(_TASK_PREFIX) :]}" if item.startswith(_TASK_PREFIX) else item
        for item in (overrides or [])
    ]
    return load_config(translated)


@app.callback()
def _root() -> None:
    """Explicit callback so the app behaves as a multi-command group."""


@app.command()
def version() -> None:
    """Print the installed mldd version."""
    typer.echo(__version__)


@app.command()
def ingest(overrides: list[str] = _OVERRIDES_ARGUMENT) -> None:
    """Download and curate one target: ``mldd ingest target=lrrk2``.

    Fetches the raw ChEMBL activities (NB01 query), runs the NB02 curation
    chain and persists both CSVs under ``data/``. LRRK2 additionally writes
    its WT/G2019S variant datasets via the configured regex split.
    """
    configure_logging()
    cfg = _compose_config(overrides)
    raw = io.fetch_raw_activities(cfg.target, cfg.data)
    io.save_raw_csv(raw, cfg.raw_path)

    curated = dataset.curate_bioactivity(raw, cfg.data)
    io.save_raw_csv(curated, cfg.curated_path)
    typer.echo(f"Curated {len(curated)} records -> {cfg.curated_path}")

    if cfg.target.variant_split is not None:
        frames = dataset.split_lrrk2_variants(curated, cfg.target.variant_split)
        split = cfg.target.variant_split
        for label, frame in (
            (split.wt_label, frames.wild_type),
            (split.g2019s_label, frames.mutant),
        ):
            variant_path = cfg.data_path / cfg.paths.curated_template.format(
                dataset=f"{cfg.target.dataset}_{label}"
            )
            io.save_raw_csv(frame, variant_path)
            typer.echo(f"Variant '{label}': {len(frame)} records -> {variant_path}")


@app.command()
def featurize(overrides: list[str] = _OVERRIDES_ARGUMENT) -> None:
    """Build the model matrix: ``mldd featurize target=maob``.

    NB03/NB04/NB05 chain: standardized structures + Lipinski descriptors,
    single-pass Z-score outlier removal, PaDEL PubChem fingerprints on the
    surviving rows, variance filters (pre bits, post merge) and the final
    X+y matrix under ``data/processed/``.
    """
    configure_logging()
    cfg = _compose_config(overrides)
    curated = io.load_raw_csv(cfg.curated_path)
    prepared = features.prepare_physchem_features(curated)
    outliers_free = features.remove_zscore_outliers(prepared, cfg.features)

    # PaDEL consumes the raw canonical_smiles of the surviving rows (NB04).
    fingerprints = io.compute_pubchem_fingerprints(
        outliers_free[SMILES], outliers_free[MOLECULE_ID], cfg.features
    )
    filtered_bits = features.filter_low_variance(fingerprints, cfg.features.variance_threshold_pre)
    targets = outliers_free[[BIOACTIVITY_CLASS, PCHEMBL_VALUE]]
    matrix = features.merge_fingerprints_and_targets(filtered_bits, targets)
    matrix = features.filter_low_variance(matrix, cfg.features.variance_threshold_post)
    io.save_raw_csv(matrix, cfg.matrix_path)
    typer.echo(f"Model matrix {matrix.shape[0]}x{matrix.shape[1]} -> {cfg.matrix_path}")


@app.command()
def train(overrides: list[str] = _OVERRIDES_ARGUMENT) -> None:
    """Tune the registry and train one target/task: ``mldd train target=maob task=regression``."""
    configure_logging()
    cfg = _compose_config(overrides)
    run = trainer.run_training(cfg, cfg.matrix_path)
    metrics = " ".join(f"{name}={value:.4f}" for name, value in sorted(run.metrics.items()))
    typer.echo(f"{run.target} [{run.task.value}] winner={run.algorithm} {metrics}")
    typer.echo(f"model={run.model_path} mlflow_run={run.mlflow_run_id}")


if __name__ == "__main__":
    app()
