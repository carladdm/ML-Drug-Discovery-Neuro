"""Command-line interface (scaffold).

The full command set (ingest, featurize, tune, train, predict, report) lands in
Fase 5 / T5.2. This scaffold already wires the ``mldd`` console script so the
packaging, Docker image and CI can exercise a real entry point.
"""

from __future__ import annotations

import typer

from mldd import __version__

app = typer.Typer(
    name="mldd",
    help="mldd — neurodegenerative bioactivity prediction pipeline.",
    add_completion=False,
)


@app.callback()
def _root() -> None:
    """Explicit callback so the app behaves as a multi-command group."""


@app.command()
def version() -> None:
    """Print the installed mldd version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
