"""Neutralise rdkit's broken vendored stubs so mypy can run.

rdkit ships a PEP 561 stub package (``rdkit-stubs``) whose generated ``.pyi``
files are malformed — e.g. ``typing.ClassVar[rdkit.rdBase._vectunsigned int]`` —
which makes mypy abort with a fatal parse error on every recent release
(verified: 2025.9.6, 2026.3.6, 2026.9.1). Because mypy gives installed stub
packages precedence over ``mypy_path``, the ``typings/rdkit`` shim cannot
shadow them.

This script deletes the broken stub directory from site-packages. Runtime
behaviour is unaffected: rdkit's importable code lives in the ``rdkit``
package, never in ``rdkit-stubs``. The script is idempotent and runs before
``mypy`` via the Makefile / CI.

Remove this workaround once upstream rdkit ships parseable stubs.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> int:
    """Delete ``rdkit-stubs`` from the active site-packages if present."""
    try:
        import rdkit
    except ImportError:
        print("rdkit not installed — nothing to do.")
        return 0

    if rdkit.__file__ is None:
        print("rdkit has no __file__ — nothing to do.")
        return 0

    site_packages = Path(rdkit.__file__).resolve().parent.parent
    stubs = site_packages / "rdkit-stubs"
    if stubs.is_dir():
        shutil.rmtree(stubs)
        print(f"Removed broken rdkit stubs: {stubs}")
    else:
        print("rdkit stubs already clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
