"""Type-checking shim for rdkit (mypy only — never imported at runtime).

RDKit ships vendored stubs (``rdkit-stubs``) that are malformed — e.g.
``typing.ClassVar[rdkit.rdBase._vectunsigned int]`` — which makes mypy abort
with a fatal parse error in every recent release (checked: 2025.9.6, 2026.3.6,
2026.9.1). This package shadows rdkit on ``mypy_path`` (see ``pyproject.toml``)
with minimal ``Any``-typed surfaces covering the APIs used by mldd
(``Chem`` basics, descriptors, and the ``MolStandardize`` standardizers).
"""

from typing import Any

Chem: Any
rdBase: Any
