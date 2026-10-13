"""Minimal surface for ``rdkit.Chem.MolStandardize.rdMolStandardize``."""

from typing import Any


class LargestFragmentChooser:
    """Keeps only the largest fragment of a possibly multi-fragment molecule."""

    def choose(self, mol: Any) -> Any: ...


class Uncharger:
    """Neutralizes formal charges where a matching protonation state exists."""

    def uncharge(self, mol: Any) -> Any: ...
