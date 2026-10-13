"""Any-typed surface for ``rdkit.Chem`` (see ``typings/rdkit/__init__.pyi``)."""

from typing import Any

# Molecule construction / serialization
MolFromSmiles: Any
MolToSmiles: Any
MolFromInchi: Any
MolToInchiKey: Any

# Descriptor calculators (MolWt, MolLogP, ...)
Descriptors: Any
Lipinski: Any
