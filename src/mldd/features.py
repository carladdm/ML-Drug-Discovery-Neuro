"""Molecular feature extraction (Lipinski descriptors via RDKit).

.. note::
    Fase 3 / T3.2-T3.3 extends this module with PaDEL PubChem fingerprints
    (881 bits), variance filtering and Z-score outlier removal.
"""

from __future__ import annotations

import logging

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from mldd.constants import (
    LOGP,
    MW,
    NUM_H_ACCEPTORS,
    NUM_H_DONORS,
    SMILES,
)
from mldd.logging_utils import get_logger

logger: logging.Logger = get_logger(__name__)


class MolecularFeatureExtractor:
    """Computes molecular physicochemical properties via RDKit."""

    @staticmethod
    def compute_lipinski_descriptors(df: pd.DataFrame, smiles_col: str = SMILES) -> pd.DataFrame:
        """Iterate over each SMILES code and extract Lipinski parameters.

        Args:
            df: Input dataframe containing a SMILES column.
            smiles_col: Name of the SMILES column to process.

        Returns:
            Dataframe with ``MW``, ``LogP``, ``NumHDonors`` and
            ``NumHAcceptors`` columns; rows with invalid SMILES are dropped.
        """
        logger.info("Computing Lipinski descriptors with RDKit...")
        mw: list[float | None] = []
        logp: list[float | None] = []
        hbd: list[int | None] = []
        hba: list[int | None] = []

        for smiles in df[smiles_col]:
            mol = Chem.MolFromSmiles(smiles)  # Convert SMILES text to RDKit Mol
            if mol is not None:
                mw.append(Descriptors.MolWt(mol))  # Molecular Weight (MW)
                logp.append(Descriptors.MolLogP(mol))  # Partition coefficient (LogP)
                hbd.append(Lipinski.NumHDonors(mol))  # H-bond donors
                hba.append(Lipinski.NumHAcceptors(mol))  # H-bond acceptors
            else:
                # Corrupted/invalid SMILES -> record null
                mw.append(None)
                logp.append(None)
                hbd.append(None)
                hba.append(None)

        df[MW] = mw
        df[LOGP] = logp
        df[NUM_H_DONORS] = hbd
        df[NUM_H_ACCEPTORS] = hba

        # Drop rows with corrupted or invalid SMILES
        df_cleaned = df.dropna(subset=[MW, LOGP, NUM_H_DONORS, NUM_H_ACCEPTORS])
        logger.info("Descriptors computed for %d/%d molecules", len(df_cleaned), len(df))
        return df_cleaned
