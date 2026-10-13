"""ChEMBL bioactivity ingestion and pIC50 label transformation.

.. note::
    Fase 2 / T2.2-T2.3 will rewrite this module to reproduce the thesis
    science exactly: assay-type filtering (B/F), the LRRK2 WT/G2019S variant
    split, ChEMBL ``pchembl_value`` rescue and InChIKey deduplication.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client

from mldd.constants import (
    BIOACTIVITY_CLASS,
    MOLECULE_ID,
    PCHEMBL_VALUE,
    SMILES,
    STANDARD_VALUE,
    TARGET_ID,
)
from mldd.logging_utils import get_logger

logger: logging.Logger = get_logger(__name__)


class MultiTargetBioactivityIngestor:
    """Handles ingestion from ChEMBL and pIC50 computation for multiple targets."""

    def __init__(self, target_chembl_ids: list[str]) -> None:
        self.target_chembl_ids = target_chembl_ids

    def fetch_all_targets_data(self) -> pd.DataFrame:
        """Download IC50 records for each configured target."""
        all_records: list[pd.DataFrame] = []
        activity = new_client.activity  # Official ChEMBL API client

        for target_id in self.target_chembl_ids:
            logger.info("Fetching IC50 records for ChEMBL target: %s", target_id)
            try:
                # Filter the API for IC50 records of the specific target
                res = activity.filter(target_chembl_id=target_id).filter(standard_type="IC50")
                df_target = pd.DataFrame(list(res))
                if not df_target.empty:
                    df_target[TARGET_ID] = target_id
                    all_records.append(df_target)
            except Exception as exc:
                logger.error("Error downloading ChEMBL data for %s: %s", target_id, exc)

        if not all_records:
            raise ValueError("No data could be retrieved from ChEMBL.")

        df_combined = pd.concat(all_records, ignore_index=True)
        logger.info("Total records fetched: %d", len(df_combined))
        return df_combined

    def preprocess_and_calculate_pic50(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the exact mathematical logic of Notebook 02."""
        logger.info("Removing null values and computing pIC50...")
        # 1. Keep records with both a SMILES structure and an IC50 value
        df_clean = df.dropna(subset=[STANDARD_VALUE, SMILES]).copy()
        df_clean[STANDARD_VALUE] = pd.to_numeric(df_clean[STANDARD_VALUE], errors="coerce")
        df_clean = df_clean[df_clean[STANDARD_VALUE] > 0]  # Drop values <= 0

        # 2. Mathematical conversion to pIC50: -log10(IC50 in Molar).
        # standard_value is in nM (10^-9 M), hence the 1e-9 multiplication.
        df_clean[PCHEMBL_VALUE] = -np.log10(df_clean[STANDARD_VALUE] * 1e-9)

        # 3. Binary classification (Thesis rule):
        #    Actives:   pIC50 >= 6.0 (IC50 <= 1000 nM)
        #    Inactives: pIC50 <= 5.0 (IC50 >= 10000 nM)
        def assign_class(pIC50_val: float) -> int | None:
            if pIC50_val >= 6.0:
                return 1  # Active
            if pIC50_val <= 5.0:
                return 0  # Inactive
            return None  # Intermediate zone (excluded from binary classification)

        df_clean[BIOACTIVITY_CLASS] = df_clean[PCHEMBL_VALUE].apply(assign_class)
        df_final = df_clean.dropna(subset=[BIOACTIVITY_CLASS]).copy()
        df_final[BIOACTIVITY_CLASS] = df_final[BIOACTIVITY_CLASS].astype(int)

        return df_final[
            [MOLECULE_ID, TARGET_ID, SMILES, STANDARD_VALUE, PCHEMBL_VALUE, BIOACTIVITY_CLASS]
        ]
