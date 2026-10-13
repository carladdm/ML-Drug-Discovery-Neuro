"""Project-wide paths, seed and baseline model hyper-parameters.

.. note::
    Temporary bridge (Fase 1 / T1.1): static settings will move to Hydra
    configs under ``configs/`` validated with Pydantic. Only the paths resolved
    from the package location will remain here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from mldd.constants import ALL_TARGETS

#: Repository root (src/mldd/config.py -> up 3 levels).
BASE_DIR: Final = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Final = BASE_DIR / "data"
RAW_DATA_DIR: Final = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final = DATA_DIR / "processed"
MODELS_DIR: Final = BASE_DIR / "models"
REPORTS_DIR: Final = BASE_DIR / "reports" / "figures"

#: Global randomness seed for reproducibility.
SEED: Final = 42

#: Unique ChEMBL IDs of the four thesis targets (TODO Fase 1: replaced by
#: per-dataset Hydra configs).
TARGET_CHEMBL_IDS: Final[list[str]] = list(dict.fromkeys(t.chembl_id for t in ALL_TARGETS))

#: Baseline Random Forest hyper-parameters.
#: TODO (Fase 4 / T4.1-T4.3): superseded by per-target Optuna-tuned parameters
#: stored under ``configs/models/frozen/``.
MODEL_PARAMS: Final[dict[str, Any]] = {
    "n_estimators": 500,
    "max_depth": 15,
    "min_samples_split": 5,
    "random_state": SEED,
    "n_jobs": -1,
}
