"""Guard-rail tests: the target registry (configs/target/*.yaml) matches the thesis.

Notebook 01 outputs + README are the reference. These tests fail if someone
reintroduces the incorrect ChEMBL accessions from the early draft
(CHEMBL1908/CHEMBL204/CHEMBL203) or breaks the 6-datasets/4-targets topology.
"""

from __future__ import annotations

import pytest

from mldd.config import load_config
from mldd.constants import Task
from mldd.exceptions import ConfigError

#: dataset -> (Hydra option, expected ChEMBL accession)
_EXPECTED: dict[str, tuple[str, str]] = {
    "AChE": ("ache", "CHEMBL220"),
    "MAO-B": ("maob", "CHEMBL2039"),
    "GSK3B": ("gsk3b", "CHEMBL262"),
    "LRRK2": ("lrrk2", "CHEMBL1075104"),
    "LRRK2_WT": ("lrrk2_wt", "CHEMBL1075104"),
    "LRRK2_G2019S": ("lrrk2_g2019s", "CHEMBL1075104"),
}


@pytest.mark.parametrize(
    ("dataset", "option", "chembl_id"), [(d, o, c) for d, (o, c) in _EXPECTED.items()]
)
def test_target_ids_match_thesis(dataset: str, option: str, chembl_id: str) -> None:
    cfg = load_config([f"target={option}"])
    assert cfg.target.dataset == dataset
    assert cfg.target.chembl_id == chembl_id


def test_six_datasets_four_unique_targets() -> None:
    ids = {load_config([f"target={opt}"]).target.chembl_id for (_, (opt, _)) in _EXPECTED.items()}
    assert len(_EXPECTED) == 6
    assert len(ids) == 4


def test_lrrk2_variants_share_parent_target() -> None:
    parent = load_config(["target=lrrk2"]).target
    for option in ("lrrk2_wt", "lrrk2_g2019s"):
        assert load_config([f"target={option}"]).target.chembl_id == parent.chembl_id


def test_mao_b_guard_rail_rejects_chembl203() -> None:
    """The trap from AGENTS.md: MAO-B must never use the CHEMBL203 accession.

    The ID itself lives only in ``configs/target/maob.yaml`` (never in ``src/``);
    this test pins that file to the correct value.
    """
    assert load_config(["target=maob"]).target.chembl_id == "CHEMBL2039"
    # And the schema rejects structurally malformed accessions on override.
    with pytest.raises(ConfigError):
        load_config(["target=maob", "target.chembl_id=203"])


def test_task_enum_values() -> None:
    assert Task.CLASSIFICATION.value == "classification"
    assert Task.REGRESSION.value == "regression"
