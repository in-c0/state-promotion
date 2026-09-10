"""v4's three seed panels must be reproducible and mutually disjoint (issue #8).

The three-seed EXP-001R representation gate selected a cell scoring 1.000 on all
three seeds that then reached only 0.800 on five fresh ones. v4's answer is to
select on one panel and validate on a second the selection never touched, so the
panels' independence is load-bearing and is asserted here rather than assumed.
"""
from __future__ import annotations

import hashlib

from state_promotion.seeds import (
    RETIRED_DEVELOPMENT_SEEDS,
    V1_DEVELOPMENT_SEEDS,
    V2_DEVELOPMENT_SEEDS,
    V3_DEVELOPMENT_SEEDS,
    V4_DOWNSTREAM_LABEL,
    V4_DOWNSTREAM_SEEDS,
    V4_SELECTION_LABEL,
    V4_SELECTION_SEEDS,
    V4_VALIDATION_LABEL,
    V4_VALIDATION_SEEDS,
)

ISSUE_8_SELECTION = (45019859, 44948853, 26370277, 98113429,
                     69325507, 13342392, 45787057, 54985860)
ISSUE_8_VALIDATION = (96176521, 84165758, 94673303, 37094281,
                      61927228, 32941886, 78804872, 42560972)
ISSUE_8_DOWNSTREAM = (81558746, 44879599, 58406108, 54685515, 32757482)


def _first_principles(label: str, count: int) -> tuple[int, ...]:
    digest = hashlib.sha256(label.encode()).digest()
    return tuple(
        (int.from_bytes(digest[i * 4:(i + 1) * 4], "big") % 90_000_000) + 10_000_000
        for i in range(count)
    )


def test_selection_panel_matches_issue_8():
    assert V4_SELECTION_SEEDS == ISSUE_8_SELECTION
    assert _first_principles(V4_SELECTION_LABEL, 8) == ISSUE_8_SELECTION


def test_validation_panel_matches_issue_8():
    assert V4_VALIDATION_SEEDS == ISSUE_8_VALIDATION
    assert _first_principles(V4_VALIDATION_LABEL, 8) == ISSUE_8_VALIDATION


def test_downstream_panel_matches_issue_8():
    assert V4_DOWNSTREAM_SEEDS == ISSUE_8_DOWNSTREAM
    assert _first_principles(V4_DOWNSTREAM_LABEL, 5) == ISSUE_8_DOWNSTREAM


def test_labels_are_the_ones_issue_8_names():
    assert V4_SELECTION_LABEL == "EXP-001-v4-representation-selection-v1"
    assert V4_VALIDATION_LABEL == "EXP-001-v4-representation-validation-v1"
    assert V4_DOWNSTREAM_LABEL == "EXP-001-v4-downstream-development-v1"


def test_validation_panel_is_disjoint_from_selection_panel():
    """Validation is only independent if it shares no seed with selection."""
    assert set(V4_SELECTION_SEEDS).isdisjoint(set(V4_VALIDATION_SEEDS))


def test_downstream_panel_is_disjoint_from_both_representation_panels():
    assert set(V4_DOWNSTREAM_SEEDS).isdisjoint(set(V4_SELECTION_SEEDS))
    assert set(V4_DOWNSTREAM_SEEDS).isdisjoint(set(V4_VALIDATION_SEEDS))


def test_every_v4_panel_avoids_all_retired_seeds():
    retired = set(RETIRED_DEVELOPMENT_SEEDS)
    assert retired == set(V1_DEVELOPMENT_SEEDS) | set(V2_DEVELOPMENT_SEEDS) | set(V3_DEVELOPMENT_SEEDS)
    for panel in (V4_SELECTION_SEEDS, V4_VALIDATION_SEEDS, V4_DOWNSTREAM_SEEDS):
        assert set(panel).isdisjoint(retired)


def test_all_twentyone_v4_seeds_are_distinct():
    combined = V4_SELECTION_SEEDS + V4_VALIDATION_SEEDS + V4_DOWNSTREAM_SEEDS
    assert len(combined) == 21
    assert len(set(combined)) == 21


def test_panel_sizes_are_the_issue_8_sizes():
    assert len(V4_SELECTION_SEEDS) == 8
    assert len(V4_VALIDATION_SEEDS) == 8
    assert len(V4_DOWNSTREAM_SEEDS) == 5
