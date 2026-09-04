"""The v2 development seed list must be reproducible, not asserted.

Issue #6 section 1 fixes five seeds and requires their provenance to be
machine-checkable, so that nobody can later substitute a more convenient list.
"""
from __future__ import annotations

import hashlib

import pytest

from state_promotion.seeds import (
    RETIRED_DEVELOPMENT_SEEDS,
    V1_DEVELOPMENT_SEEDS,
    V2_DEVELOPMENT_LABEL,
    V2_DEVELOPMENT_SEEDS,
    V3_DEVELOPMENT_LABEL,
    V3_DEVELOPMENT_SEEDS,
    derive_seeds,
    v2_development_seeds,
    v3_development_seeds,
)

# Exactly as written in issue #6.
ISSUE_6_SEEDS = (39564119, 59714453, 14200664, 69537273, 73771459)


def test_derivation_reproduces_the_issue_6_list_exactly():
    assert v2_development_seeds() == ISSUE_6_SEEDS


def test_label_is_the_one_the_issue_names():
    assert V2_DEVELOPMENT_LABEL == "EXP-001-v2-consolidation-development-v1"


def test_derivation_is_recomputable_from_first_principles():
    """Recompute independently rather than calling the module's own helper."""
    digest = hashlib.sha256(V2_DEVELOPMENT_LABEL.encode()).digest()
    expected = tuple(
        (int.from_bytes(digest[i * 4:(i + 1) * 4], "big") % 90_000_000) + 10_000_000
        for i in range(5)
    )
    assert expected == ISSUE_6_SEEDS


def test_a_different_label_gives_different_seeds():
    assert derive_seeds(V2_DEVELOPMENT_LABEL + "x", 5) != ISSUE_6_SEEDS


def test_seeds_are_distinct():
    assert len(set(V2_DEVELOPMENT_SEEDS)) == len(V2_DEVELOPMENT_SEEDS) == 5


def test_v2_seeds_are_disjoint_from_the_contaminated_v1_seeds():
    """v1 ordering was inspected on 20260901/02/03; reusing them would be tuning
    on seeds whose answers we already know."""
    assert set(V2_DEVELOPMENT_SEEDS).isdisjoint(set(V1_DEVELOPMENT_SEEDS))


def test_every_seed_has_the_same_width():
    assert all(10_000_000 <= s <= 99_999_999 for s in V2_DEVELOPMENT_SEEDS)


def test_derivation_is_pure():
    assert v2_development_seeds() == v2_development_seeds()


def test_requesting_more_seeds_than_the_digest_supports_raises():
    with pytest.raises(ValueError):
        derive_seeds(V2_DEVELOPMENT_LABEL, 9)


# --- v3 (issue #7) ---

ISSUE_7_SEEDS = (15007679, 42082468, 63400529, 74102599, 48546640)


def test_v3_derivation_reproduces_the_issue_7_list_exactly():
    assert v3_development_seeds() == ISSUE_7_SEEDS


def test_v3_label_is_the_one_issue_7_names():
    assert V3_DEVELOPMENT_LABEL == "EXP-001-v3-embedding-latent-development-v1"


def test_v3_derivation_is_recomputable_from_first_principles():
    digest = hashlib.sha256(V3_DEVELOPMENT_LABEL.encode()).digest()
    expected = tuple(
        (int.from_bytes(digest[i * 4:(i + 1) * 4], "big") % 90_000_000) + 10_000_000
        for i in range(5)
    )
    assert expected == ISSUE_7_SEEDS


def test_v3_uses_the_same_rule_already_committed_for_v2():
    assert v3_development_seeds() == derive_seeds(V3_DEVELOPMENT_LABEL, 5)


def test_v3_seeds_are_disjoint_from_all_retired_development_seeds():
    """v1 saw arm ordering; three v2 seeds saw the latent diagnostic. Neither set
    may be reused for tuning."""
    assert set(V3_DEVELOPMENT_SEEDS).isdisjoint(set(V1_DEVELOPMENT_SEEDS))
    assert set(V3_DEVELOPMENT_SEEDS).isdisjoint(set(V2_DEVELOPMENT_SEEDS))
    assert set(V3_DEVELOPMENT_SEEDS).isdisjoint(set(RETIRED_DEVELOPMENT_SEEDS))


def test_retired_set_covers_v1_and_v2():
    assert set(RETIRED_DEVELOPMENT_SEEDS) == set(V1_DEVELOPMENT_SEEDS) | set(V2_DEVELOPMENT_SEEDS)


def test_v3_seeds_are_distinct_and_uniform_width():
    assert len(set(V3_DEVELOPMENT_SEEDS)) == 5
    assert all(10_000_000 <= s <= 99_999_999 for s in V3_DEVELOPMENT_SEEDS)
