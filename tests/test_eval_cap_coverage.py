"""A pilot-only evaluation cap must buy coverage, not repeats.

PALS emits `test_repeats` consecutive copies of each mapping, so a positional
cap scored the same few mappings twice instead of sampling the segment.
"""
from state_promotion.lm import cap_distinct_examples
from state_promotion.pals import generate_retention_stream


def segment_test_examples(segment=0):
    examples, _ = generate_retention_stream(20260901)
    return [e for e in examples if e.segment == segment and e.split == "test"]


def test_cap_selects_distinct_mappings_before_repeats():
    test = segment_test_examples()
    capped = cap_distinct_examples(test, 4)
    assert len(capped) == 4
    assert len({(e.context, e.key) for e in capped}) == 4


def test_positional_cap_would_have_covered_only_half_as_many_mappings():
    """Regression pin for the defect this replaced."""
    test = segment_test_examples()
    assert len({(e.context, e.key) for e in test[:4]}) == 2
    assert len({(e.context, e.key) for e in cap_distinct_examples(test, 4)}) == 4


def test_cap_above_distinct_count_falls_back_to_repeats():
    test = segment_test_examples()
    distinct = len({(e.context, e.key) for e in test})
    capped = cap_distinct_examples(test, distinct + 3)
    assert len(capped) == distinct + 3


def test_cap_at_or_above_full_set_returns_every_example():
    test = segment_test_examples()
    assert len(cap_distinct_examples(test, len(test))) == len(test)
