"""EXP-001R Amendment G: gate runners must adapt in the protocol's order.

PALS generation order is blocked (train_repeats consecutive copies per mapping).
The protocol runner shuffles each segment. A gate runner that trains raw
generation order collapses onto the last mapping seen and reports a capacity
failure that is really a presentation-order confound.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

from state_promotion.pals import generate_retention_stream, protocol_train_order

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260901


def _segment_train(seed: int) -> dict[int, list]:
    examples, _ = generate_retention_stream(seed)
    return {
        seg: [e for e in examples if e.segment == seg and e.split == "train"]
        for seg in range(6)
    }


def _run_lengths(keys: list[str]) -> list[int]:
    runs: list[int] = []
    for k in keys:
        if runs and k == _run_lengths.prev:
            runs[-1] += 1
        else:
            runs.append(1)
        _run_lengths.prev = k
    return runs


_run_lengths.prev = None


def test_pals_generation_order_is_blocked_and_is_the_hazard():
    train = _segment_train(SEED)[0]
    _run_lengths.prev = None
    runs = _run_lengths([e.key for e in train])
    assert runs == [8, 8, 8, 8, 8, 8], f"expected blocked generation order, got {runs}"


def test_protocol_order_matches_run_lm_pals_algorithm_exactly():
    """Reproduce run_lm_pals.py's ordering independently and require equality."""
    segments = _segment_train(SEED)
    rng = random.Random(SEED)
    expected = {}
    for seg in sorted(segments):
        items = list(segments[seg])
        rng.shuffle(items)
        expected[seg] = items

    actual = protocol_train_order(segments, SEED)
    for seg in range(6):
        assert [e.key for e in actual[seg]] == [e.key for e in expected[seg]]
        assert [e.target for e in actual[seg]] == [e.target for e in expected[seg]]


def test_protocol_order_breaks_the_blocked_runs():
    ordered = protocol_train_order(_segment_train(SEED), SEED)[0]
    _run_lengths.prev = None
    runs = _run_lengths([e.key for e in ordered])
    assert max(runs) < 8, f"protocol order still blocked: {runs}"


def test_protocol_order_preserves_the_exposure_multiset():
    segments = _segment_train(SEED)
    ordered = protocol_train_order(segments, SEED)
    for seg in range(6):
        assert sorted((e.key, e.target) for e in ordered[seg]) == sorted(
            (e.key, e.target) for e in segments[seg]
        )
        assert len(ordered[seg]) == 48


def test_protocol_order_is_seed_deterministic_and_seed_sensitive():
    segments = _segment_train(SEED)
    a = protocol_train_order(segments, SEED)[0]
    b = protocol_train_order(segments, SEED)[0]
    c = protocol_train_order(segments, SEED + 1)[0]
    assert [e.key for e in a] == [e.key for e in b]
    assert [e.key for e in a] != [e.key for e in c]


def test_protocol_order_does_not_mutate_its_input():
    segments = _segment_train(SEED)
    before = [e.key for e in segments[0]]
    protocol_train_order(segments, SEED)
    assert [e.key for e in segments[0]] == before


def test_gate_runners_do_not_adapt_in_raw_generation_order():
    """Both EXP-001R gate runners must route adaptation through the helper."""
    for name in ("run_exp001r_representation.py", "run_exp001r_sequential.py"):
        src = (ROOT / "scripts" / name).read_text()
        assert "protocol_train_order" in src, f"{name} does not use protocol_train_order"


def test_protocol_runner_still_uses_the_shuffle_this_helper_mirrors():
    """If run_lm_pals.py's ordering changes, this helper is stale -- fail loudly."""
    src = (ROOT / "scripts" / "run_lm_pals.py").read_text()
    assert re.search(r"rng\s*=\s*random\.Random\(seed\)", src)
    assert re.search(r"train\s*=\s*list\(segments\[seg\]\[.train.\]\)", src)
    assert re.search(r"rng\.shuffle\(train\)", src)
