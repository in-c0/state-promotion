"""B4 must be matched to B5 in commit count AND per-commit allocation.

Issue #5 section 5 and stop condition "B4 cannot be exactly matched to B5".
Matching only the number of commits would let B4 spend a different amount of
slow-write budget per commit, which would make the routing ablation a resource
comparison instead.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "b_matrix", ROOT / "scripts" / "run_exp001b_matrix.py"
)
mx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mx)


def b5(*accepted):
    log = []
    for seg in range(6):
        hit = next((a for a in accepted if a[0] == seg), None)
        log.append({"segment": seg, "accepted": bool(hit),
                    "planned_steps": hit[1] if hit else 0})
    return {"commit_log": log}


def test_commit_count_is_matched():
    commits, steps = mx.match_b4_to_b5(b5((1, 48), (4, 48)), 20260901)
    assert len(commits) == 2
    assert len(steps) == 2


def test_per_commit_allocation_is_matched_not_just_the_count():
    commits, steps = mx.match_b4_to_b5(b5((0, 96), (3, 48)), 20260901)
    assert sorted(steps.values()) == [48, 96]


def test_declining_every_candidate_matches_to_zero_commits():
    commits, steps = mx.match_b4_to_b5(b5(), 20260901)
    assert commits == set()
    assert steps == {}


def test_matching_is_deterministic_per_seed():
    a = mx.match_b4_to_b5(b5((1, 48), (4, 48)), 20260902)
    b = mx.match_b4_to_b5(b5((1, 48), (4, 48)), 20260902)
    assert a == b


def test_matching_differs_across_seeds_so_b4_is_not_b5s_schedule():
    """B4 is a routing ablation: same budget, different placement."""
    placements = {tuple(sorted(mx.match_b4_to_b5(b5((0, 48), (1, 48), (2, 48)), s)[0]))
                  for s in (1, 2, 3, 4, 5, 6, 7, 8)}
    assert len(placements) > 1


def test_all_commits_land_on_real_segments():
    for seed in range(20):
        commits, _ = mx.match_b4_to_b5(b5((0, 48), (5, 48)), seed)
        assert all(0 <= c <= 5 for c in commits)


def test_batch_points_are_predeclared():
    assert mx.BATCH_POINTS == (1, 2, 4)


def test_run_order_puts_baselines_before_b5():
    assert mx.BASELINE_ORDER == ("b0_frozen", "b1_sequential", "b2_replay", "b3_fixed")
    assert "b5_promotion" not in mx.BASELINE_ORDER
    assert "b4_random" not in mx.BASELINE_ORDER
