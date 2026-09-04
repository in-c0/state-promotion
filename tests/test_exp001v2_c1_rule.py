"""C1's pass criterion and selection rule, verified before any v2 score exists.

Issue #6 section 2. C1 exists because Phase-B v1's negative result was localised
to slow consolidation collapsing the current segment to chance. The criterion is
set deliberately above B5's frozen `current_after >= 0.45` acceptance threshold,
so that "C1 passed" cannot mean "consolidation is barely gate-compatible".
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("c1", ROOT / "scripts" / "run_exp001v2_c1.py")
c1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c1)


def cell(batch, lr, accs, valid=True):
    mean = sum(accs) / len(accs)
    worst = min(accs)
    return {
        "cell": c1.cell_key(batch, lr), "consolidation_batch": batch, "slow_lr": lr,
        "per_seed_slow_only": accs, "mean_slow_only": mean, "worst_seed_slow_only": worst,
        "valid": valid,
        "passes": valid and mean >= c1.PASS_MEAN_MIN and worst >= c1.PASS_WORST_MIN,
        "invalidation_reasons": [] if valid else ["x"],
    }


def test_grid_is_the_predeclared_nine_cells():
    assert c1.BATCHES == (1, 2, 4)
    assert c1.SLOW_LRS == (1e-3, 2e-3, 3e-3)
    assert c1.SLOW_STEPS == 48


def test_pass_thresholds_are_the_issue_6_values():
    assert c1.PASS_MEAN_MIN == 4 / 6
    assert c1.PASS_WORST_MIN == 3 / 6


def test_criterion_is_strictly_above_b5s_frozen_acceptance_threshold():
    """If C1 passed at B5's own 0.45, passing would prove nothing about adequacy."""
    assert c1.PASS_MEAN_MIN > 0.45
    assert c1.PASS_WORST_MIN > 0.45


def test_cell_failing_the_mean_does_not_pass():
    assert not cell(1, 1e-3, [0.667, 0.5, 0.667, 0.5, 0.5])["passes"]


def test_cell_failing_one_seed_does_not_pass_despite_a_good_mean():
    """Every seed must clear 0.5; a strong mean cannot carry a weak seed."""
    c = cell(1, 1e-3, [1.0, 1.0, 1.0, 1.0, 0.333])
    assert c["mean_slow_only"] > c1.PASS_MEAN_MIN
    assert not c["passes"]


def test_cell_meeting_both_thresholds_passes():
    assert cell(1, 1e-3, [0.667, 0.667, 0.833, 0.667, 0.667])["passes"]


def test_smallest_batch_wins_even_when_a_larger_batch_scores_higher():
    """Batch is adaptation-token cost; C1 buys the cheapest adequate config."""
    cells = [
        cell(1, 1e-3, [0.667, 0.667, 0.667, 0.667, 0.667]),
        cell(4, 3e-3, [1.0, 1.0, 1.0, 1.0, 1.0]),
    ]
    assert c1.select(cells)["consolidation_batch"] == 1


def test_within_a_batch_highest_worst_seed_wins():
    cells = [
        cell(2, 1e-3, [1.0, 1.0, 1.0, 1.0, 0.5]),
        cell(2, 2e-3, [0.833, 0.833, 0.833, 0.833, 0.667]),
    ]
    assert c1.select(cells)["slow_lr"] == 2e-3


def test_ties_on_worst_seed_go_to_higher_mean_then_lower_lr():
    a = cell(1, 2e-3, [0.667, 0.667, 1.0, 0.667, 0.667])
    b = cell(1, 3e-3, [0.667, 0.667, 0.667, 0.667, 0.667])
    assert c1.select([a, b])["slow_lr"] == 2e-3
    same_hi = cell(1, 1e-3, [0.667, 0.667, 1.0, 0.667, 0.667])
    assert c1.select([a, same_hi])["slow_lr"] == 1e-3


def test_no_passing_cell_blocks_c2_and_therefore_b5():
    """If consolidation is inadequate, v2 stops rather than testing B5."""
    cells = [cell(1, 1e-3, [0.333, 0.333, 0.333, 0.333, 0.333])]
    assert c1.select(cells) is None


def test_invalid_cell_cannot_be_selected():
    cells = [cell(1, 1e-3, [1.0] * 5, valid=False), cell(4, 1e-3, [0.833] * 5)]
    assert c1.select(cells)["consolidation_batch"] == 4
