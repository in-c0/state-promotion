"""The B3 selection rule is verified before any B3 or B5 score exists.

Issue #5 section 4 selects a deliberately *strong* fixed baseline, so that B5
cannot later be flattered by a weak comparator. The property that matters is
that a cell cannot win by failing to learn: low forgetting is only reachable
from inside the eligibility band defined by the best observed diagonal.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "b3grid", ROOT / "scripts" / "run_exp001b_b3_grid.py"
)
b3grid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b3grid)


def cell(k, lr, diag, forget, auc=0.5, final=0.5, valid=True):
    return {
        "cell": b3grid.cell_key(k, lr), "cadence_k": k, "slow_lr": lr,
        "mean_diagonal": diag, "mean_average_forgetting": forget,
        "mean_retention_auc": auc, "mean_final_average": final,
        "valid": valid, "invalidation_reasons": [] if valid else ["x"],
    }


def test_grid_is_the_predeclared_nine_cells():
    assert b3grid.CADENCES == (1, 2, 3)
    assert b3grid.SLOW_LRS == (1e-3, 2e-3, 3e-3)
    assert b3grid.DEV_SEEDS == (20260901, 20260902, 20260903)
    assert len(b3grid.CADENCES) * len(b3grid.SLOW_LRS) == 9


def test_lowest_forgetting_wins_among_eligible():
    cells = [cell(1, 1e-3, 0.90, 0.40), cell(2, 2e-3, 0.90, 0.20)]
    assert b3grid.select(cells)["cadence_k"] == 2


def test_a_cell_cannot_win_by_failing_to_learn():
    """The whole point of the eligibility band: 0 forgetting at chance is not a
    strong baseline, it is a broken one."""
    cells = [
        cell(1, 1e-3, 0.95, 0.40),   # learns, forgets
        cell(3, 3e-3, 0.17, 0.00),   # chance accuracy, nothing to forget
    ]
    assert b3grid.select(cells)["cadence_k"] == 1


def test_eligibility_band_is_relative_to_best_observed_diagonal():
    cells = [cell(1, 1e-3, 1.00, 0.50), cell(2, 1e-3, 0.951, 0.10), cell(3, 1e-3, 0.949, 0.01)]
    chosen = b3grid.select(cells)
    assert chosen["cadence_k"] == 2, "0.949 < 0.95*1.00 must be excluded despite lowest forgetting"


def test_tie_breaks_in_declared_order():
    # equal forgetting -> higher retention AUC
    assert b3grid.select([
        cell(1, 1e-3, 0.9, 0.2, auc=0.40), cell(2, 1e-3, 0.9, 0.2, auc=0.60),
    ])["cadence_k"] == 2
    # equal forgetting and AUC -> higher final average
    assert b3grid.select([
        cell(1, 1e-3, 0.9, 0.2, auc=0.5, final=0.40),
        cell(2, 1e-3, 0.9, 0.2, auc=0.5, final=0.60),
    ])["cadence_k"] == 2
    # equal above -> lower slow LR
    assert b3grid.select([
        cell(1, 3e-3, 0.9, 0.2), cell(1, 1e-3, 0.9, 0.2),
    ])["slow_lr"] == 1e-3
    # equal above -> smaller cadence
    assert b3grid.select([
        cell(3, 1e-3, 0.9, 0.2), cell(1, 1e-3, 0.9, 0.2),
    ])["cadence_k"] == 1


def test_invalid_cells_are_never_selected_and_never_set_dmax():
    cells = [cell(1, 1e-3, 0.99, 0.01, valid=False), cell(2, 1e-3, 0.60, 0.30)]
    chosen = b3grid.select(cells)
    assert chosen["cadence_k"] == 2, "an invalid cell must not win"


def test_no_valid_cells_blocks_b5():
    assert b3grid.select([cell(1, 1e-3, 0.9, 0.1, valid=False)]) is None


def test_empty_grid_blocks_b5():
    assert b3grid.select([]) is None
