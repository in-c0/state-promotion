"""v4 Panel A/B rules, verified before any v4 score exists (issue #8).

The property that matters is structural: Panel B must be capable of failing, and
a Panel-B failure must not be recoverable by reselecting Panel A's runner-up.
That is the whole point of the independent panel, so it is tested rather than
trusted to discipline.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "v4rep", ROOT / "scripts" / "run_exp001v4_representation.py"
)
v4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v4)


def cell(rank, lr, accs, valid=True):
    mean = sum(accs) / len(accs)
    worst = min(accs)
    return {
        "cell": v4.cell_key(rank, lr), "rank": rank, "online_lr": lr,
        "per_seed_accuracy": accs, "mean_accuracy": mean, "worst_seed_accuracy": worst,
        "valid": valid,
        "passes": valid and mean >= v4.MEAN_MIN and worst >= v4.WORST_MIN,
        "invalidation_reasons": [] if valid else ["x"],
    }


def test_grid_is_canonical_and_unchanged_from_exp001r():
    assert v4.RANKS == (1, 2, 4)
    assert v4.LRS == (3e-4, 1e-3, 3e-3)
    assert v4.EXPOSURES == 48


def test_criterion_is_the_unchanged_exp001r_criterion():
    assert v4.MEAN_MIN == 5 / 6
    assert v4.WORST_MIN == 4 / 6


def test_a_cell_failing_one_seed_does_not_pass(fixture=None):
    """Eight seeds, one weak: the per-seed floor is what makes the panel robust."""
    c = cell(2, 3e-3, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5])
    assert c["mean_accuracy"] > v4.MEAN_MIN
    assert not c["passes"]


def test_smallest_rank_wins_then_worst_seed_then_mean_then_lower_lr():
    assert v4.select([cell(4, 3e-4, [1.0] * 8), cell(1, 3e-3, [5 / 6] * 8)])["rank"] == 1
    a = cell(2, 1e-3, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.667])
    b = cell(2, 3e-3, [5 / 6] * 8)
    assert v4.select([a, b])["online_lr"] == 3e-3, "higher worst-seed must win inside a rank"
    hi = cell(2, 3e-3, [1.0, 5 / 6, 5 / 6, 5 / 6, 5 / 6, 5 / 6, 5 / 6, 5 / 6])
    lo = cell(2, 1e-3, [5 / 6] * 8)
    assert v4.select([hi, lo])["online_lr"] == 3e-3, "then higher mean"
    assert v4.select([cell(1, 3e-3, [5 / 6] * 8), cell(1, 3e-4, [5 / 6] * 8)])["online_lr"] == 3e-4


def test_no_passing_cell_blocks_panel_b():
    assert v4.select([cell(1, 3e-4, [0.5] * 8)]) is None


def test_invalid_cell_cannot_be_selected():
    assert v4.select([cell(1, 3e-4, [1.0] * 8, valid=False), cell(4, 3e-4, [5 / 6] * 8)])["rank"] == 4


def test_panel_b_refuses_to_run_when_panel_a_selected_nothing(tmp_path):
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({
        "protocol": "EXP-001v4-panel-a-representation-selection-v1",
        "panel_b_allowed": False, "selected": None,
    }))
    with pytest.raises(ValueError, match="Panel B is forbidden"):
        v4.load_selected(p)


def test_panel_b_refuses_a_foreign_summary(tmp_path):
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({"protocol": "something-else", "panel_b_allowed": True,
                             "selected": {"rank": 2, "online_lr": 3e-3}}))
    with pytest.raises(ValueError, match="unexpected Panel-A summary protocol"):
        v4.load_selected(p)


def test_panel_b_loads_exactly_the_frozen_cell(tmp_path):
    p = tmp_path / "summary.json"
    p.write_text(json.dumps({
        "protocol": "EXP-001v4-panel-a-representation-selection-v1",
        "panel_b_allowed": True,
        "selected": {"rank": 4, "online_lr": 1e-3, "mean_accuracy": 0.9},
    }))
    sel = v4.load_selected(p)
    assert (sel["rank"], sel["online_lr"]) == (4, 1e-3)


def test_there_is_no_reselection_code_path():
    """A Panel-B failure must not be recoverable by promoting the runner-up.
    `select` is never reachable from the Panel-B branch, and Panel B loads a
    single frozen cell rather than a ranked list."""
    src = (ROOT / "scripts" / "run_exp001v4_representation.py").read_text()
    panel_b_branch = src.split('if args.panel == "a":')[1].split("    else:")[1]
    assert "select(" not in panel_b_branch, "Panel B must not reach the selection rule"
    assert "load_selected" in panel_b_branch, "Panel B must load one frozen cell"
    assert '"reselection_permitted": False' in src


def test_panel_b_evaluate_can_actually_fail():
    """A validation gate that cannot fail is not a gate."""
    weak = [{"seed": s, "accuracy": 0.5, "backbone_frozen": True, "finite_losses": True,
             "finite_adapter_parameters": True, "heldout_examples_in_optimization": 0,
             "provenance": {"tokenizer_asset_sha256": {"a": "b"},
                            "resolved_snapshot_commit": v4.MODEL_REVISION}}
            for s in range(8)]
    assert not v4.evaluate(weak, range(8))["passes"]


def test_panel_b_evaluate_passes_a_genuinely_strong_cell():
    strong = [{"seed": s, "accuracy": 1.0, "backbone_frozen": True, "finite_losses": True,
               "finite_adapter_parameters": True, "heldout_examples_in_optimization": 0,
               "provenance": {"tokenizer_asset_sha256": {"a": "b"},
                              "resolved_snapshot_commit": v4.MODEL_REVISION}}
              for s in range(8)]
    assert v4.evaluate(strong, range(8))["passes"]


def test_missing_seed_invalidates_a_cell():
    runs = [{"seed": s, "accuracy": 1.0, "backbone_frozen": True, "finite_losses": True,
             "finite_adapter_parameters": True, "heldout_examples_in_optimization": 0,
             "provenance": {"tokenizer_asset_sha256": {"a": "b"},
                            "resolved_snapshot_commit": v4.MODEL_REVISION}}
            for s in range(7)]
    assert not v4.evaluate(runs, range(8))["passes"]
