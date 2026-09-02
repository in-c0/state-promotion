from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_exp001r_sequential.py"
SPEC = importlib.util.spec_from_file_location("exp001r_sequential", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MOD)


def _row(diag: float, forgetting: float, anti_ceiling: bool = True) -> dict:
    return {
        "mean_diagonal_acquisition": diag,
        "average_forgetting": forgetting,
        "sequential_rules_out_all_arm_ceiling": anti_ceiling,
        "finite_losses": True,
        "finite_adapter_parameters": True,
        "backbone_frozen": True,
        "backbone_gradients_absent": True,
    }


def test_average_forgetting_uses_best_post_task_score_minus_final():
    matrix = [
        [0.8, float("nan"), float("nan")],
        [0.7, 0.9, float("nan")],
        [0.5, 0.8, 0.7],
    ]
    got = MOD.average_forgetting(matrix)
    expected = ((0.8 - 0.5) + (0.9 - 0.8) + (0.7 - 0.7)) / 3
    assert abs(got - expected) < 1e-12


def test_second_gate_passes_only_with_acquisition_and_interference():
    rows = [_row(0.8, 0.12), _row(0.7, 0.04), _row(0.75, 0.02)]
    summary = MOD.summarize(
        rows,
        rank=2,
        lr=1e-3,
        representation_summary={"protocol": "EXP-001R-representation-sufficiency-v1"},
    )
    assert summary["acquisition_pass"] is True
    assert summary["interference_pass"] is True
    assert summary["both_gates_pass"] is True
    assert summary["disposition"] == "resume_exp001_phase_b_multi_arm_development"


def test_acquisition_failure_returns_to_representation_gate():
    rows = [_row(0.5, 0.2), _row(0.6, 0.2), _row(0.55, 0.2)]
    summary = MOD.summarize(
        rows,
        rank=1,
        lr=3e-4,
        representation_summary={"protocol": "EXP-001R-representation-sufficiency-v1"},
    )
    assert summary["acquisition_pass"] is False
    assert summary["both_gates_pass"] is False
    assert summary["disposition"] == "return_to_representation_sufficiency"


def test_negligible_forgetting_allows_only_interference_calibration():
    rows = [_row(0.8, 0.02), _row(0.75, 0.01), _row(0.7, 0.03)]
    summary = MOD.summarize(
        rows,
        rank=4,
        lr=3e-3,
        representation_summary={"protocol": "EXP-001R-representation-sufficiency-v1"},
    )
    assert summary["acquisition_pass"] is True
    assert summary["interference_pass"] is False
    assert summary["both_gates_pass"] is False
    assert summary["disposition"] == "difficulty_or_interference_calibration_allowed"
