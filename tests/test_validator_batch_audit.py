import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_runs.py"


def run_manifest(method: str, *, online_hash: str = "online-same", eval_hash: str = "eval-same") -> dict:
    latent = method in {"fixed", "random", "promotion"}
    promotion_audit = [{"heldout_gate_example_count": 0}] if method == "promotion" else []
    return {
        "classification": "PILOT",
        "method": method,
        "stream": "retention",
        "seed": 20260901,
        "source_tree_sha256": "tree-same",
        "model": {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "snapshot_revision": "a" * 40,
            "model_revision": "a" * 40,
            "tokenizer_revision": "a" * 40,
            "backbone_frozen": True,
            "plastic_parameter_capacity": 200,
            "latent_state_enabled": latent,
        },
        "invalidation_reasons": [],
        "batch_audit": {
            "first_online_batch": {
                "all_source_splits_train": True,
                "examples": [{"audit_metadata": {"split": "train"}}],
                "model_visible_batch_sha256": online_hash,
            },
            "first_eval_query": {
                "audit_metadata": {
                    "split": "test",
                    "gold_target_is_model_privileged": False,
                },
                "model_visible_query_sha256": eval_hash,
            },
        },
        "promotion_probe_audit": promotion_audit,
        "budget": {
            "decision_tokens_processed": 1 if method == "promotion" else 0,
            "decision_forward_calls": 1 if method == "promotion" else 0,
            "examples_seen": 10,
            "write_budget_units": 1000,
            "parameter_write_units": 900,
            "tokens_processed": 100,
            "replay_capacity_examples": 32,
        },
        "accepted_commit_segments": [1] if method in {"promotion", "random"} else [],
    }


def invoke(tmp_path: Path, manifests: list[dict]) -> tuple[int, dict]:
    paths = []
    for idx, manifest in enumerate(manifests):
        path = tmp_path / f"run-{idx}.json"
        path.write_text(json.dumps(manifest))
        paths.append(path)
    proc = subprocess.run(
        [sys.executable, str(VALIDATOR), *map(str, paths)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return proc.returncode, json.loads(proc.stdout)


def test_validator_accepts_identical_first_model_inputs_across_all_six_pilot_arms(tmp_path):
    methods = ["frozen", "sequential", "replay", "fixed", "promotion", "random"]
    rc, payload = invoke(tmp_path, [run_manifest(method) for method in methods])
    assert rc == 0, payload
    assert payload["reasons"] == []


def test_validator_rejects_random_control_with_different_model_visible_first_batch(tmp_path):
    methods = ["frozen", "sequential", "replay", "fixed", "promotion"]
    manifests = [run_manifest(method) for method in methods]
    manifests.append(run_manifest("random", online_hash="different"))
    rc, payload = invoke(tmp_path, manifests)
    assert rc == 2
    assert "pilot_arms_receive_different_first_online_model_inputs" in payload["reasons"]


def test_validator_reports_missing_audit_instead_of_crashing(tmp_path):
    """A run that never archived an audit must yield a reason, not an exception.

    The runner serializes ``batch_audit`` sections as ``null`` until they are
    populated, so a degenerate or truncated arm reaches the validator with
    ``first_online_batch: null``. That is a diagnostic outcome the pilot must
    preserve, so the validator has to stay parseable.
    """
    methods = ["frozen", "sequential", "replay", "fixed", "promotion", "random"]
    manifests = [run_manifest(method) for method in methods]
    manifests[0]["batch_audit"] = {"first_online_batch": None, "first_eval_query": None}
    rc, payload = invoke(tmp_path, manifests)
    assert rc == 2
    assert "missing_online_batch_audit:frozen" in payload["reasons"]
    assert "missing_eval_query_audit:frozen" in payload["reasons"]
    assert "missing_pilot_arm_online_batch_hash" in payload["reasons"]
    assert "missing_pilot_arm_eval_query_hash" in payload["reasons"]
