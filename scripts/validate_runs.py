#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRIMARY = {"sequential", "replay", "fixed", "promotion"}


def rel_spread(values: list[float]) -> float:
    if not values:
        return 0.0
    m = max(values)
    return 0.0 if m == 0 else (m - min(values)) / m


def main() -> None:
    p = argparse.ArgumentParser(description="Validate EXP-001 run manifests before confirmatory labeling.")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--tolerance", type=float, default=0.02)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    runs = [json.loads(x.read_text()) for x in args.runs]
    reasons: list[str] = []
    methods = {r["method"] for r in runs}
    comparable = [r for r in runs if r["method"] in PRIMARY]

    if not PRIMARY.issubset(methods):
        reasons.append(f"missing_primary_methods:{sorted(PRIMARY - methods)}")
    if any(r.get("classification") != "PILOT" for r in runs):
        reasons.append("input_run_already_claims_nonpilot_status")
    if len({r.get("seed") for r in comparable}) > 1:
        reasons.append("primary_methods_not_paired_on_same_seed")
    if len({r.get("stream") for r in comparable}) > 1:
        reasons.append("primary_methods_not_on_same_stream")
    if len({r.get("model", {}).get("name") for r in comparable}) > 1:
        reasons.append("primary_methods_use_different_models")
    if any(not r.get("model", {}).get("backbone_frozen", False) for r in comparable):
        reasons.append("backbone_not_frozen")
    if any(r.get("invalidation_reasons") for r in comparable):
        reasons.append("run_level_invalidation_present")

    vals = [float(r["model"]["plastic_parameter_capacity"]) for r in comparable]
    if rel_spread(vals) > args.tolerance:
        reasons.append("parameter_capacity_mismatch:plastic_parameter_capacity")

    for key in ["examples_seen", "optimizer_steps", "parameter_write_units"]:
        vals = [float(r["budget"][key]) for r in comparable]
        if rel_spread(vals) > args.tolerance:
            reasons.append(f"budget_mismatch:{key}")

    replay_methods = [r for r in comparable if r["method"] in {"replay", "fixed", "promotion"}]
    if len({r["budget"].get("replay_capacity_examples") for r in replay_methods}) > 1:
        reasons.append("replay_capacity_mismatch")

    promotion = next((r for r in runs if r["method"] == "promotion"), None)
    random = next((r for r in runs if r["method"] == "random"), None)
    if promotion and random:
        if len(promotion.get("accepted_commit_segments", [])) != len(random.get("accepted_commit_segments", [])):
            reasons.append("random_control_commit_count_not_matched")

    payload = {
        "valid_for_confirmatory_interpretation": not reasons,
        "tolerance": args.tolerance,
        "methods": sorted(methods),
        "reasons": reasons,
        "note": "This validates machine-checkable budget/control conditions only; preregistered benchmark validity and statistics still apply.",
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)
    raise SystemExit(0 if not reasons else 2)


if __name__ == "__main__":
    main()
