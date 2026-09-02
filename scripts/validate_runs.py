#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRIMARY = {"sequential", "replay", "fixed", "promotion"}
INPUT_AUDIT_METHODS = PRIMARY | {"frozen", "random"}


def audit_hash(run: dict, section: str, field: str) -> str | None:
    """Read one model-visible input hash without assuming the audit was archived.

    A run that never reached an online batch or an evaluation query serializes the
    section as ``null``. That is a diagnostic the validator must report as a missing
    audit, not an exception that suppresses every other reason.
    """
    audit = run.get("batch_audit") or {}
    section_payload = audit.get(section) or {}
    return section_payload.get(field)


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
    input_comparable = [r for r in runs if r["method"] in INPUT_AUDIT_METHODS]

    if not PRIMARY.issubset(methods):
        reasons.append(f"missing_primary_methods:{sorted(PRIMARY - methods)}")
    if any(r.get("classification") != "PILOT" for r in runs):
        reasons.append("input_run_already_claims_nonpilot_status")
    if len({r.get("seed") for r in comparable}) > 1:
        reasons.append("primary_methods_not_paired_on_same_seed")
    if len({r.get("stream") for r in comparable}) > 1:
        reasons.append("primary_methods_not_on_same_stream")
    tree_hashes = {r.get("source_tree_sha256") for r in comparable}
    if None in tree_hashes or "" in tree_hashes:
        reasons.append("missing_source_tree_hash")
    elif len(tree_hashes) > 1:
        reasons.append("primary_methods_use_different_source_trees")
    if len({r.get("model", {}).get("name") for r in comparable}) > 1:
        reasons.append("primary_methods_use_different_models")
    snapshot_revisions = {r.get("model", {}).get("snapshot_revision") for r in input_comparable}
    if None in snapshot_revisions or "" in snapshot_revisions:
        reasons.append("pilot_model_snapshot_not_pinned")
    elif len(snapshot_revisions) > 1:
        reasons.append("pilot_arms_use_different_model_snapshots")
    elif snapshot_revisions:
        pinned = next(iter(snapshot_revisions))
        for r in input_comparable:
            reported_model = r.get("model", {}).get("model_revision")
            reported_tokenizer = r.get("model", {}).get("tokenizer_revision")
            if reported_model and reported_model != pinned:
                reasons.append(f"reported_model_revision_differs_from_pin:{r.get('method')}")
            if reported_tokenizer and reported_tokenizer != pinned:
                reasons.append(f"reported_tokenizer_revision_differs_from_pin:{r.get('method')}")
    if any(not r.get("model", {}).get("backbone_frozen", False) for r in comparable):
        reasons.append("backbone_not_frozen")
    if any(r.get("invalidation_reasons") for r in comparable):
        reasons.append("run_level_invalidation_present")

    # The strongest two-timescale controls must differ from B5 in routing, not
    # in whether a persistent latent state exists.
    matched_latent_methods = [r for r in runs if r.get("method") in {"fixed", "random", "promotion"}]
    if matched_latent_methods and any(not r.get("model", {}).get("latent_state_enabled", False) for r in matched_latent_methods):
        reasons.append("two_timescale_latent_architecture_mismatch")

    # Dynamic leakage audit: every run archives one actual tokenized online batch
    # and one actual multiple-choice evaluation query. Audit-only metadata is kept
    # structurally separate from model-visible tensors.
    for r in runs:
        method = r.get("method", "unknown")
        audit = r.get("batch_audit", {})
        online = audit.get("first_online_batch")
        eval_query = audit.get("first_eval_query")
        if not online:
            reasons.append(f"missing_online_batch_audit:{method}")
        else:
            if not online.get("all_source_splits_train", False):
                reasons.append(f"nontrain_example_in_online_batch_audit:{method}")
            if any(x.get("audit_metadata", {}).get("split") != "train" for x in online.get("examples", [])):
                reasons.append(f"online_batch_audit_split_mismatch:{method}")
        if not eval_query:
            reasons.append(f"missing_eval_query_audit:{method}")
        else:
            meta = eval_query.get("audit_metadata", {})
            if meta.get("split") != "test":
                reasons.append(f"eval_query_audit_split_mismatch:{method}")
            if meta.get("gold_target_is_model_privileged") is not False:
                reasons.append(f"eval_gold_target_privileged:{method}")

    online_hashes = {audit_hash(r, "first_online_batch", "model_visible_batch_sha256") for r in input_comparable}
    if None in online_hashes or "" in online_hashes:
        reasons.append("missing_pilot_arm_online_batch_hash")
    elif len(online_hashes) > 1:
        reasons.append("pilot_arms_receive_different_first_online_model_inputs")

    eval_hashes = {audit_hash(r, "first_eval_query", "model_visible_query_sha256") for r in input_comparable}
    if None in eval_hashes or "" in eval_hashes:
        reasons.append("missing_pilot_arm_eval_query_hash")
    elif len(eval_hashes) > 1:
        reasons.append("pilot_arms_receive_different_first_eval_model_inputs")

    promotion_runs = [r for r in runs if r.get("method", "").startswith("promotion")]
    for r in promotion_runs:
        audit = r.get("promotion_probe_audit", [])
        if r.get("method") != "promotion-no-slow" and not audit:
            reasons.append(f"missing_promotion_probe_audit:{r.get('method')}")
        leaked = sum(int(x.get("heldout_gate_example_count", 0)) for x in audit)
        if leaked:
            reasons.append(f"heldout_gate_leak:{r.get('method')}:{leaked}")
        budget = r.get("budget", {})
        if "decision_tokens_processed" not in budget or "decision_forward_calls" not in budget:
            reasons.append(f"missing_decision_compute_accounting:{r.get('method')}")

    for key in ["plastic_parameter_capacity"]:
        vals = [float(r["model"][key]) for r in comparable]
        if rel_spread(vals) > args.tolerance:
            reasons.append(f"parameter_capacity_mismatch:{key}")

    vals = [float(r["budget"]["examples_seen"]) for r in comparable]
    if rel_spread(vals) > args.tolerance:
        reasons.append("budget_mismatch:examples_seen")

    # Pre-result Amendment A: compare a common hard write ceiling rather than
    # forcing identical optimizer-step counts across differently sized write scopes.
    caps = [float(r["budget"]["write_budget_units"]) for r in comparable]
    if rel_spread(caps) > args.tolerance:
        reasons.append("budget_mismatch:write_budget_units")
    for r in comparable:
        if float(r["budget"]["parameter_write_units"]) > float(r["budget"]["write_budget_units"]) * (1 + args.tolerance):
            reasons.append(f"write_budget_exceeded:{r['method']}")

    replay_run = next((r for r in comparable if r["method"] == "replay"), None)
    fixed_run = next((r for r in comparable if r["method"] == "fixed"), None)
    promotion_run = next((r for r in comparable if r["method"] == "promotion"), None)
    if replay_run and fixed_run:
        compute_vals = [float(replay_run["budget"]["tokens_processed"]), float(fixed_run["budget"]["tokens_processed"])]
        if rel_spread(compute_vals) > 0.10:
            reasons.append("adaptation_token_envelope_mismatch:replay_vs_fixed")
    if promotion_run and replay_run and fixed_run:
        envelope = max(float(replay_run["budget"]["tokens_processed"]), float(fixed_run["budget"]["tokens_processed"]))
        if float(promotion_run["budget"]["tokens_processed"]) > envelope * 1.10:
            reasons.append("promotion_exceeds_adaptation_token_envelope")

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
