#!/usr/bin/env python3
"""EXP-001 v2 Gate C1: arm-agnostic slow-consolidation sufficiency.

Issue #6 section 2. Phase-B v1 localised its negative result to one mechanism:
48-step batch-1 slow consolidation reduced current-segment accuracy to chance,
so B5's rollback correctly declined every candidate. C1 asks whether the slow
adapter can absorb a freshly learned segment at all, using no routing logic and
no arm, before State Promotion is allowed to run again.

Parameter writes are held fixed at exactly 48P in every cell, so batch size buys
adaptation tokens and nothing else. The pass criterion is deliberately set above
B5's frozen `current_after >= 0.45` acceptance threshold: passing must mean slow
consolidation retains a meaningful majority of the six mappings, not that it is
barely gate-compatible.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp001b_arm import (
    MODEL_REVISION,
    ONLINE_LR,
    backbone_digest,
    git_sha,
    source_tree_sha256,
    unit_write_parameters,
)

from state_promotion.lm import (
    BudgetCounter,
    LMExperimentConfig,
    supervised_step,
)
from state_promotion.lora import consolidate_slow_lora, load_lora_model
from state_promotion.pals import generate_retention_stream, protocol_train_order
from state_promotion.seeds import V2_DEVELOPMENT_SEEDS

RANK = 2
SLOW_STEPS = 48            # fixed in every cell: slow write cost is 48P regardless of batch
BATCHES = (1, 2, 4)
SLOW_LRS = (1e-3, 2e-3, 3e-3)

PASS_MEAN_MIN = 4 / 6      # 0.6667, above B5's frozen 0.45 acceptance threshold
PASS_WORST_MIN = 3 / 6     # 0.5000


def cell_key(batch: int, lr: float) -> str:
    return f"b{batch}@lr{lr:.0e}"


def dedupe_mappings(items):
    seen, out = set(), []
    for ex in items:
        key = (ex.context, ex.key, ex.target)
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def run_cell(*, seed: int, batch: int, slow_lr: float, device: str) -> dict:
    cfg = LMExperimentConfig(model_revision=MODEL_REVISION, online_lr=ONLINE_LR)
    cfg.consolidation_lr = slow_lr
    cfg.consolidation_batch = batch

    examples, candidates = generate_retention_stream(seed)
    train = [e for e in examples if e.segment == 0 and e.split == "train"]
    tests = dedupe_mappings([e for e in examples if e.segment == 0 and e.split == "test"])
    if len(train) != 48:
        raise AssertionError(f"expected 48 segment-0 train exposures, got {len(train)}")
    if len(tests) != 6:
        raise AssertionError(f"expected 6 distinct segment-0 test mappings, got {len(tests)}")
    if any(e.split != "train" for e in train):
        raise AssertionError("held-out example entered the adaptation set")
    ordered = protocol_train_order({0: train}, seed)[0]

    model_init_seed = seed + 1701
    torch.manual_seed(model_init_seed)
    model, tokenizer, device_report = load_lora_model(
        cfg, rank=RANK, adapter_mode="two_timescale", device=device, seed=model_init_seed
    )
    backbone_before = backbone_digest(model)
    P = unit_write_parameters(model, "two_timescale")

    budget = BudgetCounter()
    budget.write_budget_units = 2 * SLOW_STEPS * P   # 48P fast + 48P slow, exactly

    # --- fast pass, exactly as Phase B does ---
    fast_params = model.set_trainable("fast")
    fast_opt = torch.optim.AdamW(fast_params, lr=ONLINE_LR)
    fast_losses = []
    for ex in ordered:
        fast_losses.append(
            supervised_step(
                model, tokenizer, [ex], fast_opt, budget,
                use_fast=True, use_slow=False, use_latent=False, update_latent=False,
            )
        )
    fast_writes = budget.parameter_write_units
    fast_only_accuracy, fast_predictions = _score(
        model, tokenizer, tests, candidates, use_fast=True, use_slow=False
    )

    # --- slow consolidation from already-observed segment-0 training evidence only ---
    consolidation_examples = list(ordered)
    if any(e.split != "train" for e in consolidation_examples):
        raise AssertionError("held-out example entered consolidation evidence")
    writes_before_slow = budget.parameter_write_units
    tokens_before_slow = budget.tokens_processed
    examples_before_slow = budget.training_examples_processed
    consolidate_slow_lora(
        model, tokenizer, consolidation_examples, budget, cfg,
        use_latent=False, steps=SLOW_STEPS,
    )
    slow_writes = budget.parameter_write_units - writes_before_slow

    # --- score slow-only, after consolidation, on held-out mappings ---
    slow_only_accuracy, slow_predictions = _score(
        model, tokenizer, tests, candidates, use_fast=False, use_slow=True
    )
    backbone_after = backbone_digest(model)

    if slow_writes != SLOW_STEPS * P:
        raise AssertionError(
            f"slow write cost {slow_writes} != {SLOW_STEPS * P}; batch must not change writes"
        )
    if fast_writes != SLOW_STEPS * P:
        raise AssertionError(f"fast write cost {fast_writes} != {SLOW_STEPS * P}")
    if backbone_before != backbone_after:
        raise AssertionError("frozen backbone changed during C1")

    return {
        "protocol": "EXP-001v2-C1-consolidation-sufficiency-v1",
        "status": "DEVELOPMENT_ONLY",
        "seed": seed,
        "consolidation_batch": batch,
        "slow_lr": slow_lr,
        "online_lr": ONLINE_LR,
        "rank": RANK,
        "slow_steps": SLOW_STEPS,
        "unit_write_parameters_P": P,
        "fast_only_accuracy": fast_only_accuracy,
        "slow_only_accuracy": slow_only_accuracy,
        "fast_predictions": fast_predictions,
        "slow_predictions": slow_predictions,
        "fast_loss_trajectory": fast_losses,
        "finite_fast_losses": all(torch.isfinite(torch.tensor(fast_losses)).tolist()),
        "finite_adapter_parameters": all(
            torch.isfinite(p.detach()).all().item() for p in model.parameters()
        ),
        "backbone_frozen": backbone_before == backbone_after,
        "backbone_sha256_before": backbone_before,
        "backbone_sha256_after": backbone_after,
        "heldout_examples_in_optimization": 0,
        "adaptation": {
            "unique_online_examples": len(ordered),
            "total_training_examples": budget.training_examples_processed,
            "adaptation_tokens": budget.tokens_processed,
            "optimizer_steps": budget.optimizer_steps,
            "parameter_write_units": budget.parameter_write_units,
            "write_budget_units": budget.write_budget_units,
            "fast_write_units": fast_writes,
            "slow_write_units": slow_writes,
            "consolidation_examples_processed": budget.training_examples_processed - examples_before_slow,
            "consolidation_tokens": budget.tokens_processed - tokens_before_slow,
            "replay_examples_used": budget.replay_examples_used,
            "consolidation_batch": batch,
        },
        "decision": {
            "decision_forward_calls": budget.decision_forward_calls,
            "decision_tokens_processed": budget.decision_tokens_processed,
            "candidate_consolidations_attempted": 0,
            "candidate_writes_rolled_back": 0,
            "accepted_promotions": 0,
        },
        "provenance": model.provenance,
        "device": device,
        "compute_dtype": str(next(model.base.parameters()).dtype),
        "device_numerics": device_report,
        "git_sha": git_sha(),
        "source_tree_sha256": source_tree_sha256(),
    }


def _score(model, tokenizer, tests, candidates, *, use_fast: bool, use_slow: bool,
           use_latent: bool = False):
    from state_promotion.lm import completion_nll

    rows, correct = [], 0
    for ex in tests:
        scores = {
            c: completion_nll(model, tokenizer, ex, c,
                              use_fast=use_fast, use_slow=use_slow, use_latent=use_latent)
            for c in candidates
        }
        pred = min(scores, key=scores.get)
        hit = pred == ex.target
        correct += hit
        rows.append({"key": ex.key, "gold": ex.target, "predicted": pred, "correct": bool(hit)})
    return correct / len(tests), rows


def select(cells: list[dict]) -> dict | None:
    """Predeclared C1 selection rule (issue #6 section 2)."""
    passing = [c for c in cells if c["passes"]]
    if not passing:
        return None
    smallest_batch = min(c["consolidation_batch"] for c in passing)
    within = [c for c in passing if c["consolidation_batch"] == smallest_batch]
    within.sort(key=lambda c: (-c["worst_seed_slow_only"], -c["mean_slow_only"], c["slow_lr"]))
    return within[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-dir", default="results/exp001v2-c1")
    args = ap.parse_args()
    if args.device != "cpu":
        raise SystemExit("EXP-001 is CPU-only; see results/exp001b-device-probe/")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runs: list[dict] = []

    for batch in BATCHES:
        for lr in SLOW_LRS:
            for seed in V2_DEVELOPMENT_SEEDS:
                print(f"[C1] batch={batch} slow_lr={lr:g} seed={seed}", flush=True)
                t0 = time.time()
                try:
                    r = run_cell(seed=seed, batch=batch, slow_lr=lr, device=args.device)
                    r["cell_valid"] = True
                    r["invalidation_reasons"] = []
                except Exception as exc:  # noqa: BLE001 - failed cells are archived, not dropped
                    r = {
                        "seed": seed, "consolidation_batch": batch, "slow_lr": lr,
                        "cell_valid": False,
                        "invalidation_reasons": [f"{type(exc).__name__}: {exc}"],
                    }
                (out_dir / f"seed-{seed}-b{batch}-lr{lr:.0e}.json").write_text(
                    json.dumps(r, indent=2, sort_keys=True) + "\n"
                )
                runs.append(r)
                print(
                    f"    fast_only={r.get('fast_only_accuracy', float('nan')):.3f} "
                    f"slow_only={r.get('slow_only_accuracy', float('nan')):.3f} "
                    f"{time.time() - t0:.0f}s", flush=True,
                )

    cells = []
    for batch in BATCHES:
        for lr in SLOW_LRS:
            rows = [r for r in runs if r.get("consolidation_batch") == batch and r.get("slow_lr") == lr]
            ok = [r for r in rows if r.get("cell_valid")]
            reasons = []
            if len(rows) != len(V2_DEVELOPMENT_SEEDS):
                reasons.append("missing_seed")
            if len(ok) != len(rows):
                reasons.append("run_failure")
            for r in ok:
                if not r["backbone_frozen"]:
                    reasons.append(f"backbone_not_frozen:{r['seed']}")
                if not (r["finite_fast_losses"] and r["finite_adapter_parameters"]):
                    reasons.append(f"non_finite:{r['seed']}")
                if r["heldout_examples_in_optimization"] != 0:
                    reasons.append(f"heldout_leak:{r['seed']}")
                if not (r["provenance"].get("tokenizer_asset_sha256") or {}):
                    reasons.append(f"tokenizer_provenance_unverified:{r['seed']}")
                if r["provenance"]["resolved_snapshot_commit"] != MODEL_REVISION:
                    reasons.append(f"snapshot_pin_mismatch:{r['seed']}")
                if r["adaptation"]["slow_write_units"] != SLOW_STEPS * r["unit_write_parameters_P"]:
                    reasons.append(f"slow_write_cost_wrong:{r['seed']}")
            accs = [r["slow_only_accuracy"] for r in ok]
            mean_slow = sum(accs) / len(accs) if accs else float("nan")
            worst_slow = min(accs) if accs else float("nan")
            fasts = [r["fast_only_accuracy"] for r in ok]
            passes = (
                not reasons
                and len(ok) == len(V2_DEVELOPMENT_SEEDS)
                and mean_slow >= PASS_MEAN_MIN
                and worst_slow >= PASS_WORST_MIN
            )
            cells.append({
                "cell": cell_key(batch, lr),
                "consolidation_batch": batch,
                "slow_lr": lr,
                "seeds": [r["seed"] for r in rows],
                "per_seed_slow_only": accs,
                "per_seed_fast_only": fasts,
                "mean_slow_only": mean_slow,
                "worst_seed_slow_only": worst_slow,
                "mean_fast_only": sum(fasts) / len(fasts) if fasts else float("nan"),
                "mean_consolidation_tokens": (
                    sum(r["adaptation"]["consolidation_tokens"] for r in ok) / len(ok) if ok else float("nan")
                ),
                "slow_write_units": ok[0]["adaptation"]["slow_write_units"] if ok else None,
                "valid": not reasons,
                "passes": passes,
                "invalidation_reasons": reasons,
            })

    selected = select(cells)
    summary = {
        "protocol": "EXP-001v2-C1-consolidation-sufficiency-v1",
        "status": "DEVELOPMENT_ONLY",
        "note": "Arm-agnostic. No routing logic, no arm comparison, not evidence for or against H1.",
        "development_seeds": list(V2_DEVELOPMENT_SEEDS),
        "batches": list(BATCHES),
        "slow_lrs": list(SLOW_LRS),
        "slow_steps_per_cell": SLOW_STEPS,
        "pass_rule": {
            "mean_slow_only_min": PASS_MEAN_MIN,
            "per_seed_slow_only_min": PASS_WORST_MIN,
            "note": "deliberately above B5's frozen current_after >= 0.45 threshold",
        },
        "selection_rule": (
            "smallest consolidation batch; within batch highest worst-seed slow-only "
            "accuracy; then highest mean; exact ties prefer lower slow LR"
        ),
        "cells": cells,
        "selected": selected,
        "c2_allowed": selected is not None,
        "model_revision": MODEL_REVISION,
        "git_sha": git_sha(),
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected": selected, "c2_allowed": summary["c2_allowed"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
