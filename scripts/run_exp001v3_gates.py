#!/usr/bin/env python3
"""EXP-001 v3 gates L0 and L1 under the repaired embedding-space latent.

Issue #7 sections 3-4. L0 asks whether the repaired latent channel still permits
the online plasticity already established by EXP-001R. L1 then asks, arm-
agnostically, whether slow consolidation can absorb a freshly learned segment
with the latent enabled exactly as B5 would have it -- the question issue #6
could not answer because its latent was ambiguous and defective.

Neither gate runs routing logic or compares arms.
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
from run_exp001v2_c1 import _score, dedupe_mappings

from state_promotion.lm import BudgetCounter, LMExperimentConfig, supervised_step
from state_promotion.lora import consolidate_slow_lora, load_lora_model
from state_promotion.pals import generate_retention_stream, protocol_train_order
from state_promotion.seeds import V3_DEVELOPMENT_SEEDS

RANK = 2
SLOW_STEPS = 48
BATCHES = (1, 2, 4)
SLOW_LRS = (1e-3, 2e-3, 3e-3)

# L0 reuses the EXP-001R sufficiency criterion unchanged.
L0_MEAN_MIN = 5 / 6
L0_WORST_MIN = 4 / 6
# L1, as in issue #6, sits deliberately above B5's frozen 0.45 acceptance rule.
L1_MEAN_MIN = 4 / 6
L1_WORST_MIN = 3 / 6


def _prepare(seed: int):
    examples, candidates = generate_retention_stream(seed)
    train = [e for e in examples if e.segment == 0 and e.split == "train"]
    tests = dedupe_mappings([e for e in examples if e.segment == 0 and e.split == "test"])
    if len(train) != 48:
        raise AssertionError(f"expected 48 segment-0 train exposures, got {len(train)}")
    if len(tests) != 6:
        raise AssertionError(f"expected 6 distinct segment-0 test mappings, got {len(tests)}")
    if any(e.split != "train" for e in train):
        raise AssertionError("held-out example entered the adaptation set")
    return protocol_train_order({0: train}, seed)[0], tests, candidates


def run_cell(*, seed: int, device: str, gate: str, batch: int = 1, slow_lr: float = 3e-3) -> dict:
    cfg = LMExperimentConfig(model_revision=MODEL_REVISION, online_lr=ONLINE_LR)
    cfg.consolidation_lr = slow_lr
    cfg.consolidation_batch = batch
    ordered, tests, candidates = _prepare(seed)

    model_init_seed = seed + 1701
    torch.manual_seed(model_init_seed)
    model, tokenizer, device_report = load_lora_model(
        cfg, rank=RANK, adapter_mode="two_timescale", device=device, seed=model_init_seed
    )
    backbone_before = backbone_digest(model)
    P = unit_write_parameters(model, "two_timescale")
    budget = BudgetCounter()
    budget.write_budget_units = 2 * SLOW_STEPS * P

    # --- fast pass: slow adapter present and active, repaired latent updated,
    #     exactly as B5 would have it. No replay, no consolidation, no routing.
    fast_params = model.set_trainable("fast")
    fast_opt = torch.optim.AdamW(fast_params, lr=ONLINE_LR)
    losses = [
        supervised_step(
            model, tokenizer, [ex], fast_opt, budget,
            use_fast=True, use_slow=True, use_latent=True, update_latent=True,
        )
        for ex in ordered
    ]
    fast_writes = budget.parameter_write_units
    latent_after_fast = model.latent.detach().clone()
    fast_accuracy, fast_predictions = _score(
        model, tokenizer, tests, candidates, use_fast=True, use_slow=True, use_latent=True
    )

    result = {
        "protocol": f"EXP-001v3-{gate.upper()}-v1",
        "status": "DEVELOPMENT_ONLY",
        "gate": gate,
        "seed": seed,
        "rank": RANK,
        "online_lr": ONLINE_LR,
        "unit_write_parameters_P": P,
        "fast_accuracy": fast_accuracy,
        "fast_predictions": fast_predictions,
        "fast_loss_trajectory": losses,
        "latent_norm_after_fast": float(latent_after_fast.norm()),
        "median_token_embedding_norm": float(
            model.base.get_input_embeddings().weight.norm(dim=1).median()
        ),
        "finite_losses": all(torch.isfinite(torch.tensor(losses)).tolist()),
        "heldout_examples_in_optimization": 0,
    }

    if gate == "l1":
        # Preserve the accumulated latent; consolidation must not advance it.
        consolidation_examples = list(ordered)
        if any(e.split != "train" for e in consolidation_examples):
            raise AssertionError("held-out example entered consolidation evidence")
        writes_before = budget.parameter_write_units
        tokens_before = budget.tokens_processed
        examples_before = budget.training_examples_processed
        consolidate_slow_lora(
            model, tokenizer, consolidation_examples, budget, cfg,
            use_latent=True, steps=SLOW_STEPS,
        )
        slow_writes = budget.parameter_write_units - writes_before
        if not torch.equal(model.latent, latent_after_fast):
            raise AssertionError("consolidation advanced the latent state")
        if slow_writes != SLOW_STEPS * P:
            raise AssertionError(
                f"slow write cost {slow_writes} != {SLOW_STEPS * P}; batch must not change writes"
            )
        slow_accuracy, slow_predictions = _score(
            model, tokenizer, tests, candidates, use_fast=False, use_slow=True, use_latent=True
        )
        result.update({
            "consolidation_batch": batch,
            "slow_lr": slow_lr,
            "slow_steps": SLOW_STEPS,
            "slow_only_accuracy": slow_accuracy,
            "slow_predictions": slow_predictions,
            "latent_unchanged_by_consolidation": True,
            "consolidation_examples_processed": budget.training_examples_processed - examples_before,
            "consolidation_tokens": budget.tokens_processed - tokens_before,
            "slow_write_units": slow_writes,
        })

    backbone_after = backbone_digest(model)
    if backbone_before != backbone_after:
        raise AssertionError("frozen backbone changed during the gate")
    result.update({
        "finite_adapter_parameters": all(
            torch.isfinite(p.detach()).all().item() for p in model.parameters()
        ),
        "backbone_frozen": True,
        "backbone_sha256_before": backbone_before,
        "backbone_sha256_after": backbone_after,
        "adaptation": {
            "unique_online_examples": len(ordered),
            "total_training_examples": budget.training_examples_processed,
            "adaptation_tokens": budget.tokens_processed,
            "optimizer_steps": budget.optimizer_steps,
            "parameter_write_units": budget.parameter_write_units,
            "write_budget_units": budget.write_budget_units,
            "fast_write_units": fast_writes,
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
    })
    return result


def validity_reasons(runs: list[dict]) -> list[str]:
    reasons = []
    for r in runs:
        s = r["seed"]
        if not r.get("backbone_frozen"):
            reasons.append(f"backbone_not_frozen:{s}")
        if not (r.get("finite_losses") and r.get("finite_adapter_parameters")):
            reasons.append(f"non_finite:{s}")
        if r.get("heldout_examples_in_optimization") != 0:
            reasons.append(f"heldout_leak:{s}")
        if not (r["provenance"].get("tokenizer_asset_sha256") or {}):
            reasons.append(f"tokenizer_provenance_unverified:{s}")
        if r["provenance"]["resolved_snapshot_commit"] != MODEL_REVISION:
            reasons.append(f"snapshot_pin_mismatch:{s}")
    return reasons


def select_l1(cells: list[dict]) -> dict | None:
    passing = [c for c in cells if c["passes"]]
    if not passing:
        return None
    smallest = min(c["consolidation_batch"] for c in passing)
    within = [c for c in passing if c["consolidation_batch"] == smallest]
    within.sort(key=lambda c: (-c["worst_seed_slow_only"], -c["mean_slow_only"], c["slow_lr"]))
    return within[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", choices=("l0", "l1"), required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    if args.device != "cpu":
        raise SystemExit("EXP-001 is CPU-only; see results/exp001b-device-probe/")

    out_dir = ROOT / (args.output_dir or f"results/exp001v3-{args.gate}")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if args.gate == "l0":
        runs = []
        for seed in V3_DEVELOPMENT_SEEDS:
            print(f"[L0] seed={seed}", flush=True)
            t0 = time.time()
            r = run_cell(seed=seed, device=args.device, gate="l0")
            (out_dir / f"seed-{seed}.json").write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
            runs.append(r)
            print(f"    fast={r['fast_accuracy']:.3f} latent_norm={r['latent_norm_after_fast']:.3f} "
                  f"({r['latent_norm_after_fast'] / r['median_token_embedding_norm']:.2f}x median emb) "
                  f"{time.time() - t0:.0f}s", flush=True)
        accs = [r["fast_accuracy"] for r in runs]
        reasons = validity_reasons(runs)
        mean, worst = sum(accs) / len(accs), min(accs)
        passes = not reasons and mean >= L0_MEAN_MIN and worst >= L0_WORST_MIN
        summary = {
            "protocol": "EXP-001v3-L0-latent-enabled-acquisition-v1",
            "status": "DEVELOPMENT_ONLY",
            "note": "No routing, no consolidation, no arm comparison.",
            "development_seeds": list(V3_DEVELOPMENT_SEEDS),
            "criterion": {"mean_min": L0_MEAN_MIN, "per_seed_min": L0_WORST_MIN,
                          "source": "EXP-001R sufficiency criterion, unchanged"},
            "per_seed_accuracy": accs,
            "mean_accuracy": mean,
            "worst_seed_accuracy": worst,
            "latent_norm_after_fast": [r["latent_norm_after_fast"] for r in runs],
            "latent_norm_ratio": [
                r["latent_norm_after_fast"] / r["median_token_embedding_norm"] for r in runs
            ],
            "invalidation_reasons": reasons,
            "passes": passes,
            "l1_allowed": passes,
            "model_revision": MODEL_REVISION,
            "git_sha": git_sha(),
            "elapsed_seconds": time.time() - started,
        }
    else:
        runs = []
        for batch in BATCHES:
            for lr in SLOW_LRS:
                for seed in V3_DEVELOPMENT_SEEDS:
                    print(f"[L1] batch={batch} slow_lr={lr:g} seed={seed}", flush=True)
                    t0 = time.time()
                    try:
                        r = run_cell(seed=seed, device=args.device, gate="l1",
                                     batch=batch, slow_lr=lr)
                        r["cell_valid"] = True
                        r["invalidation_reasons"] = []
                    except Exception as exc:  # noqa: BLE001 - failed cells archived, not dropped
                        r = {"seed": seed, "consolidation_batch": batch, "slow_lr": lr,
                             "cell_valid": False,
                             "invalidation_reasons": [f"{type(exc).__name__}: {exc}"]}
                    (out_dir / f"seed-{seed}-b{batch}-lr{lr:.0e}.json").write_text(
                        json.dumps(r, indent=2, sort_keys=True) + "\n")
                    runs.append(r)
                    print(f"    fast={r.get('fast_accuracy', float('nan')):.3f} "
                          f"slow_only={r.get('slow_only_accuracy', float('nan')):.3f} "
                          f"{time.time() - t0:.0f}s", flush=True)
        cells = []
        for batch in BATCHES:
            for lr in SLOW_LRS:
                rows = [r for r in runs if r.get("consolidation_batch") == batch and r.get("slow_lr") == lr]
                ok = [r for r in rows if r.get("cell_valid")]
                reasons = validity_reasons(ok)
                if len(ok) != len(V3_DEVELOPMENT_SEEDS):
                    reasons.append("run_failure_or_missing_seed")
                accs = [r["slow_only_accuracy"] for r in ok]
                mean = sum(accs) / len(accs) if accs else float("nan")
                worst = min(accs) if accs else float("nan")
                cells.append({
                    "cell": f"b{batch}@lr{lr:.0e}",
                    "consolidation_batch": batch,
                    "slow_lr": lr,
                    "per_seed_slow_only": accs,
                    "per_seed_fast": [r["fast_accuracy"] for r in ok],
                    "mean_slow_only": mean,
                    "worst_seed_slow_only": worst,
                    "mean_fast": sum(r["fast_accuracy"] for r in ok) / len(ok) if ok else float("nan"),
                    "mean_consolidation_tokens": (
                        sum(r["consolidation_tokens"] for r in ok) / len(ok) if ok else float("nan")
                    ),
                    "valid": not reasons,
                    "passes": (not reasons) and mean >= L1_MEAN_MIN and worst >= L1_WORST_MIN,
                    "invalidation_reasons": reasons,
                })
        selected = select_l1(cells)
        summary = {
            "protocol": "EXP-001v3-L1-consolidation-sufficiency-v1",
            "status": "DEVELOPMENT_ONLY",
            "note": "Arm-agnostic. No routing logic, no arm comparison.",
            "development_seeds": list(V3_DEVELOPMENT_SEEDS),
            "batches": list(BATCHES),
            "slow_lrs": list(SLOW_LRS),
            "slow_steps_per_cell": SLOW_STEPS,
            "pass_rule": {"mean_min": L1_MEAN_MIN, "per_seed_min": L1_WORST_MIN,
                          "note": "above B5's frozen current_after >= 0.45 threshold"},
            "selection_rule": ("smallest batch; then highest worst-seed; then highest mean; "
                               "exact ties prefer lower slow LR"),
            "cells": cells,
            "selected": selected,
            "l2_allowed": selected is not None,
            "model_revision": MODEL_REVISION,
            "git_sha": git_sha(),
            "elapsed_seconds": time.time() - started,
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k in
                      ("passes", "l1_allowed", "l2_allowed", "mean_accuracy",
                       "worst_seed_accuracy", "selected")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
