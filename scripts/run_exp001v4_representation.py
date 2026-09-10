#!/usr/bin/env python3
"""EXP-001 v4 representation robustness: Panel A selection, Panel B validation.

Issue #8. The EXP-001R representation gate selected r=2 @ 3e-3 from three
development seeds on which it scored 1.000/1.000/1.000; on five fresh seeds the
same cell reached 0.800. That is the winner's curse: with nine cells and three
seeds, the best-looking cell is partly best-looking by luck.

v4's answer is not a better selection rule but an independent one. Panel A
selects from the same canonical grid on eight fresh seeds. Panel B then scores
*only the selected cell* on eight seeds selection never touched, against the same
unchanged criterion. If Panel B fails, v4 stops -- reselecting the runner-up
would reintroduce exactly the bias the panel exists to measure, so this script
has no code path that can do it.
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
    backbone_digest,
    git_sha,
    source_tree_sha256,
    unit_write_parameters,
)
from run_exp001v2_c1 import _score, dedupe_mappings

from state_promotion.lm import BudgetCounter, LMExperimentConfig, supervised_step
from state_promotion.lora import load_lora_model
from state_promotion.pals import generate_retention_stream, protocol_train_order
from state_promotion.seeds import V4_SELECTION_SEEDS, V4_VALIDATION_SEEDS

# Canonical grid, unchanged from EXP-001R.
RANKS = (1, 2, 4)
LRS = (3e-4, 1e-3, 3e-3)
EXPOSURES = 48

# EXP-001R acquisition criterion, unchanged.
MEAN_MIN = 5 / 6
WORST_MIN = 4 / 6


def cell_key(rank: int, lr: float) -> str:
    return f"r{rank}@lr{lr:.0e}"


def run_cell(*, seed: int, rank: int, lr: float, device: str, use_latent: bool = True) -> dict:
    cfg = LMExperimentConfig(model_revision=MODEL_REVISION, online_lr=lr)
    examples, candidates = generate_retention_stream(seed)
    train = [e for e in examples if e.segment == 0 and e.split == "train"]
    tests = dedupe_mappings([e for e in examples if e.segment == 0 and e.split == "test"])
    if len(train) != EXPOSURES:
        raise AssertionError(f"expected {EXPOSURES} segment-0 train exposures, got {len(train)}")
    if len(tests) != 6:
        raise AssertionError(f"expected 6 distinct segment-0 test mappings, got {len(tests)}")
    if any(e.split != "train" for e in train):
        raise AssertionError("held-out example entered the adaptation set")
    ordered = protocol_train_order({0: train}, seed)[0]

    model_init_seed = seed + 1701
    torch.manual_seed(model_init_seed)
    model, tokenizer, device_report = load_lora_model(
        cfg, rank=rank, adapter_mode="two_timescale", device=device, seed=model_init_seed
    )
    backbone_before = backbone_digest(model)
    P = unit_write_parameters(model, "two_timescale")

    budget = BudgetCounter()
    # Fast only: the slow adapter is present and active for architecture parity
    # but zero-initialised and never trainable, so it takes no writes.
    params = model.set_trainable("fast")
    opt = torch.optim.AdamW(params, lr=lr)
    losses = [
        supervised_step(
            model, tokenizer, [ex], opt, budget,
            use_fast=True, use_slow=True,
            use_latent=use_latent, update_latent=use_latent,
        )
        for ex in ordered
    ]
    accuracy, predictions = _score(
        model, tokenizer, tests, candidates,
        use_fast=True, use_slow=True, use_latent=use_latent,
    )
    backbone_after = backbone_digest(model)

    if backbone_before != backbone_after:
        raise AssertionError("frozen backbone changed during representation scoring")
    if budget.parameter_write_units != EXPOSURES * P:
        raise AssertionError(
            f"expected exactly {EXPOSURES * P} fast writes, got {budget.parameter_write_units}"
        )
    if model.plastic_parameter_count("slow") != P:
        raise AssertionError("slow adapter capacity diverged from fast")

    return {
        "protocol": "EXP-001v4-representation-v1",
        "status": "DEVELOPMENT_ONLY",
        "seed": seed,
        "rank": rank,
        "online_lr": lr,
        "use_latent": use_latent,
        "accuracy": accuracy,
        "predictions": predictions,
        "loss_trajectory": losses,
        "unit_write_parameters_P": P,
        "training_exposures": len(ordered),
        "latent_norm_after": float(model.latent.norm()),
        "median_token_embedding_norm": float(
            model.base.get_input_embeddings().weight.norm(dim=1).median()
        ),
        "finite_losses": all(torch.isfinite(torch.tensor(losses)).tolist()),
        "finite_adapter_parameters": all(
            torch.isfinite(p.detach()).all().item() for p in model.parameters()
        ),
        "backbone_frozen": True,
        "backbone_sha256_before": backbone_before,
        "backbone_sha256_after": backbone_after,
        "heldout_examples_in_optimization": 0,
        "budget": {
            "optimizer_steps": budget.optimizer_steps,
            "parameter_write_units": budget.parameter_write_units,
            "slow_write_units": 0,
            "training_examples_processed": budget.training_examples_processed,
            "tokens_processed": budget.tokens_processed,
        },
        "provenance": model.provenance,
        "device": device,
        "compute_dtype": str(next(model.base.parameters()).dtype),
        "device_numerics": device_report,
        "git_sha": git_sha(),
        "source_tree_sha256": source_tree_sha256(),
    }


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


def evaluate(runs: list[dict], expected_seeds) -> dict:
    """Apply the unchanged EXP-001R criterion to one cell's runs."""
    reasons = validity_reasons(runs)
    if len(runs) != len(expected_seeds):
        reasons.append("missing_seed")
    accs = [r["accuracy"] for r in runs]
    mean = sum(accs) / len(accs) if accs else float("nan")
    worst = min(accs) if accs else float("nan")
    return {
        "per_seed_accuracy": accs,
        "mean_accuracy": mean,
        "worst_seed_accuracy": worst,
        "valid": not reasons,
        "passes": (not reasons) and mean >= MEAN_MIN and worst >= WORST_MIN,
        "invalidation_reasons": reasons,
    }


def select(cells: list[dict]) -> dict | None:
    """Panel-A selection rule, unchanged in form from EXP-001R."""
    passing = [c for c in cells if c["passes"]]
    if not passing:
        return None
    smallest_rank = min(c["rank"] for c in passing)
    within = [c for c in passing if c["rank"] == smallest_rank]
    within.sort(key=lambda c: (-c["worst_seed_accuracy"], -c["mean_accuracy"], c["online_lr"]))
    return within[0]


def load_selected(path: Path) -> dict:
    summary = json.loads(path.read_text())
    if summary.get("protocol") != "EXP-001v4-panel-a-representation-selection-v1":
        raise ValueError("unexpected Panel-A summary protocol")
    if not summary.get("panel_b_allowed") or not summary.get("selected"):
        raise ValueError("Panel A selected no cell; Panel B is forbidden")
    return summary["selected"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", choices=("a", "b", "b-control"), required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--panel-a-summary", default="results/exp001v4-panel-a/summary.json")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    if args.device != "cpu":
        raise SystemExit("EXP-001 is CPU-only; see results/exp001b-device-probe/")

    out_dir = ROOT / (args.output_dir or f"results/exp001v4-panel-{args.panel}")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if args.panel == "a":
        runs = []
        for rank in RANKS:
            for lr in LRS:
                for seed in V4_SELECTION_SEEDS:
                    print(f"[A] r={rank} lr={lr:g} seed={seed}", flush=True)
                    t0 = time.time()
                    try:
                        r = run_cell(seed=seed, rank=rank, lr=lr, device=args.device)
                        r["cell_valid"] = True
                    except Exception as exc:  # noqa: BLE001 - failed runs archived, not dropped
                        r = {"seed": seed, "rank": rank, "online_lr": lr, "cell_valid": False,
                             "invalidation_reasons": [f"{type(exc).__name__}: {exc}"]}
                    (out_dir / f"seed-{seed}-r{rank}-lr{lr:.0e}.json").write_text(
                        json.dumps(r, indent=2, sort_keys=True) + "\n")
                    runs.append(r)
                    print(f"    acc={r.get('accuracy', float('nan')):.3f} "
                          f"{time.time() - t0:.0f}s", flush=True)

        cells = []
        for rank in RANKS:
            for lr in LRS:
                ok = [r for r in runs if r.get("cell_valid")
                      and r["rank"] == rank and r["online_lr"] == lr]
                cells.append({"cell": cell_key(rank, lr), "rank": rank, "online_lr": lr,
                              **evaluate(ok, V4_SELECTION_SEEDS)})
        selected = select(cells)
        summary = {
            "protocol": "EXP-001v4-panel-a-representation-selection-v1",
            "status": "DEVELOPMENT_ONLY",
            "note": "Selection panel. Not evidence of robustness on its own -- Panel B decides that.",
            "selection_seeds": list(V4_SELECTION_SEEDS),
            "ranks": list(RANKS), "learning_rates": list(LRS),
            "criterion": {"mean_min": MEAN_MIN, "per_seed_min": WORST_MIN,
                          "source": "EXP-001R acquisition criterion, unchanged"},
            "selection_rule": ("smallest rank; within rank highest worst-seed; then highest "
                               "mean; exact ties prefer lower LR"),
            "cells": cells, "selected": selected,
            "panel_b_allowed": selected is not None,
            "model_revision": MODEL_REVISION, "git_sha": git_sha(),
            "elapsed_seconds": time.time() - started,
        }
    else:
        sel = load_selected(ROOT / args.panel_a_summary)
        rank, lr = int(sel["rank"]), float(sel["online_lr"])
        use_latent = args.panel == "b"
        label = "validation" if use_latent else "latent-off diagnostic control"
        print(f"[Panel B {label}] frozen cell r={rank} lr={lr:g}", flush=True)
        runs = []
        for seed in V4_VALIDATION_SEEDS:
            print(f"[B] seed={seed} latent={use_latent}", flush=True)
            t0 = time.time()
            r = run_cell(seed=seed, rank=rank, lr=lr, device=args.device, use_latent=use_latent)
            (out_dir / f"seed-{seed}.json").write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
            runs.append(r)
            print(f"    acc={r['accuracy']:.3f} {time.time() - t0:.0f}s", flush=True)
        verdict = evaluate(runs, V4_VALIDATION_SEEDS)
        summary = {
            "protocol": ("EXP-001v4-panel-b-representation-validation-v1" if use_latent
                         else "EXP-001v4-panel-b-latent-off-control-v1"),
            "status": "DEVELOPMENT_ONLY",
            "note": ("Independent validation of the single Panel-A cell. No reselection is "
                     "permitted on failure." if use_latent else
                     "Diagnostic only. Must not alter selection or validation."),
            "validation_seeds": list(V4_VALIDATION_SEEDS),
            "frozen_cell": {"rank": rank, "online_lr": lr},
            "use_latent": use_latent,
            "criterion": {"mean_min": MEAN_MIN, "per_seed_min": WORST_MIN,
                          "source": "EXP-001R acquisition criterion, unchanged"},
            **verdict,
            "panel_c_allowed": bool(verdict["passes"]) if use_latent else False,
            "reselection_permitted": False,
            "model_revision": MODEL_REVISION, "git_sha": git_sha(),
            "elapsed_seconds": time.time() - started,
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("selected", "panel_b_allowed", "panel_c_allowed", "passes",
                               "mean_accuracy", "worst_seed_accuracy")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
