#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_promotion.lm import BudgetCounter, LMExperimentConfig, completion_nll, supervised_step  # noqa: E402
from state_promotion.lora import load_lora_model  # noqa: E402
from state_promotion.pals import Example, generate_retention_stream, protocol_train_order  # noqa: E402


DEV_SEEDS = (20260901, 20260902, 20260903)
RANKS = (1, 2, 4)
LRS = (3e-4, 1e-3, 3e-3)
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def source_tree_sha256() -> str:
    h = hashlib.sha256()
    roots = [ROOT / "src", ROOT / "scripts", ROOT / "tests", ROOT / "experiments", ROOT / "docs"]
    files = [ROOT / "README.md", ROOT / "pyproject.toml", ROOT / "Makefile"]
    for root in roots:
        if root.exists():
            files.extend(x for x in root.rglob("*") if x.is_file())
    for path in sorted(set(files), key=lambda x: x.relative_to(ROOT).as_posix()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        rel = path.relative_to(ROOT).as_posix()
        h.update(rel.encode("utf-8")); h.update(b"\0")
        h.update(path.read_bytes()); h.update(b"\0")
    return h.hexdigest()


def dedupe_mappings(items: list[Example]) -> list[Example]:
    seen: set[tuple[str, str, str]] = set()
    out: list[Example] = []
    for ex in items:
        key = (ex.context, ex.key, ex.target)
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def score_mappings(model, tokenizer, tests: list[Example], candidates: list[str]) -> tuple[float, list[dict]]:
    rows = []
    correct = 0
    for ex in tests:
        scores = {
            candidate: completion_nll(
                model,
                tokenizer,
                ex,
                candidate,
                use_fast=True,
                use_slow=False,
                use_latent=False,
            )
            for candidate in candidates
        }
        pred = min(scores, key=scores.get)
        ok = pred == ex.target
        correct += int(ok)
        rows.append({
            "context": ex.context,
            "key": ex.key,
            "gold": ex.target,
            "pred": pred,
            "correct": ok,
            "candidate_nll": scores,
        })
    return correct / max(len(tests), 1), rows


def backbone_is_frozen(model) -> bool:
    return all(
        not p.requires_grad
        for name, p in model.named_parameters()
        if ".adapters." not in name
    )


def no_backbone_grads(model) -> bool:
    return all(
        p.grad is None
        for name, p in model.named_parameters()
        if ".adapters." not in name
    )


def run_cell(*, seed: int, rank: int, lr: float, device: str, output_dir: Path) -> dict:
    cfg = LMExperimentConfig(
        model_revision=MODEL_REVISION,
        online_lr=lr,
    )
    model_init_seed = 1701 + seed
    model, tokenizer, device_report = load_lora_model(
        cfg,
        rank=rank,
        adapter_mode="two_timescale",
        device=device,
        seed=model_init_seed,
    )
    if not backbone_is_frozen(model):
        raise AssertionError("non-LoRA backbone parameter is trainable before scoring")

    examples, candidates = generate_retention_stream(seed)
    train = [ex for ex in examples if ex.segment == 0 and ex.split == "train"]
    tests = dedupe_mappings([ex for ex in examples if ex.segment == 0 and ex.split == "test"])
    if len(train) != 48:
        raise AssertionError(f"EXP-001R requires exactly 48 segment-0 train exposures, got {len(train)}")
    if len(tests) != 6:
        raise AssertionError(f"EXP-001R requires 6 distinct segment-0 test mappings, got {len(tests)}")
    if any(ex.split != "train" for ex in train):
        raise AssertionError("held-out example entered adaptation set")

    # Amendment G: present segment-0 train events in the protocol runner's order.
    # PALS generation order is blocked (train_repeats consecutive copies per
    # mapping); run_lm_pals.py shuffles a copy of each segment with one
    # random.Random(seed) per run. Segment 0 is that RNG's first draw.
    train = protocol_train_order({0: train}, seed)[0]

    params = model.set_trainable("fast")
    fast_params = sum(p.numel() for p in params)
    slow_params = model.plastic_parameter_count("slow")
    if fast_params != slow_params:
        raise AssertionError(f"fast/slow capacity mismatch: {fast_params} != {slow_params}")

    optimizer = torch.optim.AdamW(params, lr=lr)
    budget = BudgetCounter()
    losses: list[float] = []
    started = time.time()

    for ex in train:
        losses.append(
            supervised_step(
                model,
                tokenizer,
                [ex],
                optimizer,
                budget,
                use_fast=True,
                use_slow=False,
                use_latent=False,
                update_latent=False,
            )
        )

    if budget.training_examples_processed != 48:
        raise AssertionError(
            f"expected 48 processed examples, got {budget.training_examples_processed}"
        )
    if budget.optimizer_steps != 48:
        raise AssertionError(f"expected 48 optimizer steps, got {budget.optimizer_steps}")
    if budget.parameter_write_units != 48 * fast_params:
        raise AssertionError(
            "write accounting mismatch: "
            f"{budget.parameter_write_units} != 48*{fast_params}"
        )
    if not no_backbone_grads(model):
        raise AssertionError("frozen backbone accumulated gradients")

    accuracy, predictions = score_mappings(model, tokenizer, tests, candidates)
    finite = all(torch.isfinite(torch.tensor(losses)).tolist())
    adapter_finite = all(torch.isfinite(p.detach()).all().item() for p in params)

    result = {
        "protocol": "EXP-001R-representation-sufficiency-v1",
        "status": "DEVELOPMENT_ONLY",
        "seed": seed,
        "rank": rank,
        "lr": lr,
        "target_modules": list(model.target_modules),
        "alpha_over_rank": 1.0,
        "fast_parameter_count": fast_params,
        "slow_parameter_count": slow_params,
        "total_two_timescale_parameter_count": fast_params + slow_params,
        "training_exposures": len(train),
        "test_distinct_mappings": len(tests),
        "accuracy": accuracy,
        "loss_trajectory": losses,
        "predictions": predictions,
        "finite_losses": finite,
        "finite_adapter_parameters": adapter_finite,
        "budget": {
            "optimizer_steps": budget.optimizer_steps,
            "parameter_write_units": budget.parameter_write_units,
            "training_examples_processed": budget.training_examples_processed,
            "tokens_processed": budget.tokens_processed,
        },
        "backbone_frozen": backbone_is_frozen(model),
        "backbone_gradients_absent": no_backbone_grads(model),
        "model_name": cfg.model_name,
        "requested_model_revision": cfg.model_revision,
        "loaded_model_revision": getattr(model.base.config, "_commit_hash", None),
        "loaded_tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "model_init_seed": model_init_seed,
        "device": device,
        "device_numerics": device_report,
        "git_sha": git_sha(),
        "source_tree_sha256": source_tree_sha256(),
        "elapsed_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"seed-{seed}-r{rank}-lr{lr:.0e}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def summarize(results: list[dict]) -> dict:
    cells = []
    for rank in RANKS:
        for lr in LRS:
            rows = [r for r in results if r["rank"] == rank and r["lr"] == lr]
            accuracies = [float(r["accuracy"]) for r in rows]
            mean = sum(accuracies) / len(accuracies)
            worst = min(accuracies)
            passes = (
                len(rows) == len(DEV_SEEDS)
                and mean >= 5 / 6
                and worst >= 4 / 6
                and all(r["finite_losses"] and r["finite_adapter_parameters"] for r in rows)
            )
            cells.append({
                "rank": rank,
                "lr": lr,
                "accuracies": accuracies,
                "mean_accuracy": mean,
                "worst_seed_accuracy": worst,
                "passes": passes,
            })

    passing = [c for c in cells if c["passes"]]
    selected = None
    if passing:
        smallest_rank = min(c["rank"] for c in passing)
        within_rank = [c for c in passing if c["rank"] == smallest_rank]
        within_rank.sort(key=lambda c: (-c["worst_seed_accuracy"], c["lr"]))
        selected = within_rank[0]

    return {
        "protocol": "EXP-001R-representation-sufficiency-v1",
        "status": "DEVELOPMENT_ONLY",
        "pass_rule": {
            "mean_accuracy_min": 5 / 6,
            "per_seed_accuracy_min": 4 / 6,
            "finite_required": True,
        },
        "selection_rule": (
            "smallest passing rank; within rank highest worst-seed acquisition; "
            "exact ties prefer smaller LR"
        ),
        "cells": cells,
        "selected": selected,
        "next_gate_allowed": selected is not None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-dir", default="results/exp001r-representation")
    args = ap.parse_args()

    output_dir = ROOT / args.output_dir
    results = []
    for rank in RANKS:
        for lr in LRS:
            for seed in DEV_SEEDS:
                print(f"[EXP-001R] seed={seed} rank={rank} lr={lr:g}", flush=True)
                results.append(
                    run_cell(
                        seed=seed,
                        rank=rank,
                        lr=lr,
                        device=args.device,
                        output_dir=output_dir,
                    )
                )

    summary = summarize(results)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
