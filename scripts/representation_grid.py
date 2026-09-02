#!/usr/bin/env python3
"""Development capacity probe over plastic-state representations.

This is **not** a protocol run and not an arm comparison. It does not implement
fast/slow timescales, promotion, rollback, replay, or write budgets, and it
produces no evidence for or against H1. It asks one upstream question that the
retention engineering pilot left open:

    can a given plastic representation acquire a PALS segment's mappings at all?

The pilot found acquisition at chance for every arm, and a diagnostic traced
that to `PromptStateLM._prefix` building one input-independent prefix that is
expanded across the batch: it can shift the output distribution globally but
cannot express a key-conditional lookup. Two explanations survive that
observation, and they need different fixes:

  * capacity - 7168 plastic elements are simply too few, or
  * form - an input-independent prefix cannot represent the task at any size.

The grid separates them by crossing representation form (prompt prefix vs LoRA
adapters, which are input-conditional by construction) with capacity (a large
prompt is included precisely so that "LoRA has more parameters" cannot be the
whole explanation). Learning rate is swept per representation so no arm of the
grid is handicapped by a rate chosen for a different mechanism.

The full grid is always reported. Cells are never selected after the fact.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_promotion.lm import (  # noqa: E402
    LMExperimentConfig,
    PromptStateLM,
    encode_example,
)
from state_promotion.pals import Example, generate_retention_stream  # noqa: E402

# Declared development seeds. Confirmatory seeds must be disjoint from these.
DEVELOPMENT_SEEDS = [20260901, 20260902, 20260903]

# (name, kind, kwargs, learning rate). Swept per representation so that a rate
# tuned for prompt tuning does not stand in for a verdict about LoRA.
GRID = [
    ("prompt-8tok",    "prompt", {"tokens": 4},  5e-3),   # the preregistered configuration
    ("prompt-8tok",    "prompt", {"tokens": 4},  2e-2),
    ("prompt-128tok",  "prompt", {"tokens": 64}, 5e-3),   # capacity control, same form
    ("prompt-128tok",  "prompt", {"tokens": 64}, 2e-2),
    ("lora-r4",        "lora",   {"rank": 4},    1e-4),
    ("lora-r4",        "lora",   {"rank": 4},    5e-4),
    ("lora-r4",        "lora",   {"rank": 4},    2e-3),
    ("lora-r8",        "lora",   {"rank": 8},    5e-4),
    ("lora-r8",        "lora",   {"rank": 8},    2e-3),
]

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


class PromptRepresentation:
    """The preregistered plastic state: one input-independent prefix."""

    def __init__(self, base, cfg: LMExperimentConfig, tokens: int, seed: int, device: str):
        rep_cfg = replace(cfg, slow_tokens=tokens, fast_tokens=tokens)
        hidden = int(base.get_input_embeddings().embedding_dim)
        torch.manual_seed(seed)
        self.model = PromptStateLM(base, hidden, rep_cfg, seed=seed).to(device)
        self.params = self.model.set_trainable("single")

    def loss(self, ids, mask, labels):
        # Latent state is disabled so this measures representation capacity only.
        return self.model.forward_encoded(
            ids, mask, labels, use_slow=True, use_fast=True, use_latent=False, update_latent=False,
        ).loss

    def trainable_elements(self) -> int:
        return sum(p.numel() for p in self.params)

    def backbone_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.model.base.parameters())


class LoraRepresentation:
    """Input-conditional plastic state: low-rank deltas on attention projections."""

    def __init__(self, base, rank: int, seed: int, device: str):
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(seed)
        lora_cfg = LoraConfig(
            r=rank, lora_alpha=2 * rank, target_modules=LORA_TARGETS,
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, lora_cfg).to(device)
        self.params = [p for p in self.model.parameters() if p.requires_grad]

    def loss(self, ids, mask, labels):
        return self.model(input_ids=ids, attention_mask=mask, labels=labels).loss

    def trainable_elements(self) -> int:
        return sum(p.numel() for p in self.params)

    def backbone_frozen(self) -> bool:
        return all("lora" in name for name, p in self.model.named_parameters() if p.requires_grad)


def build(kind: str, base, cfg, device: str, seed: int, **kwargs):
    if kind == "prompt":
        return PromptRepresentation(base, cfg, kwargs["tokens"], seed, device)
    if kind == "lora":
        return LoraRepresentation(base, kwargs["rank"], seed, device)
    raise ValueError(f"unknown representation kind: {kind}")


def distinct_mappings(examples, segment: int, split: str) -> list[Example]:
    seen: dict[tuple[str, str], Example] = {}
    for ex in examples:
        if ex.segment == segment and ex.split == split:
            seen.setdefault((ex.context, ex.key), ex)
    return list(seen.values())


@torch.no_grad()
def score_mappings(rep, tokenizer, mappings, candidates, device):
    """Argmin completion NLL over candidates, exactly as the pilot scores."""
    correct, predictions = 0, []
    for ex in mappings:
        scores = []
        for cand in candidates:
            ids, mask, labels = encode_example(tokenizer, replace(ex, target=cand))
            scores.append(float(rep.loss(ids.to(device), mask.to(device), labels.to(device))))
        pred = candidates[min(range(len(scores)), key=scores.__getitem__)]
        predictions.append(pred)
        correct += int(pred == ex.target)
    return correct, predictions


def run_cell(name, kind, kwargs, lr, seed, tokenizer, base_loader, cfg, device, passes, shuffle=True):
    examples, candidates = generate_retention_stream(seed)
    train = [e for e in examples if e.segment == 0 and e.split == "train"]
    train = list(train)
    # The retention runner shuffles each segment's train events before the online
    # loop, so a faithful probe must too. PALS emits `train_repeats` consecutive
    # copies of each mapping, and training on that blocked order is maximally
    # interfering: it drives recency collapse onto the last mapping seen.
    if shuffle:
        __import__("random").Random(seed).shuffle(train)
    eval_mappings = distinct_mappings(examples, 0, "test")

    base = base_loader()
    rep = build(kind, base, cfg, device, seed, **kwargs)
    opt = torch.optim.AdamW(rep.params, lr=lr)

    started = time.perf_counter()
    trajectory = []
    correct, preds = score_mappings(rep, tokenizer, eval_mappings, candidates, device)
    trajectory.append({"pass": 0, "steps": 0, "correct": correct,
                       "distinct_predictions": len(set(preds))})
    for p in range(1, passes + 1):
        losses = []
        for ex in train:
            ids, mask, labels = encode_example(tokenizer, ex)
            opt.zero_grad(set_to_none=True)
            loss = rep.loss(ids.to(device), mask.to(device), labels.to(device))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        correct, preds = score_mappings(rep, tokenizer, eval_mappings, candidates, device)
        trajectory.append({
            "pass": p, "steps": p * len(train), "mean_loss": sum(losses) / len(losses),
            "correct": correct, "distinct_predictions": len(set(preds)),
            "predictions": preds,
        })
    best = max(t["correct"] for t in trajectory)
    final = trajectory[-1]["correct"]
    cell = {
        "representation": name, "kind": kind, "params": kwargs, "learning_rate": lr,
        "seed": seed, "trainable_elements": rep.trainable_elements(),
        "backbone_frozen": rep.backbone_frozen(),
        "n_mappings": len(eval_mappings), "final_correct": final, "best_correct": best,
        "trajectory": trajectory, "wall_seconds": round(time.perf_counter() - started, 1),
    }
    del rep, base, opt
    return cell


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--model-revision", default=None, help="Immutable HF snapshot SHA.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--passes", type=int, default=6)
    p.add_argument("--seeds", type=int, nargs="+", default=DEVELOPMENT_SEEDS)
    p.add_argument("--no-shuffle", action="store_true",
                   help="Train in PALS generation order (blocked repeats) instead of the runner's shuffled order.")
    p.add_argument("--only", default=None, help="Substring filter over representation names (smoke tests).")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "diagnostics" / "representation-grid.json")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = LMExperimentConfig(model_name=args.model, model_revision=args.model_revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def base_loader():
        base = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.model_revision, dtype=torch.float32,
        )
        base.to(args.device)
        base.eval()
        return base

    grid = [g for g in GRID if args.only is None or args.only in g[0]]
    cells = []
    total = len(grid) * len(args.seeds)
    print(f"representation grid: {len(grid)} configs x {len(args.seeds)} seeds = {total} cells, "
          f"{args.passes} passes each, device={args.device}", flush=True)
    for name, kind, kwargs, lr in grid:
        for seed in args.seeds:
            cell = run_cell(name, kind, kwargs, lr, seed, tokenizer, base_loader, cfg, args.device,
                            args.passes, shuffle=not args.no_shuffle)
            cells.append(cell)
            print(f"  {name:14s} lr={lr:<7g} seed={seed}  final={cell['final_correct']}/{cell['n_mappings']} "
                  f"best={cell['best_correct']}/{cell['n_mappings']}  "
                  f"plastic={cell['trainable_elements']:>9,}  {cell['wall_seconds']:>6.1f}s", flush=True)

    summary = {}
    for name, kind, kwargs, lr in grid:
        key = f"{name}@lr{lr:g}"
        sel = [c for c in cells if c["representation"] == name and c["learning_rate"] == lr]
        if not sel:
            continue
        n = len(sel)
        # Pass 1 is the protocol cadence: the retention arms see each segment's
        # train events once, one optimizer step each. Later passes are extra
        # budget the protocol does not grant, kept to separate "cannot at this
        # budget" from "cannot at all" and to expose late collapse.
        pass1 = [c["trajectory"][1]["correct"] for c in sel if len(c["trajectory"]) > 1]
        summary[key] = {
            "representation": name, "learning_rate": lr,
            "trainable_elements": sel[0]["trainable_elements"],
            "seeds": [c["seed"] for c in sel],
            "mean_correct_at_protocol_budget": (sum(pass1) / len(pass1)) if pass1 else float("nan"),
            "per_seed_at_protocol_budget": pass1,
            "mean_final_correct": sum(c["final_correct"] for c in sel) / n,
            "mean_best_correct": sum(c["best_correct"] for c in sel) / n,
            "per_seed_final": [c["final_correct"] for c in sel],
            "per_seed_best": [c["best_correct"] for c in sel],
            "meets_5of6_protocol_budget": bool(pass1) and sum(pass1) / len(pass1) >= 5.0,
            "meets_5of6_final": sum(c["final_correct"] for c in sel) / n >= 5.0,
            "meets_5of6_best": sum(c["best_correct"] for c in sel) / n >= 5.0,
        }

    payload = {
        "classification": "DEVELOPMENT_CAPACITY_PROBE",
        "note": "Not a protocol run, not an arm comparison, not evidence for or against H1.",
        "git_sha": git_sha(),
        "model": args.model, "model_revision": args.model_revision, "device": args.device,
        "development_seeds": args.seeds,
        "shuffled_presentation": not args.no_shuffle,
        "criterion": "mean mappings acquired across development seeds >= 5 of 6",
        "passes": args.passes,
        "summary": summary, "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print("\n=== summary (mean mappings acquired across development seeds, out of 6) ===")
    print(f"  {'config':22s} {'plastic':>10} {'pass1':>7} {'final':>7} {'best':>7} "
          f"{'per-seed pass1':>16}  >=5/6@protocol")
    for key, srow in summary.items():
        print(f"  {key:22s} {srow['trainable_elements']:>10,} "
              f"{srow['mean_correct_at_protocol_budget']:>7.2f} {srow['mean_final_correct']:>7.2f} "
              f"{srow['mean_best_correct']:>7.2f} {str(srow['per_seed_at_protocol_budget']):>16}  "
              f"{'YES' if srow['meets_5of6_protocol_budget'] else 'no'}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
