#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_promotion.lm import (  # noqa: E402
    BudgetCounter,
    LMExperimentConfig,
    ReplayStore,
    consolidate_slow,
    guarded_promotion,
    load_model,
    multiple_choice_accuracy,
    supervised_step,
)
from state_promotion.metrics import summarize  # noqa: E402
from state_promotion.pals import Example, generate_retention_stream  # noqa: E402


def group(examples):
    d = defaultdict(lambda: {"train": [], "test": []})
    for ex in examples:
        d[ex.segment][ex.split].append(ex)
    return d


def run(method: str, seed: int, cfg: LMExperimentConfig, eval_cap: int | None = None) -> dict:
    examples, candidates = generate_retention_stream(seed)
    segments = group(examples)
    model, tokenizer = load_model(cfg)
    replay = ReplayStore(cfg.replay_capacity, seed + 42)
    budget = BudgetCounter()
    rng = random.Random(seed)
    matrix = []
    promotion_log = []

    use_replay = method in {"replay", "fixed", "promotion"}
    use_latent = method == "promotion"

    for seg in sorted(segments):
        train = list(segments[seg]["train"])
        rng.shuffle(train)
        scope = "single" if method in {"sequential", "replay"} else "fast"
        params = model.set_trainable(scope)
        opt = __import__("torch").optim.AdamW(params, lr=cfg.online_lr)

        for ex in train:
            batch = [ex]
            replay_used = 0
            if use_replay:
                sampled = replay.sample(cfg.replay_per_online_step)
                batch += sampled
                replay_used = len(sampled)
            supervised_step(
                model, tokenizer, batch, opt, budget,
                use_fast=True, use_slow=True, use_latent=use_latent,
                update_latent=use_latent, replay_examples=replay_used,
            )
            budget.examples_seen += 1
            replay.add(ex)

        if method == "fixed":
            consolidate_slow(model, tokenizer, replay.items, budget, cfg)
            model.reset_fast()
        elif method == "promotion":
            protected = []
            for old in range(seg):
                protected.extend(segments[old]["test"])
            accepted, evidence = guarded_promotion(
                model, tokenizer,
                current_test=segments[seg]["test"],
                protected_test=protected,
                candidates=candidates,
                replay=replay,
                budget=budget,
                cfg=cfg,
            )
            promotion_log.append({"segment": seg, "accepted": accepted, **evidence})

        row = [float("nan")] * len(segments)
        for old in range(seg + 1):
            row[old] = multiple_choice_accuracy(
                model, tokenizer, segments[old]["test"], candidates,
                use_fast=True, use_slow=True, use_latent=use_latent,
                max_examples=eval_cap,
            )
        matrix.append(row)

    m = summarize(matrix)
    return {
        "classification": "PILOT unless EXP-001-PREREG budget checks are externally verified",
        "method": method,
        "seed": seed,
        "config": asdict(cfg),
        "metrics": asdict(m),
        "budget": asdict(budget),
        "matrix": matrix,
        "promotion_log": promotion_log,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["sequential", "replay", "fixed", "promotion"], required=True)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--eval-cap", type=int, default=None, help="Pilot-only cap on test examples per segment")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = LMExperimentConfig(model_name=args.model)
    payload = run(args.method, args.seed, cfg, args.eval_cap)
    out = args.out or ROOT / "results" / f"lm-{args.method}-{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=True))
    print(json.dumps({"method": args.method, "metrics": payload["metrics"], "budget": payload["budget"]}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
