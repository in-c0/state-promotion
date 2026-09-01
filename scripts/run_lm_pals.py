#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_promotion.lm import (  # noqa: E402
    BudgetCounter,
    LMExperimentConfig,
    ReplayStore,
    completion_nll,
    consolidate_slow,
    guarded_promotion,
    load_model,
    multiple_choice_accuracy,
    supervised_step,
)
from state_promotion.metrics import summarize  # noqa: E402
from state_promotion.pals import Example, generate_retention_stream, generate_revision_stream  # noqa: E402

METHODS = [
    "frozen",
    "sequential",
    "replay",
    "fixed",
    "random",
    "promotion",
    "promotion-no-latent",
    "promotion-reset-latent",
    "promotion-no-rollback",
]


def group(examples):
    d = defaultdict(lambda: {"train": [], "test": []})
    for ex in examples:
        d[ex.segment][ex.split].append(ex)
    return d


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def replay_bytes(items: list[Example]) -> int:
    return sum(len(ex.prompt.encode("utf-8")) + len(ex.target.encode("utf-8")) for ex in items)


def predict(model, tokenizer, ex: Example, candidates: list[str], *, use_fast: bool, use_slow: bool,
            use_latent: bool) -> str:
    scores = [completion_nll(model, tokenizer, ex, c, use_fast=use_fast, use_slow=use_slow,
                             use_latent=use_latent) for c in candidates]
    return candidates[min(range(len(scores)), key=scores.__getitem__)]


def active_revision_tests(all_examples: list[Example], through_segment: int) -> tuple[list[Example], dict[tuple[str, str], str]]:
    latest: dict[tuple[str, str], Example] = {}
    stale: dict[tuple[str, str], str] = {}
    ordered = sorted(
        [e for e in all_examples if e.split == "test" and e.segment <= through_segment],
        key=lambda e: (e.segment, e.version),
    )
    for ex in ordered:
        pair = (ex.context, ex.key)
        if pair in latest and latest[pair].target != ex.target:
            stale[pair] = latest[pair].target
        latest[pair] = ex
    return list(latest.values()), stale


def revision_metrics(model, tokenizer, all_examples: list[Example], candidates: list[str], through_segment: int,
                     *, use_fast: bool, use_slow: bool, use_latent: bool, eval_cap: int | None) -> dict[str, float]:
    active, stale = active_revision_tests(all_examples, through_segment)
    if eval_cap is not None:
        active = active[:eval_cap]
    if not active:
        return {}
    preds = [(ex, predict(model, tokenizer, ex, candidates, use_fast=use_fast, use_slow=use_slow,
                          use_latent=use_latent)) for ex in active]

    def acc(rows):
        return sum(int(pred == ex.target) for ex, pred in rows) / len(rows) if rows else float("nan")

    exception = [(ex, pred) for ex, pred in preds if ex.relation == "context_exception"]
    supersession = [(ex, pred) for ex, pred in preds if ex.relation == "supersedes"]
    stale_rows = [(ex, pred) for ex, pred in supersession if (ex.context, ex.key) in stale]
    stale_rate = (
        sum(int(pred == stale[(ex.context, ex.key)]) for ex, pred in stale_rows) / len(stale_rows)
        if stale_rows else float("nan")
    )
    return {
        "active_world_accuracy": acc(preds),
        "context_exception_accuracy": acc(exception),
        "supersession_accuracy": acc(supersession),
        "stale_answer_rate": stale_rate,
    }


def run(method: str, seed: int, cfg: LMExperimentConfig, *, stream: str = "retention",
        eval_cap: int | None = None, random_commit_segments: set[int] | None = None) -> dict:
    if stream == "retention":
        examples, candidates = generate_retention_stream(seed)
    elif stream == "revision":
        examples, candidates = generate_revision_stream(seed)
    else:
        raise ValueError(stream)

    segments = group(examples)
    model, tokenizer = load_model(cfg)
    replay = ReplayStore(cfg.replay_capacity, seed + 42)
    budget = BudgetCounter()
    rng = random.Random(seed)
    matrix = []
    promotion_log = []
    revision_trajectory = []
    started = time.perf_counter()

    use_replay = method in {"replay", "fixed", "random", "promotion", "promotion-no-latent",
                            "promotion-reset-latent", "promotion-no-rollback"}
    use_latent = method in {"promotion", "promotion-reset-latent", "promotion-no-rollback"}
    if method == "promotion-no-latent":
        use_latent = False

    for seg in sorted(segments):
        if method == "promotion-reset-latent" and seg > 0:
            model.reset_latent()

        train = list(segments[seg]["train"])
        rng.shuffle(train)

        if method != "frozen":
            scope = "single" if method in {"sequential", "replay"} else "fast"
            params = model.set_trainable(scope)
            opt = torch.optim.AdamW(params, lr=cfg.online_lr)

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
        else:
            for ex in train:
                budget.examples_seen += 1
                replay.add(ex)

        if method == "fixed":
            consolidate_slow(model, tokenizer, replay.items, budget, cfg)
            model.reset_fast()
        elif method == "random":
            if random_commit_segments is None:
                raise ValueError("random method requires --random-commit-segments")
            if seg in random_commit_segments:
                consolidate_slow(model, tokenizer, replay.items, budget, cfg)
                model.reset_fast()
                promotion_log.append({"segment": seg, "accepted": True, "gate": 2.0})
            else:
                promotion_log.append({"segment": seg, "accepted": False, "gate": 0.0})
        elif method.startswith("promotion"):
            protected = []
            if stream == "retention":
                for old in range(seg):
                    protected.extend(segments[old]["test"])
            else:
                protected, _ = active_revision_tests(examples, seg - 1)
            accepted, evidence = guarded_promotion(
                model, tokenizer,
                current_test=segments[seg]["test"],
                protected_test=protected,
                candidates=candidates,
                replay=replay,
                budget=budget,
                cfg=cfg,
                use_latent=use_latent,
                rollback_on_retention=method != "promotion-no-rollback",
            )
            promotion_log.append({"segment": seg, "accepted": accepted, **evidence})

        if stream == "retention":
            row = [float("nan")] * len(segments)
            for old in range(seg + 1):
                row[old] = multiple_choice_accuracy(
                    model, tokenizer, segments[old]["test"], candidates,
                    use_fast=True, use_slow=True, use_latent=use_latent,
                    max_examples=eval_cap,
                )
            matrix.append(row)
        else:
            revision_trajectory.append({
                "segment": seg,
                **revision_metrics(
                    model, tokenizer, examples, candidates, seg,
                    use_fast=True, use_slow=True, use_latent=use_latent, eval_cap=eval_cap,
                ),
            })

    elapsed = time.perf_counter() - started
    metrics = asdict(summarize(matrix)) if stream == "retention" else (revision_trajectory[-1] if revision_trajectory else {})
    base_params = sum(p.numel() for p in model.base.parameters())
    plastic_params = model.slow_prompt.numel() + model.fast_prompt.numel()
    frozen_backbone = all(not p.requires_grad for p in model.base.parameters())
    accepted_segments = [int(x["segment"]) for x in promotion_log if x.get("accepted")]

    invalid = []
    if not frozen_backbone:
        invalid.append("foundation_backbone_not_frozen")
    vals = [v for v in metrics.values() if isinstance(v, (int, float))]
    if any(not torch.isfinite(torch.tensor(v)).item() for v in vals if v == v):
        invalid.append("non_finite_metric")

    return {
        "classification": "PILOT",
        "method": method,
        "stream": stream,
        "seed": seed,
        "git_sha": git_sha(),
        "config": asdict(cfg),
        "model": {
            "name": cfg.model_name,
            "device": str(model.device),
            "base_parameters": base_params,
            "plastic_parameter_capacity": plastic_params,
            "backbone_frozen": frozen_backbone,
        },
        "metrics": metrics,
        "budget": {
            **asdict(budget),
            "replay_capacity_examples": cfg.replay_capacity,
            "replay_final_examples": len(replay.items),
            "replay_final_bytes": replay_bytes(replay.items),
            "wall_seconds": elapsed,
        },
        "matrix": matrix,
        "revision_trajectory": revision_trajectory,
        "promotion_log": promotion_log,
        "accepted_commit_segments": accepted_segments,
        "random_commit_segments": sorted(random_commit_segments or []),
        "invalidation_reasons": invalid,
    }


def parse_segments(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return set()
    return {int(x) for x in raw.split(",")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--stream", choices=["retention", "revision"], default="retention")
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--eval-cap", type=int, default=None, help="Pilot-only cap on test examples")
    p.add_argument("--random-commit-segments", default=None, help="Comma-separated segments for matched random control")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = LMExperimentConfig(model_name=args.model)
    payload = run(
        args.method, args.seed, cfg,
        stream=args.stream,
        eval_cap=args.eval_cap,
        random_commit_segments=parse_segments(args.random_commit_segments),
    )
    out = args.out or ROOT / "results" / f"lm-{args.stream}-{args.method}-{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=True))
    print(json.dumps({"method": args.method, "stream": args.stream, "metrics": payload["metrics"],
                      "budget": payload["budget"], "invalid": payload["invalidation_reasons"]}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
