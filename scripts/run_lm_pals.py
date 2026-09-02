#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
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
    encode_example,
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
    "promotion-no-replay",
    "promotion-no-slow",
]


def group(examples):
    d = defaultdict(lambda: {"train": [], "test": []})
    for ex in examples:
        d[ex.segment][ex.split].append(ex)
    return d


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def source_tree_sha256() -> str:
    """Stable fallback provenance hash for source archives without .git metadata."""
    h = hashlib.sha256()
    roots = [ROOT / "src", ROOT / "scripts", ROOT / "tests", ROOT / "experiments", ROOT / "docs"]
    files = [ROOT / "README.md", ROOT / "pyproject.toml", ROOT / "Makefile"]
    for root in roots:
        if root.exists():
            files.extend(x for x in root.rglob("*") if x.is_file())
    for path in sorted(set(files), key=lambda x: x.relative_to(ROOT).as_posix()):
        rel = path.relative_to(ROOT).as_posix()
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        h.update(rel.encode("utf-8")); h.update(b"\0")
        h.update(path.read_bytes()); h.update(b"\0")
    return h.hexdigest()


def replay_bytes(items: list[Example]) -> int:
    return sum(len(ex.prompt.encode("utf-8")) + len(ex.target.encode("utf-8")) for ex in items)


def dedupe_probe_examples(items: list[Example]) -> list[Example]:
    """Keep one attributable observation per active context/key/target mapping."""
    seen: set[tuple[str, str, str]] = set()
    out: list[Example] = []
    for ex in items:
        key = (ex.context, ex.key, ex.target)
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def active_train_targets(all_examples: list[Example], through_segment: int) -> dict[tuple[str, str], str]:
    """Latest observed training target for each context/key pair."""
    latest: dict[tuple[str, str], Example] = {}
    ordered = sorted(
        [e for e in all_examples if e.split == "train" and e.segment <= through_segment],
        key=lambda e: (e.segment, e.version),
    )
    for ex in ordered:
        latest[(ex.context, ex.key)] = ex
    return {pair: ex.target for pair, ex in latest.items()}


def build_promotion_probes(*, stream: str, segment: int, segment_train: list[Example],
                           replay_items: list[Example], all_examples: list[Example]) -> tuple[list[Example], list[Example]]:
    """Build promotion evidence only from already-observed training examples.

    Current probes come from the segment just learned. Protected probes come from
    the bounded replay reservoir and therefore do not expose held-out evaluation
    labels to the promotion arm. On revision streams, obsolete replay entries are
    excluded from protection once a superseding training observation is seen.
    """
    current = dedupe_probe_examples([e for e in segment_train if e.split == "train"])
    protected_pool = [e for e in replay_items if e.split == "train" and e.segment < segment]
    if stream == "revision":
        active = active_train_targets(all_examples, segment)
        protected_pool = [
            e for e in protected_pool
            if active.get((e.context, e.key)) == e.target
        ]
    protected = dedupe_probe_examples(protected_pool)
    if any(e.split != "train" for e in current + protected):
        raise AssertionError("promotion probes must never contain held-out evaluation examples")
    return current, protected


def probe_hash(items: list[Example]) -> str:
    h = hashlib.sha256()
    for ex in items:
        h.update(ex.prompt.encode("utf-8"))
        h.update(b"\0")
        h.update(ex.target.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()




def _json_sha256(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def encoded_example_audit(tokenizer, ex: Example) -> dict:
    """Serialize exactly the tokenized supervised input, separated from audit metadata."""
    ids, mask, labels = encode_example(tokenizer, ex)
    model_input = {
        "input_ids": ids[0].tolist(),
        "attention_mask": mask[0].tolist(),
        "labels": labels[0].tolist(),
    }
    return {
        "audit_metadata": {
            "stream": ex.stream,
            "segment": ex.segment,
            "split": ex.split,
            "relation": ex.relation,
            "version": ex.version,
        },
        "model_text": {"prompt": ex.prompt, "completion": ex.target},
        "model_input": model_input,
        "model_input_sha256": _json_sha256(model_input),
    }


def online_batch_audit(tokenizer, examples: list[Example], *, update_applied: bool, replay_examples: int) -> dict:
    rows = [encoded_example_audit(tokenizer, ex) for ex in examples]
    model_payload = [row["model_input"] for row in rows]
    return {
        "update_applied": update_applied,
        "replay_examples": replay_examples,
        "examples": rows,
        "model_visible_batch_sha256": _json_sha256(model_payload),
        "all_source_splits_train": all(ex.split == "train" for ex in examples),
    }


def eval_query_audit(tokenizer, ex: Example, candidates: list[str]) -> dict:
    """Serialize one multiple-choice query exactly as candidates are scored.

    The gold target is audit-only metadata. Every candidate, including the gold one,
    is encoded and scored through the same completion-NLL path.
    """
    candidate_rows = []
    for candidate in candidates:
        candidate_ex = replace(ex, target=candidate)
        row = encoded_example_audit(tokenizer, candidate_ex)
        candidate_rows.append({
            "candidate": candidate,
            "model_input": row["model_input"],
            "model_input_sha256": row["model_input_sha256"],
        })
    model_payload = [row["model_input"] for row in candidate_rows]
    return {
        "audit_metadata": {
            "stream": ex.stream,
            "segment": ex.segment,
            "split": ex.split,
            "relation": ex.relation,
            "version": ex.version,
            "gold_target": ex.target,
            "gold_target_is_model_privileged": False,
        },
        "prompt": ex.prompt,
        "candidate_order": list(candidates),
        "candidate_encodings": candidate_rows,
        "model_visible_query_sha256": _json_sha256(model_payload),
    }

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
        eval_cap: int | None = None, random_commit_segments: set[int] | None = None,
        write_step_budget: int | None = None, device: str | None = None) -> dict:
    if stream == "retention":
        examples, candidates = generate_retention_stream(seed)
    elif stream == "revision":
        examples, candidates = generate_revision_stream(seed)
    else:
        raise ValueError(stream)

    segments = group(examples)
    model_init_seed = seed + 1701
    torch.manual_seed(model_init_seed)
    model, tokenizer, device_report = load_model(cfg, device=device, seed=model_init_seed)
    replay = ReplayStore(cfg.replay_capacity, seed + 42)
    budget = BudgetCounter()
    total_train_events = sum(len(v["train"]) for v in segments.values())
    write_unit_parameters = model.fast_prompt.numel()
    effective_write_step_budget = 2 * total_train_events if write_step_budget is None else write_step_budget
    budget.write_budget_units = effective_write_step_budget * write_unit_parameters
    rng = random.Random(seed)
    matrix = []
    promotion_log = []
    promotion_probe_audit = []
    batch_audit = {"first_online_batch": None, "first_eval_query": None}
    revision_trajectory = []
    started = time.perf_counter()
    online_opt = None
    online_scope = None

    # B2 spends replay compute in the online update batch. The two-timescale
    # arms reserve replay for slow consolidation so their adaptation-token envelope
    # can be matched against B2 while preserving fast/slow separation.
    use_online_replay = method in {"replay", "promotion-no-slow"}
    # Architecture-match the two-timescale routing controls: fixed, random,
    # and learned promotion all receive the same persistent latent state.
    # Otherwise B5-vs-B3/B4 would confound routing with latent-state access.
    use_latent = method in {"fixed", "random", "promotion", "promotion-reset-latent",
                            "promotion-no-rollback", "promotion-no-replay", "promotion-no-slow"}
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
            # B1/B2 are genuinely continuous baselines: optimizer state persists
            # across orchestration boundaries. Two-timescale arms keep fast
            # optimizer state until the fast parameters themselves are reset.
            if online_opt is None or online_scope != scope:
                online_opt = torch.optim.AdamW(params, lr=cfg.online_lr)
                online_scope = scope

            for ex in train:
                batch = [ex]
                replay_used = 0
                if use_online_replay:
                    sampled = replay.sample(cfg.replay_per_online_step)
                    batch += sampled
                    replay_used = len(sampled)
                if batch_audit["first_online_batch"] is None:
                    batch_audit["first_online_batch"] = online_batch_audit(
                        tokenizer, batch, update_applied=True, replay_examples=replay_used,
                    )
                supervised_step(
                    model, tokenizer, batch, online_opt, budget,
                    use_fast=True, use_slow=True, use_latent=use_latent,
                    update_latent=use_latent, replay_examples=replay_used,
                )
                budget.examples_seen += 1
                if method != "promotion-no-replay":
                    replay.add(ex)
        else:
            if train and batch_audit["first_online_batch"] is None:
                batch_audit["first_online_batch"] = online_batch_audit(
                    tokenizer, [train[0]], update_applied=False, replay_examples=0,
                )
            for ex in train:
                budget.examples_seen += 1
                replay.add(ex)

        if method == "fixed":
            consolidate_slow(model, tokenizer, replay.items, budget, cfg, steps=len(train))
            model.reset_fast()
            online_opt = None
            online_scope = None
        elif method == "random":
            if random_commit_segments is None:
                raise ValueError("random method requires --random-commit-segments")
            if seg in random_commit_segments:
                consolidate_slow(model, tokenizer, replay.items, budget, cfg, steps=len(train))
                model.reset_fast()
                online_opt = None
                online_scope = None
                promotion_log.append({"segment": seg, "accepted": True, "gate": 2.0})
            else:
                promotion_log.append({"segment": seg, "accepted": False, "gate": 0.0})
        elif method.startswith("promotion") and method != "promotion-no-slow":
            current_probe, protected_probe = build_promotion_probes(
                stream=stream,
                segment=seg,
                segment_train=train,
                replay_items=replay.items,
                all_examples=examples,
            )
            promotion_probe_audit.append({
                "segment": seg,
                "current_probe_examples": len(current_probe),
                "protected_probe_examples": len(protected_probe),
                "current_probe_sha256": probe_hash(current_probe),
                "protected_probe_sha256": probe_hash(protected_probe),
                "heldout_gate_example_count": sum(ex.split != "train" for ex in current_probe + protected_probe),
            })
            accepted, evidence = guarded_promotion(
                model, tokenizer,
                current_probe=current_probe,
                protected_probe=protected_probe,
                candidates=candidates,
                replay=replay,
                budget=budget,
                cfg=cfg,
                use_latent=use_latent,
                rollback_on_retention=method != "promotion-no-rollback",
                consolidation_steps=len(train),
                consolidation_examples=train if method == "promotion-no-replay" else None,
            )
            promotion_log.append({"segment": seg, "accepted": accepted, **evidence})
            if accepted:
                online_opt = None
                online_scope = None

        if batch_audit["first_eval_query"] is None and segments[seg]["test"]:
            batch_audit["first_eval_query"] = eval_query_audit(tokenizer, segments[seg]["test"][0], candidates)

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
        "source_tree_sha256": source_tree_sha256(),
        "config": asdict(cfg),
        "model": {
            "name": cfg.model_name,
            "snapshot_revision": cfg.model_revision,
            "device": str(model.device),
            "device_requested": device or "auto",
            "device_numerics_check": device_report,
            "base_parameters": base_params,
            "plastic_parameter_capacity": plastic_params,
            "backbone_frozen": frozen_backbone,
            "model_revision": getattr(model.base.config, "_commit_hash", None),
            "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
            "model_init_seed": model_init_seed,
            "latent_state_enabled": use_latent,
            "write_unit_parameters": write_unit_parameters,
        },
        "metrics": metrics,
        "budget": {
            **asdict(budget),
            "replay_capacity_examples": cfg.replay_capacity,
            "replay_final_examples": len(replay.items),
            "replay_final_bytes": replay_bytes(replay.items),
            "write_step_budget": effective_write_step_budget,
            "write_budget_units": budget.write_budget_units,
            "token_parameter_compute_proxy": budget.tokens_processed * base_params,
            "estimated_training_flops_frozen_backbone": 4 * budget.tokens_processed * base_params,
            "decision_token_parameter_compute_proxy": budget.decision_tokens_processed * base_params,
            "estimated_decision_flops_frozen_backbone": 2 * budget.decision_tokens_processed * base_params,
            "flop_estimate_note": "Coarse 4*N*tokens estimate for forward plus input-gradient backprop through a frozen transformer; report as an order-of-magnitude proxy, not hardware FLOPs.",
            "decision_flop_estimate_note": "Coarse 2*N*tokens estimate for no-grad promotion-gate forward passes. These are decision-time inference costs, not post-hoc evaluation costs.",
            "wall_seconds": elapsed,
        },
        "matrix": matrix,
        "revision_trajectory": revision_trajectory,
        "promotion_log": promotion_log,
        "promotion_probe_audit": promotion_probe_audit,
        "batch_audit": batch_audit,
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
    p.add_argument("--model-revision", default=None, help="Immutable Hugging Face snapshot SHA. Required for a validator-clean pilot.")
    p.add_argument("--device", default=None,
                   help="Force a device (cpu/mps/cuda). Default auto-selects and numerically verifies any accelerator.")
    p.add_argument("--eval-cap", type=int, default=None, help="Pilot-only cap on test examples")
    p.add_argument("--random-commit-segments", default=None, help="Comma-separated segments for matched random control")
    p.add_argument("--write-step-budget", type=int, default=None,
                   help="Write ceiling in fast-adapter-equivalent steps; default is 2x unique train events")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = LMExperimentConfig(model_name=args.model, model_revision=args.model_revision)
    payload = run(
        args.method, args.seed, cfg,
        stream=args.stream,
        device=args.device,
        eval_cap=args.eval_cap,
        random_commit_segments=parse_segments(args.random_commit_segments),
        write_step_budget=args.write_step_budget,
    )
    out = args.out or ROOT / "results" / f"lm-{args.stream}-{args.method}-{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=True))
    print(json.dumps({"method": args.method, "stream": args.stream, "metrics": payload["metrics"],
                      "budget": payload["budget"], "invalid": payload["invalidation_reasons"]}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
