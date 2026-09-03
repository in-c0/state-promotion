#!/usr/bin/env python3
"""EXP-001 Phase B arm runner on the explicit additive LoRA substrate.

Issue #5 is the execution contract. The historical soft-prefix runner
(`run_lm_pals.py`) is retained unchanged for reproducibility of PILOT-01/02 and
is not pooled with Phase-B statistics.

Architecture is frozen by issue #5 section 1:
  - frozen Qwen2.5-0.5B-Instruct at one immutable snapshot;
  - LoRA on every q_proj and v_proj, dropout 0, no trainable bias, alpha/r = 1;
  - representation selected by the predeclared EXP-001R gate: r = 2, online LR 3e-3;
  - B1/B2 one continuous rank-2r adapter; B3/B4/B5 independent rank-r fast and
    slow adapters, both additive at inference, sharing one latent-state channel.

Write accounting is Amendment A, on actual parameter elements rather than
nominal rank. P is one rank-r adapter's element count across all target
modules; a rank-2r write costs exactly 2P by rank linearity, asserted at run
time. For T online events every adaptive arm receives the same hard ceiling 2TP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_promotion.lm import (
    BudgetCounter,
    LMExperimentConfig,
    multiple_choice_accuracy,
    supervised_step,
)
from state_promotion.lora import (
    consolidate_slow_lora,
    guarded_promotion_lora,
    load_lora_model,
)
from state_promotion.metrics import summarize
from state_promotion.pals import (
    Example,
    generate_retention_stream,
    protocol_train_order,
)

MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
RANK = 2
ONLINE_LR = 3e-3
DEV_SEEDS = (20260901, 20260902, 20260903)

# adapter_mode, latent channel, online replay. Issue #5 section 1.
ARM_SPEC = {
    "b0_frozen":     {"mode": "two_timescale", "latent": False, "online_replay": False, "adapts": False},
    "b1_sequential": {"mode": "single",        "latent": False, "online_replay": False, "adapts": True},
    "b2_replay":     {"mode": "single",        "latent": False, "online_replay": True,  "adapts": True},
    "b3_fixed":      {"mode": "two_timescale", "latent": True,  "online_replay": False, "adapts": True},
    "b4_random":     {"mode": "two_timescale", "latent": True,  "online_replay": False, "adapts": True},
    "b5_promotion":  {"mode": "two_timescale", "latent": True,  "online_replay": False, "adapts": True},
}
ARMS = tuple(ARM_SPEC)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001 - provenance degrades to "unknown", never crashes a run
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
        h.update(path.relative_to(ROOT).as_posix().encode()); h.update(b"\0")
        h.update(path.read_bytes()); h.update(b"\0")
    return h.hexdigest()


def backbone_digest(model) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if ".adapters." in name:
            continue
        h.update(name.encode()); h.update(b"\0")
        h.update(p.detach().cpu().contiguous().numpy().tobytes()); h.update(b"\0")
    return h.hexdigest()


def unit_write_parameters(model, mode: str) -> int:
    """P: element count of one rank-r adapter across all target modules.

    Amendment A is defined on actual elements, not nominal rank. For the
    single-adapter arms the live adapter is rank 2r, so P is half its count and
    the halving must be exact -- that exactness *is* the rank-linearity claim.
    """
    if mode == "two_timescale":
        fast = model.plastic_parameter_count("fast")
        slow = model.plastic_parameter_count("slow")
        if fast != slow:
            raise AssertionError(f"fast/slow capacity mismatch: {fast} != {slow}")
        return fast
    live = model.plastic_parameter_count("single")
    if live % 2 != 0:
        raise AssertionError(f"rank-2r adapter count {live} is not twice any integer")
    return live // 2


def build_probes(*, segment_train, replay_items, all_examples, segment):
    """Reuse the preregistered probe construction from the historical runner."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_lm_pals import build_promotion_probes

    return build_promotion_probes(
        stream="retention",
        segment=segment,
        segment_train=segment_train,
        replay_items=replay_items,
        all_examples=all_examples,
    )


def run_arm(
    *,
    arm: str,
    seed: int,
    cfg: LMExperimentConfig,
    device: str,
    cadence_k: int | None = None,
    random_commit_segments: set[int] | None = None,
    commit_steps: dict[int, int] | None = None,
    eval_cap: int | None = None,
) -> dict:
    from state_promotion.lm import ReplayStore

    spec = ARM_SPEC[arm]
    examples, candidates = generate_retention_stream(seed)
    segments: dict[int, dict[str, list[Example]]] = {}
    for seg in range(6):
        segments[seg] = {
            "train": [e for e in examples if e.segment == seg and e.split == "train"],
            "test": [e for e in examples if e.segment == seg and e.split == "test"],
        }
    # Amendment G: the protocol's presentation order, one RNG per run.
    ordered = protocol_train_order({s: segments[s]["train"] for s in segments}, seed)

    model_init_seed = seed + 1701
    torch.manual_seed(model_init_seed)
    model, tokenizer, device_report = load_lora_model(
        cfg, rank=RANK, adapter_mode=spec["mode"], device=device, seed=model_init_seed
    )
    backbone_before = backbone_digest(model)

    P = unit_write_parameters(model, spec["mode"])
    T = sum(len(v["train"]) for v in segments.values())
    ceiling = 2 * T * P

    budget = BudgetCounter()
    budget.write_budget_units = ceiling
    replay = ReplayStore(cfg.replay_capacity, seed + 42)

    scope = "single" if spec["mode"] == "single" else "fast"
    online_opt = None
    online_scope = None
    matrix: list[list[float]] = []
    commit_log: list[dict] = []
    pending_events = 0
    started = time.time()

    for seg in range(6):
        train = ordered[seg]
        pending_events += len(train)

        if spec["adapts"]:
            params = model.set_trainable(scope)
            # Amendment C: optimizer state is continuous across segment
            # boundaries and is reset only when the fast parameters themselves
            # are reset.
            if online_opt is None or online_scope != scope:
                online_opt = torch.optim.AdamW(params, lr=cfg.online_lr)
                online_scope = scope
            for ex in train:
                batch = [ex]
                replay_used = 0
                if spec["online_replay"]:
                    sampled = replay.sample(cfg.replay_per_online_step)
                    batch += sampled
                    replay_used = len(sampled)
                supervised_step(
                    model, tokenizer, batch, online_opt, budget,
                    use_fast=True, use_slow=True,
                    use_latent=spec["latent"], update_latent=spec["latent"],
                    replay_examples=replay_used,
                )
                budget.examples_seen += 1
                replay.add(ex)
        else:
            for ex in train:
                budget.examples_seen += 1
                replay.add(ex)

        # ---- consolidation policy ----
        if arm == "b3_fixed":
            if cadence_k is None:
                raise ValueError("b3_fixed requires cadence_k")
            if (seg + 1) % cadence_k == 0:
                # Accumulated allowance: a commit after k segments may spend the
                # slow writes the skipped segments did not. Every cadence
                # therefore reaches the same lifetime slow allocation.
                steps = pending_events
                writes_before = budget.parameter_write_units
                consolidate_slow_lora(
                    model, tokenizer, replay.items, budget, cfg,
                    use_latent=spec["latent"], steps=steps,
                )
                commit_log.append({
                    "segment": seg, "accepted": True, "planned_steps": steps,
                    "write_units": budget.parameter_write_units - writes_before,
                })
                model.reset_fast()
                online_opt = None
                online_scope = None
                pending_events = 0

        elif arm == "b4_random":
            if random_commit_segments is None:
                raise ValueError("b4_random requires random_commit_segments matched to B5")
            if seg in random_commit_segments:
                steps = (commit_steps or {}).get(seg)
                if steps is None:
                    raise ValueError(f"b4_random segment {seg} has no B5-matched allocation")
                writes_before = budget.parameter_write_units
                consolidate_slow_lora(
                    model, tokenizer, replay.items, budget, cfg,
                    use_latent=spec["latent"], steps=steps,
                )
                commit_log.append({
                    "segment": seg, "accepted": True, "planned_steps": steps,
                    "write_units": budget.parameter_write_units - writes_before,
                })
                model.reset_fast()
                online_opt = None
                online_scope = None
                pending_events = 0
            else:
                commit_log.append({"segment": seg, "accepted": False, "planned_steps": 0, "write_units": 0})

        elif arm == "b5_promotion":
            current_probe, protected_probe = build_probes(
                segment_train=train, replay_items=replay.items,
                all_examples=examples, segment=seg,
            )
            leaked = sum(ex.split != "train" for ex in current_probe + protected_probe)
            if leaked:
                raise AssertionError(f"held-out example entered the promotion gate: {leaked}")
            steps = len(train)
            writes_before = budget.parameter_write_units
            accepted, evidence = guarded_promotion_lora(
                model, tokenizer,
                current_probe=current_probe, protected_probe=protected_probe,
                candidates=candidates, replay=replay, budget=budget, cfg=cfg,
                use_latent=spec["latent"], rollback_on_retention=True,
                consolidation_steps=steps,
            )
            commit_log.append({
                "segment": seg, "accepted": bool(accepted), "planned_steps": steps if accepted else 0,
                "write_units": budget.parameter_write_units - writes_before,
                **{k: float(v) for k, v in evidence.items()},
            })
            if accepted:
                online_opt = None
                online_scope = None
                pending_events = 0

        row = [float("nan")] * 6
        for old in range(seg + 1):
            row[old] = multiple_choice_accuracy(
                model, tokenizer, segments[old]["test"], candidates,
                use_fast=True, use_slow=True, use_latent=spec["latent"],
                max_examples=eval_cap,
            )
        matrix.append(row)

    backbone_after = backbone_digest(model)
    diagonal = [matrix[i][i] for i in range(6)]
    final_row = matrix[-1]
    # Canonical preregistered metric definitions (src/state_promotion/metrics.py),
    # so B3 selection reads the same numbers the rest of EXP-001 reports.
    m = summarize(matrix)

    result = {
        "protocol": "EXP-001B-lora-arm-v1",
        "status": "DEVELOPMENT_ONLY",
        "arm": arm,
        "stream": "retention",
        "seed": seed,
        "adapter_mode": spec["mode"],
        "latent_state_enabled": spec["latent"],
        "online_replay": spec["online_replay"],
        "rank": RANK,
        "online_lr": cfg.online_lr,
        "consolidation_lr": cfg.consolidation_lr,
        "consolidation_batch": cfg.consolidation_batch,
        "cadence_k": cadence_k,
        "unit_write_parameters_P": P,
        "online_events_T": T,
        "write_ceiling_2TP": ceiling,
        "retention_matrix": matrix,
        "diagonal": diagonal,
        "mean_diagonal": m.average_plasticity,
        "final_row": final_row,
        "final_average": m.final_average,
        "average_forgetting": m.average_forgetting,
        "retention_auc": m.retention_auc,
        "commit_log": commit_log,
        "accepted_commit_segments": [c["segment"] for c in commit_log if c.get("accepted")],
        "budget": {
            "optimizer_steps": budget.optimizer_steps,
            "parameter_write_units": budget.parameter_write_units,
            "write_budget_units": budget.write_budget_units,
            "write_budget_exhausted_steps": budget.write_budget_exhausted_steps,
            "examples_seen": budget.examples_seen,
            "training_examples_processed": budget.training_examples_processed,
            "tokens_processed": budget.tokens_processed,
            "replay_examples_used": budget.replay_examples_used,
            "decision_forward_calls": budget.decision_forward_calls,
            "decision_tokens_processed": budget.decision_tokens_processed,
        },
        "backbone_sha256_before": backbone_before,
        "backbone_sha256_after": backbone_after,
        "backbone_frozen": backbone_before == backbone_after,
        "finite_matrix": all(
            all(math.isfinite(v) for v in row if not math.isnan(v))
            for row in matrix
        ),
        "provenance": model.provenance,
        "device": device,
        "compute_dtype": str(next(model.base.parameters()).dtype),
        "device_numerics": device_report,
        "git_sha": git_sha(),
        "source_tree_sha256": source_tree_sha256(),
        "elapsed_seconds": time.time() - started,
    }

    # ---- machine-checked invariants (issue #5 sections 1-2) ----
    if not result["backbone_frozen"]:
        raise AssertionError("frozen foundation weights changed during the run")
    if budget.parameter_write_units > ceiling:
        raise AssertionError(
            f"write budget exceeded: {budget.parameter_write_units} > {ceiling}"
        )
    if spec["adapts"] and spec["mode"] == "single":
        expected = 2 * P * T
        if budget.parameter_write_units != expected:
            raise AssertionError(
                f"{arm}: rank-2r arm should write exactly {expected}, got {budget.parameter_write_units}"
            )
    if spec["adapts"] and spec["mode"] == "two_timescale" and budget.parameter_write_units < T * P:
        raise AssertionError(
            f"{arm}: fast pass should write at least {T * P}, got {budget.parameter_write_units}"
        )
    if not spec["adapts"] and budget.parameter_write_units != 0:
        raise AssertionError("frozen arm performed parameter writes")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--allow-nonreproducible-device", action="store_true",
        help="Acknowledge that a non-CPU device does not reproduce the EXP-001R gates.",
    )
    ap.add_argument("--cadence-k", type=int, default=None)
    ap.add_argument("--slow-lr", type=float, default=None)
    ap.add_argument("--consolidation-batch", type=int, default=1)
    ap.add_argument("--random-commit-segments", default=None)
    ap.add_argument("--commit-steps", default=None,
                    help="JSON {segment: steps} matched to B5 per-commit allocation")
    ap.add_argument("--eval-cap", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.device != "cpu" and not args.allow_nonreproducible_device:
        raise SystemExit(
            f"refusing device={args.device!r}: verify_device_numerics compares one forward "
            "loss and does not establish training equivalence. A full B1 lifetime on MPS "
            "diverged from CPU (mean diagonal 0.667 vs 0.944, forgetting 0.333 vs 0.533) "
            "while every existing guard passed. EXP-001B is CPU-only; pass "
            "--allow-nonreproducible-device to override deliberately."
        )

    cfg = LMExperimentConfig(
        model_revision=MODEL_REVISION,
        online_lr=ONLINE_LR,
        consolidation_batch=args.consolidation_batch,
    )
    if args.slow_lr is not None:
        cfg.consolidation_lr = args.slow_lr

    commits = None
    if args.random_commit_segments:
        commits = {int(x) for x in args.random_commit_segments.split(",") if x.strip() != ""}
    steps_map = {int(k): int(v) for k, v in json.loads(args.commit_steps).items()} if args.commit_steps else None

    result = run_arm(
        arm=args.arm, seed=args.seed, cfg=cfg, device=args.device,
        cadence_k=args.cadence_k, random_commit_segments=commits,
        commit_steps=steps_map, eval_cap=args.eval_cap,
    )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in (
        "arm", "seed", "mean_diagonal", "average_forgetting", "final_average",
        "accepted_commit_segments", "backbone_frozen",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
