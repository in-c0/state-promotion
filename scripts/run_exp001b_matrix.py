#!/usr/bin/env python3
"""EXP-001 Phase B Stage B2: the development arm matrix.

Issue #5 section 5. Run order is B0/B1/B2/B3, then B5, then B4 -- B4 is
generated only after B5 so it can be matched to B5's actual commit count *and*
its per-commit slow-write allocation, not merely its number of commits.

B3 uses the configuration frozen by the Stage B1 grid; this script reads it from
that grid's summary and refuses to run if the grid did not select one.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp001b_arm import MODEL_REVISION, ONLINE_LR, git_sha, run_arm

from state_promotion.lm import LMExperimentConfig

DEV_SEEDS = (20260901, 20260902, 20260903)
BASELINE_ORDER = ("b0_frozen", "b1_sequential", "b2_replay", "b3_fixed")
BATCH_POINTS = (1, 2, 4)   # issue #5 section 3, predeclared before B5 is seen


def load_frozen_b3(path: Path) -> tuple[int, float]:
    summary = json.loads(path.read_text())
    if summary.get("protocol") != "EXP-001B-b3-strengthening-grid-v1":
        raise ValueError("unexpected B3 grid protocol")
    if not summary.get("b5_allowed") or not summary.get("selected"):
        raise ValueError("B3 grid selected no cell; B5 is forbidden")
    sel = summary["selected"]
    return int(sel["cadence_k"]), float(sel["slow_lr"])


def make_cfg(*, slow_lr: float, consolidation_batch: int = 1,
             replay_per_online_step: int = 1) -> LMExperimentConfig:
    cfg = LMExperimentConfig(model_revision=MODEL_REVISION, online_lr=ONLINE_LR)
    cfg.consolidation_lr = slow_lr
    cfg.consolidation_batch = consolidation_batch
    cfg.replay_per_online_step = replay_per_online_step
    return cfg


def match_b4_to_b5(b5: dict, seed: int) -> tuple[set[int], dict[int, int]]:
    """Count- and allocation-matched random routing.

    B4 receives the same number of commits as B5 and the same per-commit slow
    step allocation, placed at randomly chosen segments. If B5 declined every
    candidate, B4 commits nothing -- matching means matching, including to zero.
    """
    accepted = [c for c in b5["commit_log"] if c.get("accepted")]
    n = len(accepted)
    rng = random.Random(seed + 4004)
    chosen = sorted(rng.sample(range(6), n)) if n else []
    steps = {seg: int(accepted[i]["planned_steps"]) for i, seg in enumerate(chosen)}
    return set(chosen), steps


def write(out_dir: Path, name: str, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def brief(tag: str, r: dict, t0: float) -> None:
    print(
        f"    {tag} diag={r['mean_diagonal']:.3f} forget={r['average_forgetting']:.3f} "
        f"auc={r['retention_auc']:.3f} final={r['final_average']:.3f} "
        f"commits={len(r['accepted_commit_segments'])} "
        f"writes={r['budget']['parameter_write_units']}/{r['budget']['write_budget_units']} "
        f"{time.time() - t0:.0f}s",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--b3-summary", default="results/exp001b-b3-grid/summary.json")
    ap.add_argument("--output-dir", default="results/exp001b-matrix")
    ap.add_argument("--skip-resource-curve", action="store_true")
    args = ap.parse_args()

    if args.device != "cpu":
        raise SystemExit("EXP-001B is CPU-only; see results/exp001b-device-probe/")

    cadence_k, slow_lr = load_frozen_b3(ROOT / args.b3_summary)
    out_dir = ROOT / args.output_dir
    started = time.time()
    print(f"[matrix] B3 frozen at k={cadence_k}, slow_lr={slow_lr:g}", flush=True)

    runs: dict[str, dict] = {}

    # ---- B0/B1/B2/B3, then B5, then B4 (issue #5 section 5 run order) ----
    for arm in BASELINE_ORDER:
        for seed in DEV_SEEDS:
            print(f"[matrix] {arm} seed={seed}", flush=True)
            t0 = time.time()
            r = run_arm(
                arm=arm, seed=seed, cfg=make_cfg(slow_lr=slow_lr),
                device=args.device,
                cadence_k=cadence_k if arm == "b3_fixed" else None,
            )
            write(out_dir, f"{arm}-seed{seed}.json", r)
            runs[f"{arm}:{seed}"] = r
            brief(arm, r, t0)

    for seed in DEV_SEEDS:
        print(f"[matrix] b5_promotion seed={seed}", flush=True)
        t0 = time.time()
        r = run_arm(arm="b5_promotion", seed=seed, cfg=make_cfg(slow_lr=slow_lr),
                    device=args.device)
        write(out_dir, f"b5_promotion-seed{seed}.json", r)
        runs[f"b5_promotion:{seed}"] = r
        brief("b5_promotion", r, t0)

    for seed in DEV_SEEDS:
        b5 = runs[f"b5_promotion:{seed}"]
        commits, steps = match_b4_to_b5(b5, seed)
        print(f"[matrix] b4_random seed={seed} matched to B5 "
              f"commits={sorted(commits)} steps={steps}", flush=True)
        t0 = time.time()
        r = run_arm(arm="b4_random", seed=seed, cfg=make_cfg(slow_lr=slow_lr),
                    device=args.device, random_commit_segments=commits,
                    commit_steps=steps)
        # Matching is asserted, not assumed.
        b5_accepted = [c for c in b5["commit_log"] if c.get("accepted")]
        b4_accepted = [c for c in r["commit_log"] if c.get("accepted")]
        if len(b4_accepted) != len(b5_accepted):
            raise AssertionError(
                f"seed {seed}: B4 commit count {len(b4_accepted)} != B5 {len(b5_accepted)}"
            )
        if sorted(c["planned_steps"] for c in b4_accepted) != \
                sorted(c["planned_steps"] for c in b5_accepted):
            raise AssertionError(f"seed {seed}: B4 per-commit allocation does not match B5")
        if sorted(c["write_units"] for c in b4_accepted) != \
                sorted(c["write_units"] for c in b5_accepted):
            raise AssertionError(f"seed {seed}: B4 per-commit write units do not match B5")
        r["matched_to_b5"] = {
            "b5_accepted_segments": [c["segment"] for c in b5_accepted],
            "b4_commit_segments": sorted(commits),
            "per_commit_steps": {str(k): v for k, v in steps.items()},
        }
        write(out_dir, f"b4_random-seed{seed}.json", r)
        runs[f"b4_random:{seed}"] = r
        brief("b4_random", r, t0)

    # ---- resource curve: predeclared batch points, reported whole ----
    curve: list[dict] = []
    if not args.skip_resource_curve:
        for batch in BATCH_POINTS:
            if batch == 1:
                continue  # already run above
            for seed in DEV_SEEDS:
                for arm in ("b2_replay", "b3_fixed"):
                    print(f"[curve] {arm} batch={batch} seed={seed}", flush=True)
                    t0 = time.time()
                    cfg = make_cfg(
                        slow_lr=slow_lr,
                        consolidation_batch=batch if arm == "b3_fixed" else 1,
                        replay_per_online_step=batch if arm == "b2_replay" else 1,
                    )
                    r = run_arm(arm=arm, seed=seed, cfg=cfg, device=args.device,
                                cadence_k=cadence_k if arm == "b3_fixed" else None)
                    r["batch_point"] = batch
                    write(out_dir, f"curve-{arm}-batch{batch}-seed{seed}.json", r)
                    curve.append(r)
                    brief(f"{arm}@b{batch}", r, t0)

    def agg(arm: str, key: str) -> float:
        vals = [runs[f"{arm}:{s}"][key] for s in DEV_SEEDS]
        return sum(vals) / len(vals)

    arms = [*BASELINE_ORDER, "b5_promotion", "b4_random"]
    summary = {
        "protocol": "EXP-001B-development-arm-matrix-v1",
        "status": "DEVELOPMENT_ONLY",
        "note": "Not confirmatory. Not a paper claim. Does not authorize changing H1.",
        "development_seeds": list(DEV_SEEDS),
        "b3_frozen": {"cadence_k": cadence_k, "slow_lr": slow_lr},
        "online_lr": ONLINE_LR,
        "promotion_thresholds": {
            "promotion_min_fast_gain": 0.08,
            "promotion_min_current_acc": 0.45,
            "promotion_max_retention_drop": 0.03,
        },
        "arms": {
            arm: {
                "mean_diagonal": agg(arm, "mean_diagonal"),
                "mean_average_forgetting": agg(arm, "average_forgetting"),
                "mean_retention_auc": agg(arm, "retention_auc"),
                "mean_final_average": agg(arm, "final_average"),
                "per_seed_diagonal": [runs[f"{arm}:{s}"]["mean_diagonal"] for s in DEV_SEEDS],
                "per_seed_forgetting": [runs[f"{arm}:{s}"]["average_forgetting"] for s in DEV_SEEDS],
                "commit_counts": [len(runs[f"{arm}:{s}"]["accepted_commit_segments"]) for s in DEV_SEEDS],
                "adaptation": {
                    "parameter_write_units": [runs[f"{arm}:{s}"]["budget"]["parameter_write_units"] for s in DEV_SEEDS],
                    "optimizer_steps": [runs[f"{arm}:{s}"]["budget"]["optimizer_steps"] for s in DEV_SEEDS],
                    "training_examples_processed": [runs[f"{arm}:{s}"]["budget"]["training_examples_processed"] for s in DEV_SEEDS],
                    "tokens_processed": [runs[f"{arm}:{s}"]["budget"]["tokens_processed"] for s in DEV_SEEDS],
                    "replay_examples_used": [runs[f"{arm}:{s}"]["budget"]["replay_examples_used"] for s in DEV_SEEDS],
                    "wall_seconds": [runs[f"{arm}:{s}"]["elapsed_seconds"] for s in DEV_SEEDS],
                },
                "decision": {
                    "decision_forward_calls": [runs[f"{arm}:{s}"]["budget"]["decision_forward_calls"] for s in DEV_SEEDS],
                    "decision_tokens_processed": [runs[f"{arm}:{s}"]["budget"]["decision_tokens_processed"] for s in DEV_SEEDS],
                },
            }
            for arm in arms
        },
        "resource_curve": [
            {
                "arm": r["arm"], "batch_point": r["batch_point"], "seed": r["seed"],
                "mean_diagonal": r["mean_diagonal"],
                "average_forgetting": r["average_forgetting"],
                "parameter_write_units": r["budget"]["parameter_write_units"],
                "write_budget_units": r["budget"]["write_budget_units"],
                "optimizer_steps": r["budget"]["optimizer_steps"],
                "training_examples_processed": r["budget"]["training_examples_processed"],
                "tokens_processed": r["budget"]["tokens_processed"],
                "wall_seconds": r["elapsed_seconds"],
            }
            for r in curve
        ],
        "model_revision": MODEL_REVISION,
        "git_sha": git_sha(),
        "elapsed_seconds": time.time() - started,
    }
    write(out_dir, "summary.json", summary)
    print(json.dumps({a: {k: round(v, 4) for k, v in summary["arms"][a].items()
                          if isinstance(v, float)} for a in arms}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
