#!/usr/bin/env python3
"""EXP-001 Phase B Stage B1: strengthen B3 before B5 is ever run.

Issue #5 section 4. This deliberately selects a *strong* fixed baseline using
the same stability/plasticity criterion B5 will later have to beat, so that we
cannot choose a weak comparator after seeing State Promotion.

The grid and the selection rule are predeclared here as module constants and are
evaluated by code, not by hand, before any B5 result exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp001b_arm import MODEL_REVISION, ONLINE_LR, git_sha, run_arm

from state_promotion.lm import LMExperimentConfig

DEV_SEEDS = (20260901, 20260902, 20260903)
CADENCES = (1, 2, 3)
SLOW_LRS = (1e-3, 2e-3, 3e-3)

ELIGIBILITY_DIAGONAL_FRACTION = 0.95


def cell_key(k: int, lr: float) -> str:
    return f"k{k}@lr{lr:.0e}"


def select(cells: list[dict]) -> dict | None:
    """Predeclared B3 selection rule (issue #5 section 4).

    Eligibility is defined against the best mean diagonal actually observed, so
    a cell may not buy low forgetting by failing to learn.
    """
    valid = [c for c in cells if c["valid"]]
    if not valid:
        return None
    dmax = max(c["mean_diagonal"] for c in valid)
    eligible = [
        c for c in valid
        if c["mean_diagonal"] >= ELIGIBILITY_DIAGONAL_FRACTION * dmax
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda c: (
            c["mean_average_forgetting"],      # 1. lowest forgetting
            -c["mean_retention_auc"],          # 2. higher retention AUC
            -c["mean_final_average"],          # 3. higher final average
            c["slow_lr"],                      # 4. lower slow LR
            c["cadence_k"],                    # 5. smaller cadence
        )
    )
    return eligible[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-dir", default="results/exp001b-b3-grid")
    args = ap.parse_args()

    if args.device != "cpu":
        raise SystemExit("EXP-001B is CPU-only; see results/exp001b-device-probe/")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runs: list[dict] = []

    for k in CADENCES:
        for lr in SLOW_LRS:
            for seed in DEV_SEEDS:
                print(f"[B3 grid] k={k} slow_lr={lr:g} seed={seed}", flush=True)
                cfg = LMExperimentConfig(
                    model_revision=MODEL_REVISION, online_lr=ONLINE_LR
                )
                cfg.consolidation_lr = lr
                t0 = time.time()
                try:
                    r = run_arm(
                        arm="b3_fixed", seed=seed, cfg=cfg,
                        device=args.device, cadence_k=k,
                    )
                    r["cell_valid"] = True
                    r["invalidation_reasons"] = []
                except Exception as exc:  # noqa: BLE001 - a failed cell is archived, not discarded
                    r = {
                        "arm": "b3_fixed", "seed": seed, "cadence_k": k,
                        "consolidation_lr": lr, "cell_valid": False,
                        "invalidation_reasons": [f"{type(exc).__name__}: {exc}"],
                    }
                r["slow_lr"] = lr
                path = out_dir / f"seed-{seed}-k{k}-lr{lr:.0e}.json"
                path.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
                runs.append(r)
                print(
                    f"    diag={r.get('mean_diagonal', float('nan')):.3f} "
                    f"forget={r.get('average_forgetting', float('nan')):.3f} "
                    f"auc={r.get('retention_auc', float('nan')):.3f} "
                    f"commits={len(r.get('accepted_commit_segments', []))} "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )

    cells: list[dict] = []
    for k in CADENCES:
        for lr in SLOW_LRS:
            rows = [r for r in runs if r.get("cadence_k") == k and r.get("slow_lr") == lr]
            ok = [r for r in rows if r.get("cell_valid")]
            reasons: list[str] = []
            if len(rows) != len(DEV_SEEDS):
                reasons.append("missing_seed")
            if len(ok) != len(rows):
                reasons.append("run_failure")
            for r in ok:
                if not r.get("backbone_frozen"):
                    reasons.append(f"backbone_not_frozen:{r['seed']}")
                if not r.get("finite_matrix"):
                    reasons.append(f"non_finite_scores:{r['seed']}")
                b = r.get("budget", {})
                if b.get("write_budget_units") is not None and \
                        b.get("parameter_write_units", 0) > b["write_budget_units"]:
                    reasons.append(f"write_budget_exceeded:{r['seed']}")
                prov = r.get("provenance", {})
                if not (prov.get("tokenizer_asset_sha256") or {}):
                    reasons.append(f"tokenizer_provenance_unverified:{r['seed']}")
                if prov.get("resolved_snapshot_commit") != MODEL_REVISION:
                    reasons.append(f"snapshot_pin_mismatch:{r['seed']}")

            def mean(field: str, _rows: list[dict] = ok) -> float:
                vals = [float(r[field]) for r in _rows if field in r]
                return sum(vals) / len(vals) if vals else float("nan")

            cells.append({
                "cell": cell_key(k, lr),
                "cadence_k": k,
                "slow_lr": lr,
                "seeds": [r["seed"] for r in rows],
                "per_seed_diagonal": [r.get("mean_diagonal") for r in ok],
                "per_seed_forgetting": [r.get("average_forgetting") for r in ok],
                "mean_diagonal": mean("mean_diagonal"),
                "mean_average_forgetting": mean("average_forgetting"),
                "mean_retention_auc": mean("retention_auc"),
                "mean_final_average": mean("final_average"),
                "commit_counts": [len(r.get("accepted_commit_segments", [])) for r in ok],
                "valid": not reasons,
                "invalidation_reasons": reasons,
            })

    selected = select(cells)
    dmax = max((c["mean_diagonal"] for c in cells if c["valid"]), default=float("nan"))
    summary = {
        "protocol": "EXP-001B-b3-strengthening-grid-v1",
        "status": "DEVELOPMENT_ONLY",
        "note": "Selects a strong fixed baseline BEFORE B5 is run or inspected.",
        "development_seeds": list(DEV_SEEDS),
        "cadences": list(CADENCES),
        "slow_lrs": list(SLOW_LRS),
        "online_lr": ONLINE_LR,
        "eligibility_rule": (
            f"mean diagonal >= {ELIGIBILITY_DIAGONAL_FRACTION} * Dmax, finite, "
            "provenance valid, no resource invalidation"
        ),
        "selection_rule": (
            "lowest mean average forgetting; ties: higher retention AUC, then higher "
            "final average, then lower slow LR, then smaller cadence k"
        ),
        "dmax": dmax,
        "cells": cells,
        "selected": selected,
        "b5_allowed": selected is not None,
        "model_revision": MODEL_REVISION,
        "git_sha": git_sha(),
        "elapsed_seconds": time.time() - started,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected": selected, "b5_allowed": summary["b5_allowed"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
