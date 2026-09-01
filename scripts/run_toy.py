#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_promotion.toy import run_method  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(6)))
    p.add_argument("--out", type=Path, default=ROOT / "results" / "toy-pilot.json")
    args = p.parse_args()

    methods = ["sequential", "replay", "fixed", "promotion"]
    rows = []
    for seed in args.seeds:
        for method in methods:
            r = run_method(method, seed)
            rows.append({
                "method": method,
                "seed": seed,
                "final_average": r.metrics.final_average,
                "average_forgetting": r.metrics.average_forgetting,
                "average_plasticity": r.metrics.average_plasticity,
                "retention_auc": r.metrics.retention_auc,
                "consolidations": r.consolidations,
                "rejected_consolidations": r.rejected_consolidations,
                "optimizer_steps": r.optimizer_steps,
                "score_matrix": r.score_matrix,
            })

    summary = {}
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        summary[method] = {}
        for metric in ["final_average", "average_forgetting", "average_plasticity", "retention_auc"]:
            vals = [float(r[metric]) for r in subset]
            summary[method][metric] = {
                "mean": statistics.fmean(vals),
                "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }

    payload = {
        "warning": "EXP-000 engineering smoke test only; not confirmatory paper evidence.",
        "seeds": args.seeds,
        "rows": rows,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=True))

    print(payload["warning"])
    print("method       final_avg     forgetting    plasticity    retention_auc")
    for method in methods:
        s = summary[method]
        print(
            f"{method:11s}  "
            f"{s['final_average']['mean']:.3f}±{s['final_average']['stdev']:.3f}   "
            f"{s['average_forgetting']['mean']:.3f}±{s['average_forgetting']['stdev']:.3f}   "
            f"{s['average_plasticity']['mean']:.3f}±{s['average_plasticity']['stdev']:.3f}   "
            f"{s['retention_auc']['mean']:.3f}±{s['retention_auc']['stdev']:.3f}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
