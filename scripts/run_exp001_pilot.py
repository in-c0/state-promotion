#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ["frozen", "sequential", "replay", "fixed", "promotion"]


def run_cmd(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def result_path(out_dir: Path, stream: str, method: str, seed: int) -> Path:
    return out_dir / f"lm-{stream}-{method}-{seed}.json"


def resolve_model_revision(model: str, requested_revision: str | None = None) -> str:
    """Resolve one immutable HF snapshot before any arm starts."""
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": model,
        "revision": requested_revision or "main",
        "allow_patterns": ["config.json"],
    }
    try:
        snapshot = Path(snapshot_download(**kwargs))
    except Exception:
        # A fully cached host should remain usable if Hugging Face is temporarily
        # unreachable. The cached ref still resolves to an immutable snapshot dir.
        snapshot = Path(snapshot_download(**kwargs, local_files_only=True))
    revision = snapshot.name
    if len(revision) < 12:
        raise RuntimeError(f"Could not resolve immutable model snapshot from {snapshot}")
    return revision


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run the EXP-001 engineering pilot, derive a count-matched random control, and validate manifests."
    )
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--model-revision", default=None, help="Optional HF ref/SHA; resolved to one immutable snapshot before the first arm.")
    p.add_argument("--stream", choices=["retention", "revision"], default="retention")
    p.add_argument("--eval-cap", type=int, default=4,
                   help="Pilot evaluation cap. Use a small value for the first engineering run.")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "exp001-pilot")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    resolved_revision = resolve_model_revision(args.model, args.model_revision)
    print(f"Pinned model snapshot: {args.model}@{resolved_revision}", flush=True)
    common = [
        sys.executable, str(ROOT / "scripts" / "run_lm_pals.py"),
        "--seed", str(args.seed),
        "--model", args.model,
        "--model-revision", resolved_revision,
        "--stream", args.stream,
        "--eval-cap", str(args.eval_cap),
    ]

    for method in PRIMARY:
        out = result_path(args.out_dir, args.stream, method, args.seed)
        run_cmd(common + ["--method", method, "--out", str(out)])

    promotion_path = result_path(args.out_dir, args.stream, "promotion", args.seed)
    promotion = json.loads(promotion_path.read_text())
    accepted = promotion.get("accepted_commit_segments", [])
    n_segments = len(promotion.get("promotion_log", []))
    if n_segments == 0 and promotion.get("matrix"):
        n_segments = len(promotion["matrix"])
    if n_segments == 0:
        raise RuntimeError("Could not infer segment count from promotion result")

    rng = random.Random(args.seed + 90421)
    k = min(len(accepted), n_segments)
    random_segments = sorted(rng.sample(range(n_segments), k)) if k else []
    random_out = result_path(args.out_dir, args.stream, "random", args.seed)
    run_cmd(common + [
        "--method", "random",
        "--random-commit-segments", ",".join(map(str, random_segments)),
        "--out", str(random_out),
    ])

    validation_paths = [result_path(args.out_dir, args.stream, m, args.seed) for m in ["frozen", "sequential", "replay", "fixed", "promotion"]]
    validation_path = args.out_dir / f"validation-{args.stream}-{args.seed}.json"
    validator = [
        sys.executable, str(ROOT / "scripts" / "validate_runs.py"),
        *map(str, validation_paths), str(random_out),
        "--out", str(validation_path),
    ]
    validation_rc = subprocess.run(validator, cwd=ROOT).returncode

    manifest = {
        "classification": "ENGINEERING_PILOT",
        "seed": args.seed,
        "model": args.model,
        "model_revision": resolved_revision,
        "stream": args.stream,
        "eval_cap": args.eval_cap,
        "promotion_accepted_segments": accepted,
        "matched_random_segments": random_segments,
        "validator_return_code": validation_rc,
        "validator_passed": validation_rc == 0,
        "note": "A validator failure is a diagnostic pilot outcome, not an experiment failure and not evidence for H1.",
    }
    manifest_path = args.out_dir / f"pilot-manifest-{args.stream}-{args.seed}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
