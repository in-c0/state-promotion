#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_promotion.pals import generate_retention_stream, generate_revision_stream, write_jsonl  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--out-dir", type=Path, default=ROOT / "data")
    args = p.parse_args()

    retention, retention_codes = generate_retention_stream(args.seed)
    revision, revision_codes = generate_revision_stream(args.seed)
    write_jsonl(args.out_dir / f"pals-retention-{args.seed}.jsonl", retention)
    write_jsonl(args.out_dir / f"pals-revision-{args.seed}.jsonl", revision)
    (args.out_dir / f"pals-codes-{args.seed}.txt").write_text(
        "retention=" + ",".join(retention_codes) + "\nrevision=" + ",".join(revision_codes) + "\n"
    )
    print(f"retention examples: {len(retention)}")
    print(f"revision examples: {len(revision)}")


if __name__ == "__main__":
    main()
