#!/usr/bin/env python3
"""Preflight the EXP-001 macOS execution environment without mutating system Python.

This script is intentionally stdlib-only so it can run before project dependencies
are installed. It prints a machine-readable JSON report Claude/local operators can
use to decide whether to create a venv and run the pilot.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> dict:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, check=False)
        return {"rc": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"rc": -1, "stdout": "", "stderr": repr(exc)}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    usage = shutil.disk_usage(root)
    payload = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "python_executable": sys.executable,
        "python_ok": sys.version_info >= (3, 11),
        "repo": str(root),
        "disk_free_gib": round(usage.free / (1024**3), 2),
        "git": run(["git", "--version"]),
        "xcode_select": run(["xcode-select", "-p"]),
        "torch_probe": run([
            sys.executable,
            "-c",
            "import json,torch; print(json.dumps({'torch':torch.__version__,'mps_built':torch.backends.mps.is_built(),'mps_available':torch.backends.mps.is_available()}))",
        ]),
        "transformers_probe": run([
            sys.executable,
            "-c",
            "import transformers; print(transformers.__version__)",
        ]),
        "hf_cache_candidates": [
            str(Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct"),
            str(Path.home() / ".cache" / "huggingface" / "hub"),
        ],
    }
    print(json.dumps(payload, indent=2))
    if not payload["python_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
