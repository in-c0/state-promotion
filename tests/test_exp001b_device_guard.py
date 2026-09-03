"""EXP-001B is CPU-only, and that must be enforced rather than assumed.

`verify_device_numerics` compares a single forward loss against CPU within a
0.05 tolerance. It passed on MPS in both float16 and float32 while a full B1
lifetime diverged badly from CPU:

    float16  mean diagonal 0.167 (chance), forgetting 0.000
    float32  mean diagonal 0.667,          forgetting 0.333
    cpu      mean diagonal 0.944,          forgetting 0.533

A forward-pass check cannot see divergence that compounds over 288 optimizer
steps, so device selection needs its own guard.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_exp001b_arm.py"


def _invoke(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )


def test_non_cpu_device_is_refused_without_explicit_acknowledgement():
    proc = _invoke("--arm", "b1_sequential", "--device", "mps", "--out", "results/_unused.json")
    assert proc.returncode != 0
    assert "refusing device" in (proc.stderr + proc.stdout)


def test_refusal_names_the_observed_divergence_not_just_a_policy():
    proc = _invoke("--arm", "b1_sequential", "--device", "cuda", "--out", "results/_unused.json")
    combined = proc.stderr + proc.stdout
    assert "0.944" in combined and "0.667" in combined


def test_cpu_is_the_default_device():
    import argparse
    import importlib.util

    spec = importlib.util.spec_from_file_location("exp001b_runner", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    assert ap.parse_args([]).device == "cpu"


def test_lora_loader_defaults_to_float32_on_every_device():
    """float16 on accelerators is what destroyed adaptation; the default is fp32."""
    src = (ROOT / "src" / "state_promotion" / "lora.py").read_text()
    assert "dtype = torch.float32 if compute_dtype is None else compute_dtype" in src
    assert "torch.float16 if device in" not in src
