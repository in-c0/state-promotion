import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_exp001_pilot", ROOT / "scripts" / "run_exp001_pilot.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_resolve_model_revision_returns_snapshot_sha(monkeypatch):
    seen = {}
    def snapshot_download(**kwargs):
        seen.update(kwargs)
        return "/tmp/hf/models--Qwen/snapshots/0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    revision = MOD.resolve_model_revision("Qwen/Qwen2.5-0.5B-Instruct")
    assert revision == "0123456789abcdef0123456789abcdef01234567"
    assert seen["revision"] == "main"
    assert seen["allow_patterns"] == ["config.json"]


def test_resolve_model_revision_honors_requested_ref(monkeypatch):
    seen = {}
    def snapshot_download(**kwargs):
        seen.update(kwargs)
        return "/tmp/hf/snapshots/fedcba9876543210fedcba9876543210fedcba98"
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    revision = MOD.resolve_model_revision("Qwen/Qwen2.5-0.5B-Instruct", "candidate-ref")
    assert revision == "fedcba9876543210fedcba9876543210fedcba98"
    assert seen["revision"] == "candidate-ref"


def test_resolve_model_revision_falls_back_to_cached_snapshot(monkeypatch):
    calls = []
    def snapshot_download(**kwargs):
        calls.append(kwargs)
        if not kwargs.get("local_files_only"):
            raise OSError("offline")
        return "/tmp/hf/snapshots/1111111111111111111111111111111111111111"
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    revision = MOD.resolve_model_revision("Qwen/Qwen2.5-0.5B-Instruct")
    assert revision == "1111111111111111111111111111111111111111"
    assert calls[-1]["local_files_only"] is True
