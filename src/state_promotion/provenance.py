"""Snapshot provenance for EXP-001 / EXP-001R runs.

Amendment F pins one immutable Hugging Face snapshot for every arm. The pin was
only half-verifiable in practice: ``tokenizer.init_kwargs["_commit_hash"]`` is
not populated reliably by Transformers 5.x, so run manifests recorded
``loaded_tokenizer_revision: null`` and the tokenizer half of the pin went
unverified.

This module records provenance from the resolved snapshot directory instead of
from optional attributes on the loaded objects, and hashes the tokenizer assets
actually present in that snapshot. Conflicts raise; they never degrade to
``None``.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path

# Tokenizer-defining assets across the formats this project may encounter.
# Absence of an individual name is normal (Qwen2.5 folds special tokens into
# tokenizer_config.json); absence of *all* of them is a provenance failure.
TOKENIZER_ASSET_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "spiece.model",
)

_TOKENIZER_PATTERNS = [*TOKENIZER_ASSET_NAMES, "config.json"]


class ProvenanceError(RuntimeError):
    """A snapshot pin could not be verified. Never downgrade this to a warning."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_snapshot_dir(model_name: str, revision: str) -> Path:
    """Resolve the immutable snapshot directory for an exact revision.

    Falls back to the local cache so a fully cached host stays usable when
    Hugging Face is unreachable, matching run_exp001_pilot.resolve_model_revision.
    """
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": model_name,
        "revision": revision,
        "allow_patterns": _TOKENIZER_PATTERNS,
    }
    try:
        snapshot = Path(snapshot_download(**kwargs))
    except Exception:  # noqa: BLE001 - a fully cached host stays usable when HF is unreachable
        snapshot = Path(snapshot_download(**kwargs, local_files_only=True))
    return snapshot


def tokenizer_asset_digests(snapshot_dir: Path) -> dict[str, str]:
    """SHA-256 of every tokenizer-defining asset present in the snapshot."""
    digests: dict[str, str] = {}
    for name in TOKENIZER_ASSET_NAMES:
        path = snapshot_dir / name
        if path.is_file():
            digests[name] = sha256_file(path)
    return digests


def code_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - provenance must degrade to "unknown", never crash a run
        return "unknown"


def collect_provenance(
    *,
    model_name: str,
    requested_revision: str,
    model=None,
    tokenizer=None,
    device: str = "cpu",
    root: Path | None = None,
) -> dict:
    """Build a provenance record and verify the snapshot pin.

    Raises ProvenanceError rather than recording an unverified pin.
    """
    import torch
    import transformers

    if not requested_revision:
        raise ProvenanceError("no model revision requested; Amendment F requires an explicit pin")

    snapshot_dir = resolve_snapshot_dir(model_name, requested_revision)
    resolved_commit = snapshot_dir.name
    digests = tokenizer_asset_digests(snapshot_dir)

    record = {
        "model_name": model_name,
        "requested_revision": requested_revision,
        "resolved_snapshot_commit": resolved_commit,
        "snapshot_dir": str(snapshot_dir),
        "tokenizer_asset_sha256": digests,
        "tokenizer_asset_count": len(digests),
        "reported_model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
        "reported_tokenizer_revision": (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash"),
        "tokenizer_class": type(tokenizer).__name__ if tokenizer is not None else None,
        "tokenizer_vocab_size": len(tokenizer) if tokenizer is not None else None,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "code_sha": code_sha(root or Path(__file__).resolve().parents[2]),
    }
    assert_provenance_consistent(record)
    return record


def assert_provenance_consistent(record: dict, *, expected_assets: dict[str, str] | None = None) -> None:
    """Fail loudly on any snapshot-pin conflict.

    A reported revision of None is absent, not conflicting, and is tolerated --
    that is precisely why the asset digests exist. A reported revision that
    *disagrees* with the pin is a hard failure.
    """
    requested = record.get("requested_revision")
    resolved = record.get("resolved_snapshot_commit")

    if not requested:
        raise ProvenanceError("provenance record has no requested_revision")
    if not resolved:
        raise ProvenanceError("provenance record has no resolved_snapshot_commit")
    if resolved != requested:
        raise ProvenanceError(
            f"snapshot pin conflict: requested {requested}, resolved {resolved}"
        )

    for field in ("reported_model_revision", "reported_tokenizer_revision"):
        reported = record.get(field)
        if reported is not None and reported != requested:
            raise ProvenanceError(
                f"{field} {reported} conflicts with pinned revision {requested}"
            )

    digests = record.get("tokenizer_asset_sha256") or {}
    if not digests:
        raise ProvenanceError(
            "no tokenizer assets hashed; the tokenizer half of the snapshot pin is unverified"
        )

    if expected_assets is not None:
        if set(digests) != set(expected_assets):
            raise ProvenanceError(
                f"tokenizer asset set changed: {sorted(digests)} != {sorted(expected_assets)}"
            )
        for name, digest in expected_assets.items():
            if digests[name] != digest:
                raise ProvenanceError(
                    f"tokenizer asset {name} sha256 {digests[name]} != expected {digest}"
                )


def assert_arms_share_snapshot(records: list[dict]) -> None:
    """Amendment F clause 4: every arm in a pilot must share one snapshot."""
    if not records:
        raise ProvenanceError("no provenance records to compare across arms")
    commits = {r.get("resolved_snapshot_commit") for r in records}
    if len(commits) != 1:
        raise ProvenanceError(f"arms resolved to different snapshots: {sorted(map(str, commits))}")
    assets = [tuple(sorted((r.get("tokenizer_asset_sha256") or {}).items())) for r in records]
    if len(set(assets)) != 1:
        raise ProvenanceError("arms loaded different tokenizer assets from the same snapshot")
