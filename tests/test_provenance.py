"""Provenance conflicts must fail loudly, never degrade to a null field.

Amendment F pins one immutable snapshot. Before this module the tokenizer half
of that pin was recorded as `loaded_tokenizer_revision: null`, because
Transformers 5.x does not reliably populate `tokenizer.init_kwargs._commit_hash`.
An unverified pin that looks like a recorded pin is the failure mode these tests
exist to prevent.
"""
from __future__ import annotations

import hashlib

import pytest

from state_promotion.provenance import (
    TOKENIZER_ASSET_NAMES,
    ProvenanceError,
    assert_arms_share_snapshot,
    assert_provenance_consistent,
    sha256_file,
    tokenizer_asset_digests,
)

PIN = "7ae557604adf67be50417f59c2c2f167def9a775"


def _record(**overrides) -> dict:
    base = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "requested_revision": PIN,
        "resolved_snapshot_commit": PIN,
        "tokenizer_asset_sha256": {"tokenizer.json": "a" * 64, "vocab.json": "b" * 64},
        "reported_model_revision": None,
        "reported_tokenizer_revision": None,
    }
    base.update(overrides)
    return base


def test_consistent_record_passes():
    assert_provenance_consistent(_record())


def test_resolved_snapshot_disagreeing_with_pin_fails():
    with pytest.raises(ProvenanceError, match="snapshot pin conflict"):
        assert_provenance_consistent(_record(resolved_snapshot_commit="0" * 40))


def test_missing_requested_revision_fails():
    with pytest.raises(ProvenanceError, match="no requested_revision"):
        assert_provenance_consistent(_record(requested_revision=""))


def test_missing_resolved_commit_fails():
    with pytest.raises(ProvenanceError, match="no resolved_snapshot_commit"):
        assert_provenance_consistent(_record(resolved_snapshot_commit=None))


def test_conflicting_reported_model_revision_fails():
    with pytest.raises(ProvenanceError, match="reported_model_revision"):
        assert_provenance_consistent(_record(reported_model_revision="0" * 40))


def test_conflicting_reported_tokenizer_revision_fails():
    with pytest.raises(ProvenanceError, match="reported_tokenizer_revision"):
        assert_provenance_consistent(_record(reported_tokenizer_revision="0" * 40))


def test_absent_reported_revision_is_tolerated_because_assets_are_hashed():
    """None is absent, not conflicting -- that is why asset digests exist."""
    assert_provenance_consistent(
        _record(reported_model_revision=None, reported_tokenizer_revision=None)
    )


def test_zero_hashed_tokenizer_assets_fails_instead_of_silently_passing():
    """The exact regression: an unverified tokenizer pin must not look verified."""
    with pytest.raises(ProvenanceError, match="unverified"):
        assert_provenance_consistent(_record(tokenizer_asset_sha256={}))


def test_changed_tokenizer_asset_digest_fails():
    expected = {"tokenizer.json": "a" * 64, "vocab.json": "b" * 64}
    tampered = {"tokenizer.json": "a" * 64, "vocab.json": "c" * 64}
    with pytest.raises(ProvenanceError, match="vocab.json"):
        assert_provenance_consistent(
            _record(tokenizer_asset_sha256=tampered), expected_assets=expected
        )


def test_added_or_removed_tokenizer_asset_fails():
    expected = {"tokenizer.json": "a" * 64, "vocab.json": "b" * 64}
    fewer = {"tokenizer.json": "a" * 64}
    with pytest.raises(ProvenanceError, match="asset set changed"):
        assert_provenance_consistent(
            _record(tokenizer_asset_sha256=fewer), expected_assets=expected
        )


def test_arms_on_different_snapshots_fail():
    a = _record()
    b = _record(resolved_snapshot_commit="0" * 40, requested_revision="0" * 40)
    with pytest.raises(ProvenanceError, match="different snapshots"):
        assert_arms_share_snapshot([a, b])


def test_arms_with_different_tokenizer_assets_fail():
    a = _record()
    b = _record(tokenizer_asset_sha256={"tokenizer.json": "z" * 64, "vocab.json": "b" * 64})
    with pytest.raises(ProvenanceError, match="different tokenizer assets"):
        assert_arms_share_snapshot([a, b])


def test_arms_sharing_one_snapshot_pass():
    assert_arms_share_snapshot([_record(), _record()])


def test_empty_arm_list_fails():
    with pytest.raises(ProvenanceError):
        assert_arms_share_snapshot([])


def test_asset_digests_hash_real_file_contents(tmp_path):
    (tmp_path / "tokenizer.json").write_bytes(b'{"model":"bpe"}')
    (tmp_path / "vocab.json").write_bytes(b"{}")
    (tmp_path / "unrelated.bin").write_bytes(b"ignored")
    digests = tokenizer_asset_digests(tmp_path)
    assert set(digests) == {"tokenizer.json", "vocab.json"}
    assert digests["tokenizer.json"] == hashlib.sha256(b'{"model":"bpe"}').hexdigest()
    assert "unrelated.bin" not in digests


def test_asset_digest_changes_when_file_changes(tmp_path):
    p = tmp_path / "tokenizer.json"
    p.write_bytes(b"one")
    before = sha256_file(p)
    p.write_bytes(b"two")
    assert sha256_file(p) != before


def test_asset_name_list_covers_the_known_tokenizer_formats():
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "special_tokens_map.json", "added_tokens.json", "tokenizer.model"):
        assert name in TOKENIZER_ASSET_NAMES
