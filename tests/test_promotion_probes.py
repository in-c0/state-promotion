import importlib.util
from pathlib import Path

from state_promotion.pals import generate_retention_stream, generate_revision_stream

SPEC = importlib.util.spec_from_file_location("run_lm_pals", Path(__file__).parents[1] / "scripts" / "run_lm_pals.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_retention_promotion_probes_use_only_observed_train_examples():
    xs, _ = generate_retention_stream(7)
    grouped = MOD.group(xs)
    replay = list(grouped[0]["train"]) + list(grouped[1]["train"])
    current, protected = MOD.build_promotion_probes(
        stream="retention",
        segment=1,
        segment_train=list(grouped[1]["train"]),
        replay_items=replay,
        all_examples=xs,
    )
    assert current
    assert protected
    assert all(x.split == "train" for x in current + protected)
    assert all(x.segment == 1 for x in current)
    assert all(x.segment < 1 for x in protected)


def test_revision_protection_drops_superseded_replay_target():
    xs, _ = generate_revision_stream(11)
    grouped = MOD.group(xs)
    replay = list(grouped[0]["train"]) + list(grouped[1]["train"]) + list(grouped[2]["train"])
    _, protected = MOD.build_promotion_probes(
        stream="revision",
        segment=2,
        segment_train=list(grouped[2]["train"]),
        replay_items=replay,
        all_examples=xs,
    )
    active = MOD.active_train_targets(xs, 2)
    assert protected
    assert all(x.split == "train" for x in protected)
    assert all(active[(x.context, x.key)] == x.target for x in protected)


def test_probe_hash_is_deterministic():
    xs, _ = generate_retention_stream(3)
    grouped = MOD.group(xs)
    probe = MOD.dedupe_probe_examples(list(grouped[0]["train"]))
    assert MOD.probe_hash(probe) == MOD.probe_hash(probe)
