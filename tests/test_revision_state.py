import importlib.util
from pathlib import Path

from state_promotion.pals import generate_revision_stream

SPEC = importlib.util.spec_from_file_location("run_lm_pals", Path(__file__).parents[1] / "scripts" / "run_lm_pals.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_active_revision_replaces_superseded_pair():
    xs, _ = generate_revision_stream(11)
    before, stale_before = MOD.active_revision_tests(xs, 1)
    after, stale_after = MOD.active_revision_tests(xs, 2)
    assert not stale_before
    assert stale_after
    latest = {(x.context, x.key): x.target for x in after}
    for pair, stale in stale_after.items():
        assert latest[pair] != stale
    assert len(after) <= len(before)
