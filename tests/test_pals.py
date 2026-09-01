from collections import defaultdict

from state_promotion.pals import generate_retention_stream, generate_revision_stream


def test_retention_stream_has_context_conflicts_without_duplicate_pair_targets():
    xs, codes = generate_retention_stream(7)
    train = [x for x in xs if x.split == "train"]
    pair_targets = defaultdict(set)
    key_targets = defaultdict(set)
    for x in train:
        pair_targets[(x.context, x.key)].add(x.target)
        key_targets[x.key].add(x.target)
    assert all(len(v) == 1 for v in pair_targets.values())
    assert any(len(v) > 1 for v in key_targets.values())
    assert len(codes) == 6


def test_revision_stream_contains_real_supersession():
    xs, _ = generate_revision_stream(11)
    old = {(x.context, x.key): x.target for x in xs if x.segment == 0 and x.split == "test"}
    revised = {(x.context, x.key): x.target for x in xs if x.segment == 2 and x.split == "test"}
    assert revised
    assert all(pair in old for pair in revised)
    assert all(old[pair] != target for pair, target in revised.items())


def test_prompt_does_not_expose_hidden_task_metadata():
    xs, _ = generate_revision_stream(19)
    ex = xs[0]
    prompt = ex.prompt.lower()
    assert "segment" not in prompt
    assert ex.stream.lower() not in prompt
    assert ex.relation.lower() not in prompt
    assert str(ex.version) not in prompt
    assert ex.split.lower() not in prompt
