from state_promotion.toy import ToyConfig, run_method


def test_toy_harness_runs_and_shapes_matrix():
    cfg = ToyConfig(tasks=3, train_per_task=12, test_per_task=20, replay_capacity=16, replay_batch=2)
    r = run_method("promotion", seed=1, cfg=cfg)
    assert len(r.score_matrix) == 3
    assert len(r.score_matrix[-1]) == 3
    assert r.optimizer_steps == 36
