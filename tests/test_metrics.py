from state_promotion.metrics import summarize


def test_forgetting_matrix():
    r = [
        [0.8, float("nan"), float("nan")],
        [0.7, 0.9, float("nan")],
        [0.6, 0.8, 0.85],
    ]
    m = summarize(r)
    assert abs(m.final_average - 0.75) < 1e-9
    assert abs(m.average_forgetting - 0.15) < 1e-9
