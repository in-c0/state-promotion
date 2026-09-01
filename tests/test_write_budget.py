import torch
from torch import nn

from state_promotion.lm import BudgetCounter


def test_write_budget_counts_parameter_elements_and_caps():
    p = nn.Parameter(torch.zeros(5))
    b = BudgetCounter(write_budget_units=10)
    assert b.may_write([p])
    b.record_step([p])
    assert b.optimizer_steps == 1
    assert b.parameter_write_units == 5
    assert b.may_write([p])
    b.record_step([p])
    assert b.parameter_write_units == 10
    assert not b.may_write([p])


def test_decision_compute_is_accounted_separately():
    b = BudgetCounter()
    b.record_compute(tokens=100, examples=2, replay_examples=1)
    b.record_decision_compute(tokens=30)
    b.record_decision_compute(tokens=12, forward_calls=2)
    assert b.tokens_processed == 100
    assert b.decision_tokens_processed == 42
    assert b.decision_forward_calls == 3
