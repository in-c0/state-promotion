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
