"""Structural tests for the EXP-001 v3 embedding-space latent (issue #7 section 1).

The v1/v2 port EMAed `out.hidden_states[-1]` into a buffer that was injected as
an input-embedding prefix, with nothing reconciling the two spaces; the latent
reached ~445x the median token-embedding norm. These tests pin the repaired
contract structurally, so the defect cannot silently return.

The norm bound asserted here is the mathematical one implied by a convex EMA
from zero initialisation, not a threshold derived from the observed defect.
"""
from __future__ import annotations

import math

import pytest
import torch

from state_promotion.lm import (
    BudgetCounter,
    LMExperimentConfig,
    completion_nll,
    multiple_choice_accuracy,
    supervised_step,
)
from state_promotion.lora import consolidate_slow_lora, load_lora_model
from state_promotion.pals import generate_retention_stream, protocol_train_order

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SEED = 15007679


@pytest.fixture(scope="module")
def fixture():
    cfg = LMExperimentConfig(model_revision=REVISION, online_lr=3e-3)
    examples, candidates = generate_retention_stream(SEED)
    train = [e for e in examples if e.segment == 0 and e.split == "train"]
    tests = [e for e in examples if e.segment == 0 and e.split == "test"]
    ordered = protocol_train_order({0: train}, SEED)[0]
    torch.manual_seed(SEED + 1701)
    model, tok, _ = load_lora_model(
        cfg, rank=2, adapter_mode="two_timescale", device="cpu", seed=SEED + 1701
    )
    return cfg, model, tok, ordered, tests, candidates


def _observation(model, tok, ex):
    """Independently recompute the masked-mean input embedding for one example."""
    from state_promotion.lm import encode_example

    ids, mask, _ = encode_example(tok, ex)
    with torch.no_grad():
        emb = model.base.get_input_embeddings()(ids)
        m = mask.to(emb.dtype).unsqueeze(-1)
        obs = (emb * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return obs.mean(dim=0, keepdim=True).to(model.latent.dtype)


# 1 + 2
def test_update_is_the_ema_of_input_embeddings_not_hidden_states(fixture):
    cfg, model, tok, ordered, _, _ = fixture
    model.reset_latent()
    budget = BudgetCounter()
    params = model.set_trainable("fast")
    opt = torch.optim.AdamW(params, lr=cfg.online_lr)
    decay = model.latent_decay
    for ex in ordered[:6]:
        expected_obs = _observation(model, tok, ex)
        before = model.latent.detach().clone()
        supervised_step(model, tok, [ex], opt, budget,
                        use_fast=True, use_slow=True, use_latent=True, update_latent=True)
        expected = before * decay + expected_obs * (1.0 - decay)
        assert torch.allclose(model.latent, expected, atol=1e-6), \
            "latent update is not the EMA of masked-mean input embeddings"


def test_source_does_not_read_final_hidden_states_into_latent():
    """The historical PromptStateLM path in lm.py keeps its own implementation for
    reproducibility; the v3 LoRA substrate must not read hidden states at all."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "src" / "state_promotion" / "lora.py").read_text()
    code = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#")
    )
    assert "hidden_states" not in code, "latent must never be fed from final hidden states"
    assert "get_input_embeddings" in code


# 3 + 4
def test_latent_is_a_convex_sum_and_is_norm_bounded_by_its_observations(fixture):
    cfg, model, tok, ordered, _, _ = fixture
    model.reset_latent()
    budget = BudgetCounter()
    params = model.set_trainable("fast")
    opt = torch.optim.AdamW(params, lr=cfg.online_lr)
    max_obs_norm = 0.0
    for ex in ordered[:12]:
        max_obs_norm = max(max_obs_norm, float(_observation(model, tok, ex).norm()))
        supervised_step(model, tok, [ex], opt, budget,
                        use_fast=True, use_slow=True, use_latent=True, update_latent=True)
        # ||z_t|| <= max_i ||obs_i||: mathematical, not performance-derived.
        assert float(model.latent.norm()) <= max_obs_norm + 1e-5


def test_convex_weights_sum_below_one_from_zero_init(fixture):
    _, model, _, _, _, _ = fixture
    decay = model.latent_decay
    for t in (1, 5, 20, 48):
        assert math.isclose(sum((1 - decay) * decay ** i for i in range(t)), 1 - decay ** t, rel_tol=1e-9)
        assert 1 - decay ** t < 1.0


# 5
def test_disabling_latent_keeps_prefix_length_and_uses_the_zero_anchor(fixture):
    _, model, tok, ordered, _, _ = fixture
    from state_promotion.lm import encode_example
    assert model.latent_anchor.shape == model.latent.shape
    assert float(model.latent_anchor.norm()) == 0.0
    ids, mask, labels = encode_example(tok, ordered[0])
    with torch.no_grad():
        on = model.forward_encoded(ids, mask, labels, use_latent=True, update_latent=False)
        off = model.forward_encoded(ids, mask, labels, use_latent=False, update_latent=False)
    assert on.logits.shape == off.logits.shape, "prefix length must match when latent is disabled"


# 6
def test_heldout_scoring_never_advances_latent(fixture):
    _, model, tok, _, tests, candidates = fixture
    before = model.latent.detach().clone()
    multiple_choice_accuracy(model, tok, tests[:3], candidates,
                             use_fast=True, use_slow=True, use_latent=True)
    completion_nll(model, tok, tests[0], candidates[0],
                   use_fast=True, use_slow=True, use_latent=True)
    assert torch.equal(model.latent, before), "held-out scoring must not update latent"


def test_slow_consolidation_never_advances_latent(fixture):
    cfg, model, tok, ordered, _, _ = fixture
    before = model.latent.detach().clone()
    consolidate_slow_lora(model, tok, list(ordered[:8]), BudgetCounter(), cfg,
                          use_latent=True, steps=3)
    assert torch.equal(model.latent, before), "consolidation must use fixed online-accumulated state"


def test_update_latent_false_does_not_advance_latent(fixture):
    cfg, model, tok, ordered, _, _ = fixture
    budget = BudgetCounter()
    opt = torch.optim.AdamW(model.set_trainable("fast"), lr=cfg.online_lr)
    before = model.latent.detach().clone()
    supervised_step(model, tok, [ordered[0]], opt, budget,
                    use_fast=True, use_slow=True, use_latent=True, update_latent=False)
    assert torch.equal(model.latent, before)


# 7
def test_snapshot_restore_returns_bit_identical_latent_and_outputs(fixture):
    cfg, model, tok, ordered, _, _ = fixture
    from state_promotion.lm import encode_example
    budget = BudgetCounter()
    opt = torch.optim.AdamW(model.set_trainable("fast"), lr=cfg.online_lr)
    supervised_step(model, tok, [ordered[0]], opt, budget,
                    use_fast=True, use_slow=True, use_latent=True, update_latent=True)
    snap = model.snapshot_plastic_state()
    ids, mask, labels = encode_example(tok, ordered[1])
    with torch.no_grad():
        before_logits = model.forward_encoded(ids, mask, labels,
                                              use_latent=True, update_latent=False).logits.clone()
    latent_before = model.latent.detach().clone()
    for _ in range(3):
        supervised_step(model, tok, [ordered[2]], opt, budget,
                        use_fast=True, use_slow=True, use_latent=True, update_latent=True)
    assert not torch.equal(model.latent, latent_before)
    model.restore_plastic_state(snap)
    assert torch.equal(model.latent, latent_before), "restore must return latent bit-identically"
    with torch.no_grad():
        after_logits = model.forward_encoded(ids, mask, labels,
                                             use_latent=True, update_latent=False).logits
    assert torch.equal(before_logits, after_logits), "restore must return outputs bit-identically"


def test_snapshot_missing_latent_is_rejected(fixture):
    _, model, _, _, _, _ = fixture
    snap = dict(model.snapshot_plastic_state())
    snap.pop("__latent__")
    with pytest.raises(ValueError):
        model.restore_plastic_state(snap)


# 8
def test_b3_b4_b5_share_one_latent_implementation(fixture):
    cfg, model, _, _, _, _ = fixture
    torch.manual_seed(1)
    other, _, _ = load_lora_model(cfg, rank=2, adapter_mode="two_timescale",
                                  device="cpu", seed=1)
    assert model.latent.shape == other.latent.shape
    assert model.latent_decay == other.latent_decay == 0.95
    assert type(model).forward_encoded is type(other).forward_encoded


# 9
def test_latent_adds_no_trainable_parameters_and_preserves_write_accounting(fixture):
    _, model, _, _, _, _ = fixture
    assert not model.latent.requires_grad
    assert not model.latent_anchor.requires_grad
    latent_names = {n for n, _ in model.named_parameters() if "latent" in n}
    assert latent_names == set(), "latent must not be a trainable parameter"
    assert model.plastic_parameter_count("fast") == 135168
    assert model.plastic_parameter_count("slow") == 135168
