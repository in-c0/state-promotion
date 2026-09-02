from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from state_promotion.lm import BudgetCounter
from state_promotion.lora import (
    AdditiveLoRALinear,
    LoRAAdapterSpec,
    LoRAStateLM,
    adapter_parameter_count,
    frozen_backbone_parameters,
    inject_additive_lora,
    restore_adapters,
    set_active_adapters,
    set_trainable_adapters,
    snapshot_adapters,
)


class TinyAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return torch.tanh(self.q_proj(x) + self.v_proj(x))


class TinyBase(nn.Module):
    def __init__(self, vocab: int = 17, hidden: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([TinyAttention(hidden), TinyAttention(hidden)])
        self.head = nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask=None,
        labels=None,
        output_hidden_states=False,
        use_cache=False,
    ):
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h)
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        hidden_states = (h,) if output_hidden_states else None
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=hidden_states)


def _perturb_b(model, adapter: str, value: float):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, AdditiveLoRALinear):
                module.adapters[adapter].B.fill_(value)


def test_capacity_matches_single_rank_2r_to_fast_plus_slow_rank_r():
    for rank in (1, 2, 4):
        single = TinyBase()
        inject_additive_lora(single, adapter_ranks={"single": 2 * rank}, seed=11)
        two = TinyBase()
        inject_additive_lora(two, adapter_ranks={"fast": rank, "slow": rank}, seed=11)
        assert adapter_parameter_count(single, ("single",)) == adapter_parameter_count(
            two, ("fast", "slow")
        )


def test_fast_slow_are_additive_and_independently_switchable():
    base = nn.Linear(5, 4, bias=False)
    layer = AdditiveLoRALinear(
        base,
        [
            LoRAAdapterSpec("fast", 2, 2),
            LoRAAdapterSpec("slow", 2, 2),
        ],
        seed=9,
    )
    with torch.no_grad():
        layer.adapters["fast"].B.fill_(0.1)
        layer.adapters["slow"].B.fill_(-0.07)
    x = torch.randn(3, 5)

    layer.set_active(())
    y0 = layer(x)
    layer.set_active(("fast",))
    yf = layer(x)
    layer.set_active(("slow",))
    ys = layer(x)
    layer.set_active(("fast", "slow"))
    yfs = layer(x)

    assert not torch.equal(y0, yf)
    assert not torch.equal(y0, ys)
    assert torch.allclose(yfs, yf + ys - y0, atol=1e-6, rtol=1e-6)


def test_gradient_isolation_with_both_deltas_active():
    model = TinyBase()
    inject_additive_lora(model, adapter_ranks={"fast": 2, "slow": 2}, seed=3)
    _perturb_b(model, "fast", 0.01)
    _perturb_b(model, "slow", 0.01)
    set_active_adapters(model, ("fast", "slow"))
    params = set_trainable_adapters(model, ("fast",))
    opt = torch.optim.AdamW(params, lr=1e-3)
    x = torch.randn(2, 4, 8)

    opt.zero_grad(set_to_none=True)
    model(inputs_embeds=x).logits.square().mean().backward()

    fast_grad = []
    slow_grad = []
    for module in model.modules():
        if isinstance(module, AdditiveLoRALinear):
            fast_grad.extend(p.grad for p in module.adapters["fast"].parameters())
            slow_grad.extend(p.grad for p in module.adapters["slow"].parameters())
    assert any(g is not None and torch.count_nonzero(g) for g in fast_grad)
    assert all(g is None for g in slow_grad)


def test_frozen_backbone_is_byte_identical_after_adapter_step():
    model = TinyBase()
    inject_additive_lora(model, adapter_ranks={"fast": 2, "slow": 2}, seed=7)
    _perturb_b(model, "fast", 0.01)
    before = frozen_backbone_parameters(model)
    set_active_adapters(model, ("fast", "slow"))
    params = set_trainable_adapters(model, ("fast",))
    opt = torch.optim.AdamW(params, lr=1e-3)
    x = torch.randn(2, 4, 8)

    opt.zero_grad(set_to_none=True)
    model(inputs_embeds=x).logits.square().mean().backward()
    opt.step()

    after = frozen_backbone_parameters(model)
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_snapshot_restore_recovers_exact_adapter_output():
    model = TinyBase()
    inject_additive_lora(model, adapter_ranks={"fast": 2, "slow": 2}, seed=13)
    _perturb_b(model, "fast", 0.03)
    _perturb_b(model, "slow", -0.02)
    set_active_adapters(model, ("fast", "slow"))
    x = torch.randn(2, 3, 8)

    state = snapshot_adapters(model)
    expected = model(inputs_embeds=x).logits.detach().clone()
    _perturb_b(model, "fast", 1.0)
    assert not torch.equal(expected, model(inputs_embeds=x).logits)
    restore_adapters(model, state)
    actual = model(inputs_embeds=x).logits.detach()
    assert torch.equal(expected, actual)


def test_lora_state_snapshot_includes_latent_and_write_count_is_actual_params():
    model = LoRAStateLM(
        TinyBase(),
        hidden_size=8,
        adapter_mode="two_timescale",
        rank=2,
        seed=17,
    )
    params = model.set_trainable("fast")
    assert BudgetCounter.step_write_units(params) == model.plastic_parameter_count("fast")

    with torch.no_grad():
        model.latent.fill_(2.0)
    state = model.snapshot_plastic_state()
    with torch.no_grad():
        model.latent.zero_()
    _perturb_b(model.base, "fast", 0.5)
    model.restore_plastic_state(state)
    assert torch.equal(model.latent, torch.full_like(model.latent, 2.0))


def test_initial_single_and_two_timescale_adapters_are_noop_and_deterministic():
    torch.manual_seed(123)
    base_a = TinyBase()
    torch.manual_seed(123)
    base_b = TinyBase()
    single = LoRAStateLM(base_a, hidden_size=8, adapter_mode="single", rank=2, seed=19)
    two = LoRAStateLM(base_b, hidden_size=8, adapter_mode="two_timescale", rank=2, seed=19)

    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    out_single = single.forward_encoded(
        ids, mask, None, use_fast=True, use_slow=True, use_latent=False
    ).logits
    out_two = two.forward_encoded(
        ids, mask, None, use_fast=True, use_slow=True, use_latent=False
    ).logits
    assert torch.equal(out_single, out_two)


class TinyTokenizer:
    eos_token_id = None

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [1, 2, 3]}


def test_lora_consolidation_uses_same_explicit_latent_mode_as_scoring():
    from state_promotion.lm import LMExperimentConfig
    from state_promotion.lora import consolidate_slow_lora
    from state_promotion.pals import Example

    model = LoRAStateLM(
        TinyBase(),
        hidden_size=8,
        adapter_mode="two_timescale",
        rank=1,
        seed=23,
    )
    seen_latent_modes = []
    original_forward = model.forward_encoded

    def recorded_forward(*args, **kwargs):
        seen_latent_modes.append(kwargs["use_latent"])
        return original_forward(*args, **kwargs)

    model.forward_encoded = recorded_forward
    cfg = LMExperimentConfig(consolidation_steps=1, consolidation_batch=1)
    ex = Example(
        stream="retention",
        segment=0,
        context="CTX",
        key="KEY",
        target="CODE",
        split="train",
    )

    consolidate_slow_lora(
        model,
        TinyTokenizer(),
        [ex],
        BudgetCounter(),
        cfg,
        use_latent=True,
        steps=1,
    )
    assert seen_latent_modes == [True]
