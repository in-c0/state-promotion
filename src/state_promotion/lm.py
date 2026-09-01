from __future__ import annotations

from dataclasses import dataclass
import copy
import random
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .pals import Example


@dataclass
class LMExperimentConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    slow_tokens: int = 4
    fast_tokens: int = 4
    latent_decay: float = 0.95
    online_lr: float = 5e-3
    consolidation_lr: float = 2e-3
    replay_capacity: int = 96
    replay_per_online_step: int = 1
    consolidation_steps: int = 24
    consolidation_batch: int = 1
    promotion_min_fast_gain: float = 0.08
    promotion_min_current_acc: float = 0.45
    promotion_max_retention_drop: float = 0.03


@dataclass
class BudgetCounter:
    optimizer_steps: int = 0
    parameter_write_units: int = 0
    examples_seen: int = 0
    training_examples_processed: int = 0
    tokens_processed: int = 0
    replay_examples_used: int = 0
    write_budget_units: int | None = None
    write_budget_exhausted_steps: int = 0

    @staticmethod
    def step_write_units(params: Iterable[nn.Parameter]) -> int:
        return sum(p.numel() for p in params if p.requires_grad)

    def may_write(self, params: Iterable[nn.Parameter]) -> bool:
        if self.write_budget_units is None:
            return True
        return self.parameter_write_units + self.step_write_units(params) <= self.write_budget_units

    def record_compute(self, *, tokens: int, examples: int, replay_examples: int = 0) -> None:
        self.training_examples_processed += examples
        self.tokens_processed += tokens
        self.replay_examples_used += replay_examples

    def record_step(self, params: Iterable[nn.Parameter]) -> None:
        self.optimizer_steps += 1
        self.parameter_write_units += self.step_write_units(params)


class ReplayStore:
    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.items: list[Example] = []
        self.seen = 0

    def add(self, ex: Example) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(ex)
            return
        j = self.rng.randrange(self.seen)
        if j < self.capacity:
            self.items[j] = ex

    def sample(self, n: int) -> list[Example]:
        if not self.items:
            return []
        return self.rng.sample(self.items, min(n, len(self.items)))


class PromptStateLM(nn.Module):
    """Frozen causal LM with slow/fast prompt parameters and persistent latent state.

    The same prefix length is always present. Disabling the fast or slow component
    substitutes its immutable initial anchor, avoiding a positional-length confound.
    """

    def __init__(self, base: nn.Module, hidden_size: int, cfg: LMExperimentConfig, seed: int):
        super().__init__()
        self.base = base
        self.cfg = cfg
        for p in self.base.parameters():
            p.requires_grad_(False)

        g = torch.Generator(device="cpu").manual_seed(seed)
        scale = 0.015
        slow_init = torch.randn(cfg.slow_tokens, hidden_size, generator=g) * scale
        fast_init = torch.randn(cfg.fast_tokens, hidden_size, generator=g) * scale
        self.slow_prompt = nn.Parameter(slow_init.clone())
        self.fast_prompt = nn.Parameter(fast_init.clone())
        self.register_buffer("slow_anchor", slow_init.clone())
        self.register_buffer("fast_anchor", fast_init.clone())
        self.register_buffer("latent", torch.zeros(1, hidden_size))

    @property
    def device(self) -> torch.device:
        return next(self.base.parameters()).device

    def reset_fast(self) -> None:
        with torch.no_grad():
            self.fast_prompt.copy_(self.fast_anchor)

    def reset_latent(self) -> None:
        self.latent.zero_()

    def set_trainable(self, scope: str) -> list[nn.Parameter]:
        self.slow_prompt.requires_grad_(scope in {"single", "slow"})
        self.fast_prompt.requires_grad_(scope in {"single", "fast"})
        params = [p for p in (self.slow_prompt, self.fast_prompt) if p.requires_grad]
        return params

    def _prefix(self, batch: int, use_slow: bool, use_fast: bool, use_latent: bool) -> torch.Tensor:
        slow = self.slow_prompt if use_slow else self.slow_anchor
        fast = self.fast_prompt if use_fast else self.fast_anchor
        latent = self.latent if use_latent else torch.zeros_like(self.latent)
        prefix = torch.cat([slow, fast, latent], dim=0).to(self.device)
        return prefix.unsqueeze(0).expand(batch, -1, -1)

    def forward_encoded(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None,
                        *, use_slow: bool = True, use_fast: bool = True, use_latent: bool = True,
                        update_latent: bool = False):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device) if labels is not None else None
        token_emb = self.base.get_input_embeddings()(input_ids)
        prefix = self._prefix(token_emb.shape[0], use_slow, use_fast, use_latent).to(token_emb.dtype)
        inputs_embeds = torch.cat([prefix, token_emb], dim=1)
        prefix_mask = torch.ones((attention_mask.shape[0], prefix.shape[1]), device=self.device, dtype=attention_mask.dtype)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        full_labels = None
        if labels is not None:
            ignored = torch.full((labels.shape[0], prefix.shape[1]), -100, device=self.device, dtype=labels.dtype)
            full_labels = torch.cat([ignored, labels], dim=1)

        out = self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
            output_hidden_states=update_latent,
            use_cache=False,
        )
        if update_latent:
            with torch.no_grad():
                h = out.hidden_states[-1][:, prefix.shape[1]:, :]
                mask = attention_mask.to(h.dtype).unsqueeze(-1)
                obs = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                obs = obs.mean(dim=0, keepdim=True)
                self.latent.mul_(self.cfg.latent_decay).add_(obs * (1.0 - self.cfg.latent_decay))
        return out

    def snapshot_plastic_state(self) -> dict[str, torch.Tensor]:
        return {
            "slow_prompt": self.slow_prompt.detach().clone(),
            "fast_prompt": self.fast_prompt.detach().clone(),
            "latent": self.latent.detach().clone(),
        }

    def restore_plastic_state(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            self.slow_prompt.copy_(state["slow_prompt"])
            self.fast_prompt.copy_(state["fast_prompt"])
            self.latent.copy_(state["latent"])


def load_model(cfg: LMExperimentConfig, device: str | None = None) -> tuple[PromptStateLM, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype)
    base.to(device)
    base.eval()
    hidden = int(base.get_input_embeddings().embedding_dim)
    model = PromptStateLM(base, hidden, cfg, seed=1701).to(device)
    return model, tokenizer


def encode_example(tokenizer, ex: Example) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer(ex.prompt, add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(ex.target, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]
    ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long)
    mask = torch.ones_like(ids)
    labels = torch.tensor([[-100] * len(prompt_ids) + target_ids], dtype=torch.long)
    return ids, mask, labels


def supervised_step(model: PromptStateLM, tokenizer, examples: Sequence[Example], optimizer: torch.optim.Optimizer,
                    budget: BudgetCounter, *, use_fast: bool, use_slow: bool, use_latent: bool,
                    update_latent: bool = True, replay_examples: int = 0) -> float:
    """Process one adaptation batch and perform at most one optimizer write.

    Replay examples contribute to the same optimizer step as the current example.
    This makes optimizer/write accounting independent of batch cardinality and lets
    EXP-001 match adaptation-token exposure separately from parameter writes.
    """
    params = [p for group in optimizer.param_groups for p in group["params"]]
    can_write = budget.may_write(params)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    total_tokens = 0
    denom = max(len(examples), 1)
    for i, ex in enumerate(examples):
        ids, mask, labels = encode_example(tokenizer, ex)
        total_tokens += int(ids.numel())
        out = model.forward_encoded(
            ids, mask, labels,
            use_slow=use_slow, use_fast=use_fast, use_latent=use_latent,
            update_latent=update_latent and i == len(examples) - 1,
        )
        if can_write:
            (out.loss / denom).backward()
        total += float(out.loss.detach().cpu())
    budget.record_compute(tokens=total_tokens, examples=len(examples), replay_examples=replay_examples)
    if can_write:
        optimizer.step()
        budget.record_step(params)
    else:
        budget.write_budget_exhausted_steps += 1
    return total / denom


@torch.no_grad()
def completion_nll(model: PromptStateLM, tokenizer, ex: Example, candidate: str, *,
                   use_fast: bool = True, use_slow: bool = True, use_latent: bool = True) -> float:
    candidate_ex = Example(
        stream=ex.stream,
        segment=ex.segment,
        context=ex.context,
        key=ex.key,
        target=candidate,
        split=ex.split,
        relation=ex.relation,
        version=ex.version,
    )
    ids, mask, labels = encode_example(tokenizer, candidate_ex)
    out = model.forward_encoded(ids, mask, labels, use_slow=use_slow, use_fast=use_fast,
                                use_latent=use_latent, update_latent=False)
    return float(out.loss.detach().float().cpu())


@torch.no_grad()
def multiple_choice_accuracy(model: PromptStateLM, tokenizer, examples: Sequence[Example], candidates: Sequence[str],
                             *, use_fast: bool = True, use_slow: bool = True, use_latent: bool = True,
                             max_examples: int | None = None) -> float:
    if max_examples is not None:
        examples = examples[:max_examples]
    if not examples:
        return float("nan")
    correct = 0
    for ex in examples:
        scores = [completion_nll(model, tokenizer, ex, c, use_fast=use_fast, use_slow=use_slow,
                                 use_latent=use_latent) for c in candidates]
        pred = candidates[min(range(len(scores)), key=scores.__getitem__)]
        correct += int(pred == ex.target)
    return correct / len(examples)


def consolidate_slow(model: PromptStateLM, tokenizer, replay_examples: Sequence[Example], budget: BudgetCounter,
                     cfg: LMExperimentConfig, steps: int | None = None) -> None:
    """Train slow state on replay with the current fast prompt disabled.

    This is deliberately not an algebraic fast->slow merge. Slow consolidation is
    a separate optimization process on selected attributable evidence.
    """
    params = model.set_trainable("slow")
    optimizer = torch.optim.AdamW(params, lr=cfg.consolidation_lr)
    if not replay_examples:
        return
    rng = random.Random(7729 + budget.optimizer_steps)
    planned_steps = cfg.consolidation_steps if steps is None else steps
    for _ in range(planned_steps):
        if not budget.may_write(params):
            break
        batch = [rng.choice(replay_examples) for _ in range(min(cfg.consolidation_batch, len(replay_examples)))]
        supervised_step(
            model, tokenizer, batch, optimizer, budget,
            use_fast=False, use_slow=True, use_latent=False, update_latent=False,
            replay_examples=len(batch),
        )


def guarded_promotion(model: PromptStateLM, tokenizer, *, current_test: Sequence[Example],
                      protected_test: Sequence[Example], candidates: Sequence[str], replay: ReplayStore,
                      budget: BudgetCounter, cfg: LMExperimentConfig, use_latent: bool = True,
                      rollback_on_retention: bool = True, consolidation_steps: int | None = None) -> tuple[bool, dict[str, float]]:
    """Attempt slow consolidation behind an explicit candidate/commit boundary.

    ``use_latent`` supports the preregistered latent-state ablation.
    ``rollback_on_retention=False`` is the evidence-gate-without-retention-rollback
    ablation; it should never be used for the primary method.
    """
    current_with_fast = multiple_choice_accuracy(model, tokenizer, current_test, candidates,
                                                 use_fast=True, use_slow=True, use_latent=use_latent)
    current_slow_only = multiple_choice_accuracy(model, tokenizer, current_test, candidates,
                                                 use_fast=False, use_slow=True, use_latent=use_latent)
    fast_gain = current_with_fast - current_slow_only
    evidence = {
        "current_with_fast": current_with_fast,
        "current_slow_only": current_slow_only,
        "fast_gain": fast_gain,
    }
    if current_with_fast < cfg.promotion_min_current_acc or fast_gain < cfg.promotion_min_fast_gain:
        evidence["gate"] = 0.0
        return False, evidence

    retention_before = multiple_choice_accuracy(model, tokenizer, protected_test, candidates,
                                                use_fast=True, use_slow=True, use_latent=use_latent) if protected_test else 1.0
    snapshot = model.snapshot_plastic_state()
    consolidate_slow(model, tokenizer, replay.items, budget, cfg, steps=consolidation_steps)
    current_after = multiple_choice_accuracy(model, tokenizer, current_test, candidates,
                                             use_fast=False, use_slow=True, use_latent=use_latent)
    retention_after = multiple_choice_accuracy(model, tokenizer, protected_test, candidates,
                                               use_fast=False, use_slow=True, use_latent=use_latent) if protected_test else 1.0
    retention_drop = retention_before - retention_after
    evidence.update({
        "retention_before": retention_before,
        "retention_after": retention_after,
        "retention_drop": retention_drop,
        "current_after": current_after,
    })

    if rollback_on_retention and (retention_drop > cfg.promotion_max_retention_drop or current_after < cfg.promotion_min_current_acc):
        model.restore_plastic_state(snapshot)
        evidence["gate"] = -1.0
        return False, evidence

    model.reset_fast()
    evidence["gate"] = 1.0
    return True, evidence
