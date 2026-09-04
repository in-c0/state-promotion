from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LoRAAdapterSpec:
    name: str
    rank: int
    alpha: float

    @property
    def scaling(self) -> float:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        return float(self.alpha) / float(self.rank)


class LoRADelta(nn.Module):
    """One explicit LoRA delta, kept in fp32 even when the frozen base is fp16."""

    def __init__(self, in_features: int, out_features: int, spec: LoRAAdapterSpec, *, seed: int):
        super().__init__()
        if spec.rank <= 0:
            raise ValueError("rank must be positive")
        self.spec = spec
        g = torch.Generator(device="cpu").manual_seed(seed)
        std = 1.0 / math.sqrt(max(in_features, 1))
        a = torch.randn(spec.rank, in_features, generator=g, dtype=torch.float32) * std
        b = torch.zeros(out_features, spec.rank, dtype=torch.float32)
        self.A = nn.Parameter(a)
        self.B = nn.Parameter(b)
        self.register_buffer("initial_A", a.clone(), persistent=False)
        self.register_buffer("initial_B", b.clone(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.to(dtype=self.A.dtype)
        delta = F.linear(F.linear(x32, self.A), self.B)
        return delta * self.spec.scaling

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.A.copy_(self.initial_A)
            self.B.copy_(self.initial_B)


class AdditiveLoRALinear(nn.Module):
    """Frozen Linear layer plus any simultaneously active additive LoRA deltas."""

    def __init__(
        self,
        base: nn.Linear,
        adapter_specs: Sequence[LoRAAdapterSpec],
        *,
        seed: int,
    ):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.adapters = nn.ModuleDict()
        for i, spec in enumerate(adapter_specs):
            self.adapters[spec.name] = LoRADelta(
                base.in_features,
                base.out_features,
                spec,
                seed=seed + 104729 * (i + 1),
            )
        self._active: tuple[str, ...] = tuple(spec.name for spec in adapter_specs)
        self.set_trainable(())

    @property
    def active_adapters(self) -> tuple[str, ...]:
        return self._active

    def set_active(self, names: Iterable[str]) -> None:
        names = tuple(names)
        missing = [name for name in names if name not in self.adapters]
        if missing:
            raise KeyError(f"unknown LoRA adapters: {missing}")
        self._active = names

    def set_trainable(self, names: Iterable[str]) -> list[nn.Parameter]:
        requested = set(names)
        missing = requested.difference(self.adapters.keys())
        if missing:
            raise KeyError(f"unknown LoRA adapters: {sorted(missing)}")
        params: list[nn.Parameter] = []
        for name, adapter in self.adapters.items():
            train = name in requested
            for p in adapter.parameters():
                p.requires_grad_(train)
                if train:
                    params.append(p)
        for p in self.base.parameters():
            p.requires_grad_(False)
        return params

    def adapter_parameters(self, name: str) -> list[nn.Parameter]:
        if name not in self.adapters:
            raise KeyError(name)
        return list(self.adapters[name].parameters())

    def adapter_parameter_count(self, name: str) -> int:
        return sum(p.numel() for p in self.adapter_parameters(name))

    def reset_adapter(self, name: str) -> None:
        self.adapters[name].reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        for name in self._active:
            delta = self.adapters[name](x).to(dtype=out.dtype)
            out = out + delta
        return out


def _resolve_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, child_name


def inject_additive_lora(
    model: nn.Module,
    *,
    adapter_ranks: Mapping[str, int],
    target_suffixes: Sequence[str] = ("q_proj", "v_proj"),
    seed: int = 1701,
) -> list[str]:
    """Replace target Linear layers with explicit additive LoRA wrappers.

    alpha/r is fixed to one by setting alpha=rank for every adapter.
    """
    if not adapter_ranks:
        raise ValueError("adapter_ranks must not be empty")
    specs = [
        LoRAAdapterSpec(name=name, rank=int(rank), alpha=float(rank))
        for name, rank in adapter_ranks.items()
    ]

    targets: list[tuple[str, nn.Linear]] = []
    suffixes = set(target_suffixes)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in suffixes:
            targets.append((name, module))
    if not targets:
        raise ValueError(f"no Linear targets found for suffixes={tuple(target_suffixes)}")

    replaced: list[str] = []
    for i, (name, module) in enumerate(targets):
        parent, child = _resolve_parent(model, name)
        setattr(
            parent,
            child,
            AdditiveLoRALinear(module, specs, seed=seed + 1000003 * (i + 1)),
        )
        replaced.append(name)

    freeze_non_lora_parameters(model)
    return replaced


def iter_lora_layers(model: nn.Module) -> Iterable[tuple[str, AdditiveLoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, AdditiveLoRALinear):
            yield name, module


def freeze_non_lora_parameters(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def set_active_adapters(model: nn.Module, names: Iterable[str]) -> None:
    names = tuple(names)
    for _, layer in iter_lora_layers(model):
        layer.set_active(names)


def set_trainable_adapters(model: nn.Module, names: Iterable[str]) -> list[nn.Parameter]:
    names = tuple(names)
    params: list[nn.Parameter] = []
    for _, layer in iter_lora_layers(model):
        params.extend(layer.set_trainable(names))
    return params


def adapter_parameter_count(model: nn.Module, names: Iterable[str]) -> int:
    wanted = set(names)
    total = 0
    for _, layer in iter_lora_layers(model):
        for name in wanted:
            total += layer.adapter_parameter_count(name)
    return total


def reset_adapter(model: nn.Module, name: str) -> None:
    for _, layer in iter_lora_layers(model):
        layer.reset_adapter(name)


def snapshot_adapters(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for module_name, layer in iter_lora_layers(model):
        for adapter_name, adapter in layer.adapters.items():
            state[f"{module_name}::{adapter_name}::A"] = adapter.A.detach().clone()
            state[f"{module_name}::{adapter_name}::B"] = adapter.B.detach().clone()
    return state


def restore_adapters(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    expected = snapshot_adapters(model)
    if set(state) != set(expected):
        missing = sorted(set(expected).difference(state))
        extra = sorted(set(state).difference(expected))
        raise ValueError(f"adapter snapshot mismatch: missing={missing} extra={extra}")
    with torch.no_grad():
        for module_name, layer in iter_lora_layers(model):
            for adapter_name, adapter in layer.adapters.items():
                adapter.A.copy_(state[f"{module_name}::{adapter_name}::A"])
                adapter.B.copy_(state[f"{module_name}::{adapter_name}::B"])


def frozen_backbone_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return frozen non-LoRA parameters for exact before/after equality tests."""
    out: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if ".adapters." not in name:
            out[name] = p.detach().clone()
    return out


class LoRAStateLM(nn.Module):
    """Frozen causal LM with additive fast/slow LoRA and a separate latent token."""

    def __init__(
        self,
        base: nn.Module,
        *,
        hidden_size: int,
        adapter_mode: str,
        rank: int,
        latent_decay: float = 0.95,
        seed: int = 1701,
        target_suffixes: Sequence[str] = ("q_proj", "v_proj"),
    ):
        super().__init__()
        if adapter_mode not in {"single", "two_timescale"}:
            raise ValueError("adapter_mode must be 'single' or 'two_timescale'")
        self.base = base
        self.hidden_size = int(hidden_size)
        self.adapter_mode = adapter_mode
        self.rank = int(rank)
        self.latent_decay = float(latent_decay)

        ranks = {"single": 2 * rank} if adapter_mode == "single" else {"fast": rank, "slow": rank}
        self.target_modules = inject_additive_lora(
            self.base,
            adapter_ranks=ranks,
            target_suffixes=target_suffixes,
            seed=seed,
        )
        self.register_buffer("latent", torch.zeros(1, hidden_size, dtype=torch.float32))
        self.register_buffer("latent_anchor", torch.zeros(1, hidden_size, dtype=torch.float32), persistent=False)
        self._configure_active(use_fast=True, use_slow=True)
        self.set_trainable("none")

    @property
    def device(self) -> torch.device:
        return next(self.base.parameters()).device

    def _available(self) -> tuple[str, ...]:
        return ("single",) if self.adapter_mode == "single" else ("fast", "slow")

    def _configure_active(self, *, use_fast: bool, use_slow: bool) -> None:
        if self.adapter_mode == "single":
            active = ("single",) if (use_fast or use_slow) else ()
        else:
            active = tuple(
                name
                for name, enabled in (("fast", use_fast), ("slow", use_slow))
                if enabled
            )
        set_active_adapters(self.base, active)

    def set_trainable(self, scope: str) -> list[nn.Parameter]:
        if scope == "none":
            names: tuple[str, ...] = ()
        elif self.adapter_mode == "single" and scope == "single":
            names = ("single",)
        elif self.adapter_mode == "two_timescale" and scope in {"fast", "slow"}:
            names = (scope,)
        else:
            raise ValueError(f"scope={scope!r} invalid for adapter_mode={self.adapter_mode!r}")
        return set_trainable_adapters(self.base, names)

    def plastic_parameter_count(self, scope: str | None = None) -> int:
        if scope is None:
            names = self._available()
        elif scope == "single":
            names = ("single",)
        elif scope in {"fast", "slow"}:
            names = (scope,)
        else:
            raise ValueError(scope)
        return adapter_parameter_count(self.base, names)

    def reset_fast(self) -> None:
        if self.adapter_mode != "two_timescale":
            raise RuntimeError("reset_fast applies only to two-timescale models")
        reset_adapter(self.base, "fast")

    def reset_latent(self) -> None:
        self.latent.zero_()

    def forward_encoded(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        *,
        use_slow: bool = True,
        use_fast: bool = True,
        use_latent: bool = True,
        update_latent: bool = False,
    ):
        self._configure_active(use_fast=use_fast, use_slow=use_slow)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device) if labels is not None else None

        token_emb = self.base.get_input_embeddings()(input_ids)

        # EXP-001 v3 latent repair (issue #7 section 0). The persistent state is a
        # vector in the *frozen input-embedding space*, observed from token_emb
        # before prefix concatenation. The v1/v2 port EMAed out.hidden_states[-1]
        # into this same buffer and injected the result as an input embedding,
        # mixing two spaces with nothing between them; the latent reached ~445x the
        # median token-embedding norm. Observation is computed here, but applied
        # after the forward, so the prefix seen while predicting example t is
        # z_{t-1} and an example never summarises its own target into its own
        # prefix.
        latent_observation = None
        if update_latent:
            with torch.no_grad():
                mask = attention_mask.to(token_emb.dtype).unsqueeze(-1)
                obs = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                latent_observation = obs.mean(dim=0, keepdim=True).to(self.latent.dtype)

        latent = self.latent if use_latent else self.latent_anchor
        prefix = latent.to(device=self.device, dtype=token_emb.dtype).unsqueeze(0).expand(
            token_emb.shape[0], -1, -1
        )
        inputs_embeds = torch.cat([prefix, token_emb], dim=1)
        prefix_mask = torch.ones(
            (attention_mask.shape[0], 1),
            device=self.device,
            dtype=attention_mask.dtype,
        )
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        full_labels = None
        if labels is not None:
            ignored = torch.full(
                (labels.shape[0], 1),
                -100,
                device=self.device,
                dtype=labels.dtype,
            )
            full_labels = torch.cat([ignored, labels], dim=1)

        out = self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
            use_cache=False,
        )
        if latent_observation is not None:
            with torch.no_grad():
                # z_t = decay * z_{t-1} + (1 - decay) * obs_t, decay unchanged at 0.95.
                # From zero init this is a convex exponentially weighted sum of
                # embedding-space observations, so ||z_t|| <= max_i ||obs_i||
                # by construction rather than by any chosen threshold.
                self.latent.mul_(self.latent_decay).add_(
                    latent_observation * (1.0 - self.latent_decay)
                )
        return out

    def snapshot_plastic_state(self) -> dict[str, torch.Tensor]:
        state = snapshot_adapters(self.base)
        state["__latent__"] = self.latent.detach().clone()
        return state

    def restore_plastic_state(self, state: Mapping[str, torch.Tensor]) -> None:
        state = dict(state)
        latent = state.pop("__latent__", None)
        if latent is None:
            raise ValueError("plastic snapshot missing latent")
        restore_adapters(self.base, state)
        with torch.no_grad():
            self.latent.copy_(latent)


def load_lora_model(
    cfg,
    *,
    rank: int,
    adapter_mode: str,
    device: str | None = None,
    seed: int = 1701,
    verify_numerics: bool = True,
    compute_dtype: torch.dtype | None = None,
):
    """Load the same frozen LM snapshot used by EXP-001, then inject explicit LoRA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .lm import verify_device_numerics
    from .provenance import collect_provenance

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Accelerators previously defaulted to float16. Under EXP-001B that silently
    # destroys adaptation: a full B1 lifetime on MPS returned chance accuracy
    # (mean diagonal 0.167) and zero forgetting while every existing guard
    # passed, because verify_device_numerics only compares a forward loss and
    # never exercises an optimizer step. Training precision is float32 unless a
    # caller explicitly opts out.
    dtype = torch.float32 if compute_dtype is None else compute_dtype
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        revision=cfg.model_revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        revision=cfg.model_revision,
        torch_dtype=dtype,
    )
    base.to(device)
    base.eval()

    device_report = {"device": device, "verified": None, "checked": False}
    if verify_numerics and device != "cpu":
        device_report = verify_device_numerics(
            base,
            tokenizer,
            device,
            tolerance=cfg.device_probe_tolerance,
        )
        device_report["checked"] = True

    hidden = int(base.get_input_embeddings().embedding_dim)
    model = LoRAStateLM(
        base,
        hidden_size=hidden,
        adapter_mode=adapter_mode,
        rank=rank,
        latent_decay=cfg.latent_decay,
        seed=seed,
    ).to(device)

    # Amendment F provenance is verified from the resolved snapshot directory and
    # the SHA-256 of the tokenizer assets it contains, not from optional
    # `_commit_hash` attributes that Transformers 5.x leaves unset. A conflicting
    # pin raises here rather than being recorded as null.
    model.provenance = collect_provenance(
        model_name=cfg.model_name,
        requested_revision=cfg.model_revision,
        model=base,
        tokenizer=tokenizer,
        device=device,
    )
    return model, tokenizer, device_report


def consolidate_slow_lora(
    model: LoRAStateLM,
    tokenizer,
    replay_examples: Sequence,
    budget,
    cfg,
    *,
    use_latent: bool,
    steps: int | None = None,
) -> None:
    """Optimize slow LoRA under the same latent mode used by candidate scoring."""
    from .lm import supervised_step

    params = model.set_trainable("slow")
    optimizer = torch.optim.AdamW(params, lr=cfg.consolidation_lr)
    if not replay_examples:
        return
    rng = random.Random(7729 + budget.optimizer_steps)
    planned_steps = cfg.consolidation_steps if steps is None else steps
    for _ in range(planned_steps):
        if not budget.may_write(params):
            break
        batch = [
            rng.choice(replay_examples)
            for _ in range(min(cfg.consolidation_batch, len(replay_examples)))
        ]
        supervised_step(
            model,
            tokenizer,
            batch,
            optimizer,
            budget,
            use_fast=False,
            use_slow=True,
            use_latent=use_latent,
            update_latent=False,
            replay_examples=len(batch),
        )


def guarded_promotion_lora(
    model: LoRAStateLM,
    tokenizer,
    *,
    current_probe: Sequence,
    protected_probe: Sequence,
    candidates: Sequence[str],
    replay,
    budget,
    cfg,
    use_latent: bool = True,
    rollback_on_retention: bool = True,
    consolidation_steps: int | None = None,
    consolidation_examples: Sequence | None = None,
) -> tuple[bool, dict[str, float]]:
    """LoRA promotion path with explicit latent-mode consistency and rollback."""
    from .lm import multiple_choice_accuracy

    if any(ex.split != "train" for ex in current_probe):
        raise ValueError("current promotion probe contains non-training examples")
    if any(ex.split != "train" for ex in protected_probe):
        raise ValueError("protected promotion probe contains non-training examples")

    current_with_fast = multiple_choice_accuracy(
        model,
        tokenizer,
        current_probe,
        candidates,
        use_fast=True,
        use_slow=True,
        use_latent=use_latent,
        decision_budget=budget,
    )
    current_slow_only = multiple_choice_accuracy(
        model,
        tokenizer,
        current_probe,
        candidates,
        use_fast=False,
        use_slow=True,
        use_latent=use_latent,
        decision_budget=budget,
    )
    fast_gain = current_with_fast - current_slow_only
    evidence = {
        "current_with_fast": current_with_fast,
        "current_slow_only": current_slow_only,
        "fast_gain": fast_gain,
    }
    if current_with_fast < cfg.promotion_min_current_acc or fast_gain < cfg.promotion_min_fast_gain:
        evidence["gate"] = 0.0
        return False, evidence

    retention_before = (
        multiple_choice_accuracy(
            model,
            tokenizer,
            protected_probe,
            candidates,
            use_fast=True,
            use_slow=True,
            use_latent=use_latent,
            decision_budget=budget,
        )
        if protected_probe
        else 1.0
    )
    snapshot = model.snapshot_plastic_state()
    evidence_examples = replay.items if consolidation_examples is None else consolidation_examples
    consolidate_slow_lora(
        model,
        tokenizer,
        evidence_examples,
        budget,
        cfg,
        use_latent=use_latent,
        steps=consolidation_steps,
    )
    current_after = multiple_choice_accuracy(
        model,
        tokenizer,
        current_probe,
        candidates,
        use_fast=False,
        use_slow=True,
        use_latent=use_latent,
        decision_budget=budget,
    )
    retention_after = (
        multiple_choice_accuracy(
            model,
            tokenizer,
            protected_probe,
            candidates,
            use_fast=False,
            use_slow=True,
            use_latent=use_latent,
            decision_budget=budget,
        )
        if protected_probe
        else 1.0
    )
    retention_drop = retention_before - retention_after
    evidence.update(
        {
            "retention_before": retention_before,
            "retention_after": retention_after,
            "retention_drop": retention_drop,
            "current_after": current_after,
        }
    )

    if rollback_on_retention and (
        retention_drop > cfg.promotion_max_retention_drop
        or current_after < cfg.promotion_min_current_acc
    ):
        model.restore_plastic_state(snapshot)
        evidence["gate"] = -1.0
        return False, evidence

    model.reset_fast()
    evidence["gate"] = 1.0
    return True, evidence
