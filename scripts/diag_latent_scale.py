#!/usr/bin/env python3
"""Diagnostic: the persistent latent channel injects a scale-mismatched prefix.

Arm-agnostic, single segment, fresh v2 development seeds. Not a protocol run,
not an arm comparison, not evidence for or against H1.

LoRAStateLM.forward_encoded prepends `self.latent` to the *input embeddings*,
while the update rule at the same call site EMAs `out.hidden_states[-1]`, which
lives in the final hidden-state space. Nothing projects or renormalises between
the two. The prefix therefore drifts toward last-layer activation magnitudes
while the tokens beside it stay at input-embedding magnitudes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp001v2_c1 import MODEL_REVISION, ONLINE_LR, _score, dedupe_mappings

from state_promotion.lm import BudgetCounter, LMExperimentConfig, supervised_step
from state_promotion.lora import consolidate_slow_lora, load_lora_model
from state_promotion.pals import generate_retention_stream, protocol_train_order
from state_promotion.seeds import V2_DEVELOPMENT_SEEDS


def probe(seed: int, *, use_latent: bool) -> dict:
    cfg = LMExperimentConfig(model_revision=MODEL_REVISION, online_lr=ONLINE_LR)
    cfg.consolidation_lr = 3e-3
    cfg.consolidation_batch = 1
    examples, candidates = generate_retention_stream(seed)
    train = [e for e in examples if e.segment == 0 and e.split == "train"]
    tests = dedupe_mappings([e for e in examples if e.segment == 0 and e.split == "test"])
    ordered = protocol_train_order({0: train}, seed)[0]

    torch.manual_seed(seed + 1701)
    model, tok, _ = load_lora_model(
        cfg, rank=2, adapter_mode="two_timescale", device="cpu", seed=seed + 1701
    )
    emb = model.base.get_input_embeddings().weight
    median_token_norm = float(emb.norm(dim=1).median())

    budget = BudgetCounter()
    params = model.set_trainable("fast")
    opt = torch.optim.AdamW(params, lr=ONLINE_LR)
    trace = []
    for i, ex in enumerate(ordered):
        supervised_step(model, tok, [ex], opt, budget, use_fast=True, use_slow=False,
                        use_latent=use_latent, update_latent=use_latent)
        if i + 1 in (1, 6, 12, 24, 48):
            trace.append({"updates": i + 1, "latent_norm": float(model.latent.norm())})
    fast_only, _ = _score(model, tok, tests, candidates, use_fast=True, use_slow=False)
    consolidate_slow_lora(model, tok, list(ordered), budget, cfg,
                          use_latent=use_latent, steps=48)
    slow_only, _ = _score(model, tok, tests, candidates, use_fast=False, use_slow=True)
    return {
        "seed": seed,
        "use_latent": use_latent,
        "fast_only_accuracy": fast_only,
        "slow_only_accuracy": slow_only,
        "final_latent_norm": float(model.latent.norm()),
        "median_token_embedding_norm": median_token_norm,
        "latent_norm_ratio": float(model.latent.norm()) / median_token_norm,
        "latent_growth_trace": trace,
    }


def main() -> None:
    rows = []
    for seed in V2_DEVELOPMENT_SEEDS[:3]:
        for use_latent in (False, True):
            r = probe(seed, use_latent=use_latent)
            rows.append(r)
            print(f"seed={seed} latent={use_latent} fast={r['fast_only_accuracy']:.3f} "
                  f"slow={r['slow_only_accuracy']:.3f} latent_norm={r['final_latent_norm']:.1f} "
                  f"({r['latent_norm_ratio']:.0f}x median token embedding)", flush=True)
    out = ROOT / "results/exp001v2-latent-diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({
        "classification": "DEVELOPMENT_DIAGNOSTIC",
        "note": "Arm-agnostic. Not a protocol run, not an arm comparison, not evidence for or against H1.",
        "question": "Does the persistent latent channel help or harm acquisition and consolidation?",
        "mechanism": (
            "forward_encoded prepends self.latent to input embeddings while EMAing "
            "out.hidden_states[-1] into it. The two spaces are never reconciled, so the "
            "prefix drifts to last-layer activation scale beside input-scale tokens."
        ),
        "affected_arms": "B3, B4 and B5 use the latent channel; B0, B1 and B2 do not.",
        "runs": rows,
    }, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}/summary.json")


if __name__ == "__main__":
    main()
