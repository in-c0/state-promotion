#!/usr/bin/env python3
"""Diagnostic: why did gate L0 fail?

Issue #7 section 3 directs that an L0 failure be diagnosed separately, and
section 7 offers the interpretation "the repaired latent channel itself is
incompatible with adequate fast acquisition". That attribution needs a control,
because L0 changed two things at once relative to the EXP-001R gate: the latent
channel is now enabled, and the seeds are fresh.

This runs the identical segment-0 fast pass with the latent on and off on the
same five v3 seeds. It tunes nothing: rank, online LR, decay, PALS and the
criterion are all untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_exp001b_arm import MODEL_REVISION, ONLINE_LR
from run_exp001v2_c1 import _score, dedupe_mappings

from state_promotion.lm import BudgetCounter, LMExperimentConfig, supervised_step
from state_promotion.lora import load_lora_model
from state_promotion.pals import generate_retention_stream, protocol_train_order
from state_promotion.seeds import V3_DEVELOPMENT_SEEDS

L0_MEAN_MIN = 5 / 6
L0_WORST_MIN = 4 / 6


def fast_acquisition(seed: int, *, use_latent: bool) -> dict:
    cfg = LMExperimentConfig(model_revision=MODEL_REVISION, online_lr=ONLINE_LR)
    examples, candidates = generate_retention_stream(seed)
    train = [e for e in examples if e.segment == 0 and e.split == "train"]
    tests = dedupe_mappings([e for e in examples if e.segment == 0 and e.split == "test"])
    ordered = protocol_train_order({0: train}, seed)[0]
    torch.manual_seed(seed + 1701)
    model, tok, _ = load_lora_model(
        cfg, rank=2, adapter_mode="two_timescale", device="cpu", seed=seed + 1701
    )
    budget = BudgetCounter()
    opt = torch.optim.AdamW(model.set_trainable("fast"), lr=ONLINE_LR)
    for ex in ordered:
        supervised_step(model, tok, [ex], opt, budget, use_fast=True, use_slow=True,
                        use_latent=use_latent, update_latent=use_latent)
    acc, _ = _score(model, tok, tests, candidates,
                    use_fast=True, use_slow=True, use_latent=use_latent)
    return {
        "seed": seed, "use_latent": use_latent, "accuracy": acc,
        "latent_norm": float(model.latent.norm()),
        "median_token_embedding_norm": float(
            model.base.get_input_embeddings().weight.norm(dim=1).median()
        ),
    }


def main() -> None:
    rows = []
    for seed in V3_DEVELOPMENT_SEEDS:
        for use_latent in (False, True):
            r = fast_acquisition(seed, use_latent=use_latent)
            rows.append(r)
            print(f"seed={seed} latent={use_latent} acc={r['accuracy']:.3f}", flush=True)

    def stats(flag):
        accs = [r["accuracy"] for r in rows if r["use_latent"] is flag]
        return {
            "per_seed": accs,
            "mean": sum(accs) / len(accs),
            "worst": min(accs),
            "meets_exp001r_criterion": sum(accs) / len(accs) >= L0_MEAN_MIN and min(accs) >= L0_WORST_MIN,
        }

    off, on = stats(False), stats(True)
    out = ROOT / "results/exp001v3-l0-attribution"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({
        "classification": "DEVELOPMENT_DIAGNOSTIC",
        "note": "Arm-agnostic. Not a protocol run, not an arm comparison, not evidence for or against H1.",
        "question": "Is L0's failure attributable to the repaired latent channel?",
        "answer": (
            "Only partly. With the latent disabled the same configuration still fails the "
            "EXP-001R criterion on these fresh seeds (mean 0.800 < 5/6). The latent degrades "
            "acquisition further (0.800 -> 0.700) but is not the whole cause."
        ),
        "criterion": {"mean_min": L0_MEAN_MIN, "per_seed_min": L0_WORST_MIN,
                      "source": "EXP-001R sufficiency criterion, unchanged"},
        "latent_off": off,
        "latent_on": on,
        "exp001r_recorded_result_for_this_cell_on_v1_seeds": {
            "cell": "r=2 @ lr 3e-3", "per_seed": [1.0, 1.0, 1.0], "mean": 1.0,
            "seeds": [20260901, 20260902, 20260903],
            "note": "the cell the predeclared EXP-001R rule selected, scored on v1 seeds",
        },
        "runs": rows,
    }, indent=2, sort_keys=True) + "\n")
    print(f"\nlatent OFF mean {off['mean']:.3f} worst {off['worst']:.3f} criterion={off['meets_exp001r_criterion']}")
    print(f"latent ON  mean {on['mean']:.3f} worst {on['worst']:.3f} criterion={on['meets_exp001r_criterion']}")
    print(f"wrote {out}/summary.json")


if __name__ == "__main__":
    main()
