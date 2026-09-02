#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_promotion.lm import BudgetCounter, LMExperimentConfig, supervised_step  # noqa: E402
from state_promotion.lora import load_lora_model  # noqa: E402
from state_promotion.pals import generate_retention_stream  # noqa: E402

MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SMOKE_SEED = 20260901
SMOKE_RANK = 1
SMOKE_LR = 3e-4


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def tensor_sha256(named_tensors: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(named_tensors):
        h.update(name.encode("utf-8")); h.update(b"\0")
        t = named_tensors[name].detach().cpu().contiguous()
        h.update(t.numpy().tobytes()); h.update(b"\0")
    return h.hexdigest()


def frozen_backbone(model) -> dict[str, torch.Tensor]:
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if ".adapters." not in name
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score-free real-Qwen structural smoke for EXP-001R. Uses training data only."
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/exp001r-structural-smoke.json")
    args = ap.parse_args()

    cfg = LMExperimentConfig(model_revision=MODEL_REVISION, online_lr=SMOKE_LR)
    model_init_seed = 1701 + SMOKE_SEED
    model, tokenizer, device_report = load_lora_model(
        cfg,
        rank=SMOKE_RANK,
        adapter_mode="two_timescale",
        device=args.device,
        seed=model_init_seed,
    )

    expected_targets = int(getattr(model.base.config, "num_hidden_layers")) * 2
    actual_targets = len(model.target_modules)
    if actual_targets != expected_targets:
        raise AssertionError(
            f"expected q_proj+v_proj in every layer ({expected_targets}), got {actual_targets}"
        )

    fast_params = model.plastic_parameter_count("fast")
    slow_params = model.plastic_parameter_count("slow")
    if fast_params != slow_params:
        raise AssertionError(f"fast/slow capacity mismatch: {fast_params} != {slow_params}")

    before = frozen_backbone(model)
    before_hash = tensor_sha256(before)

    examples, _ = generate_retention_stream(SMOKE_SEED)
    first_train = next(ex for ex in examples if ex.segment == 0 and ex.split == "train")
    params = model.set_trainable("fast")
    optimizer = torch.optim.AdamW(params, lr=SMOKE_LR)
    budget = BudgetCounter()
    loss = supervised_step(
        model,
        tokenizer,
        [first_train],
        optimizer,
        budget,
        use_fast=True,
        use_slow=False,
        use_latent=False,
        update_latent=False,
    )

    after = frozen_backbone(model)
    after_hash = tensor_sha256(after)
    backbone_equal = before_hash == after_hash and all(
        torch.equal(before[name], after[name]) for name in before
    )
    if not backbone_equal:
        raise AssertionError("frozen backbone changed during LoRA smoke step")

    if budget.optimizer_steps != 1:
        raise AssertionError(f"expected one optimizer step, got {budget.optimizer_steps}")
    if budget.parameter_write_units != fast_params:
        raise AssertionError(
            f"expected one fast-adapter write ({fast_params}), got {budget.parameter_write_units}"
        )
    if not torch.isfinite(torch.tensor(loss)).item():
        raise AssertionError("non-finite training loss in structural smoke")
    if not all(torch.isfinite(p.detach()).all().item() for p in params):
        raise AssertionError("non-finite LoRA parameter in structural smoke")
    if any(
        p.grad is not None
        for name, p in model.named_parameters()
        if ".adapters." not in name
    ):
        raise AssertionError("frozen backbone accumulated gradients")

    report = {
        "protocol": "EXP-001R-real-Qwen-structural-smoke-v1",
        "status": "DEVELOPMENT_ONLY_NO_HELDOUT_SCORE",
        "heldout_examples_scored": 0,
        "seed": SMOKE_SEED,
        "rank": SMOKE_RANK,
        "lr": SMOKE_LR,
        "target_module_count": actual_targets,
        "expected_target_module_count": expected_targets,
        "fast_parameter_count": fast_params,
        "slow_parameter_count": slow_params,
        "one_training_step_loss": float(loss),
        "optimizer_steps": budget.optimizer_steps,
        "parameter_write_units": budget.parameter_write_units,
        "tokens_processed": budget.tokens_processed,
        "backbone_byte_identical": backbone_equal,
        "backbone_sha256_before": before_hash,
        "backbone_sha256_after": after_hash,
        "model_name": cfg.model_name,
        "requested_model_revision": cfg.model_revision,
        "loaded_model_revision": getattr(model.base.config, "_commit_hash", None),
        "loaded_tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "device": args.device,
        "device_numerics": device_report,
        "git_sha": git_sha(),
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
