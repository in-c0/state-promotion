#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from state_promotion.lm import BudgetCounter, LMExperimentConfig, completion_nll, supervised_step  # noqa: E402
from state_promotion.lora import load_lora_model  # noqa: E402
from state_promotion.pals import Example, generate_retention_stream  # noqa: E402

DEV_SEEDS = (20260901, 20260902, 20260903)
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def source_tree_sha256() -> str:
    h = hashlib.sha256()
    roots = [ROOT / "src", ROOT / "scripts", ROOT / "tests", ROOT / "experiments", ROOT / "docs"]
    files = [ROOT / "README.md", ROOT / "pyproject.toml", ROOT / "Makefile"]
    for root in roots:
        if root.exists():
            files.extend(x for x in root.rglob("*") if x.is_file())
    for path in sorted(set(files), key=lambda x: x.relative_to(ROOT).as_posix()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        rel = path.relative_to(ROOT).as_posix()
        h.update(rel.encode("utf-8")); h.update(b"\0")
        h.update(path.read_bytes()); h.update(b"\0")
    return h.hexdigest()


def dedupe_mappings(items: list[Example]) -> list[Example]:
    seen: set[tuple[str, str, str]] = set()
    out: list[Example] = []
    for ex in items:
        key = (ex.context, ex.key, ex.target)
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def score_mappings(model, tokenizer, tests: list[Example], candidates: list[str]) -> float:
    if not tests:
        return float("nan")
    correct = 0
    for ex in tests:
        scores = [
            completion_nll(
                model,
                tokenizer,
                ex,
                candidate,
                use_fast=True,
                use_slow=False,
                use_latent=False,
            )
            for candidate in candidates
        ]
        pred = candidates[min(range(len(scores)), key=scores.__getitem__)]
        correct += int(pred == ex.target)
    return correct / len(tests)


def load_selected(summary_path: Path) -> tuple[int, float, dict]:
    if not summary_path.exists():
        raise FileNotFoundError(f"representation summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("protocol") != "EXP-001R-representation-sufficiency-v1":
        raise ValueError("unexpected representation-summary protocol")
    if not summary.get("next_gate_allowed") or not summary.get("selected"):
        raise RuntimeError(
            "representation sufficiency did not pass; sequential gate is forbidden"
        )
    selected = summary["selected"]
    return int(selected["rank"]), float(selected["lr"]), summary


def average_forgetting(matrix: list[list[float]]) -> float:
    if not matrix:
        return float("nan")
    final = matrix[-1]
    per_task = []
    for task in range(len(final)):
        observed = [matrix[i][task] for i in range(task, len(matrix))]
        per_task.append(max(observed) - final[task])
    return sum(per_task) / len(per_task)


def run_seed(*, seed: int, rank: int, lr: float, device: str, output_dir: Path) -> dict:
    cfg = LMExperimentConfig(model_revision=MODEL_REVISION, online_lr=lr)
    model_init_seed = 1701 + seed
    model, tokenizer, device_report = load_lora_model(
        cfg,
        rank=rank,
        adapter_mode="single",
        device=device,
        seed=model_init_seed,
    )

    # adapter_mode='single' constructs one rank-2r adapter, matching fast-r + slow-r total capacity.
    params = model.set_trainable("single")
    plastic_params = sum(p.numel() for p in params)
    optimizer = torch.optim.AdamW(params, lr=lr)
    budget = BudgetCounter()

    examples, candidates = generate_retention_stream(seed)
    segments: dict[int, dict[str, list[Example]]] = {}
    for seg in range(6):
        segments[seg] = {
            "train": [e for e in examples if e.segment == seg and e.split == "train"],
            "test": dedupe_mappings([e for e in examples if e.segment == seg and e.split == "test"]),
        }
        if len(segments[seg]["train"]) != 48:
            raise AssertionError(f"segment {seg}: expected 48 train exposures")
        if len(segments[seg]["test"]) != 6:
            raise AssertionError(f"segment {seg}: expected 6 distinct test mappings")

    started = time.time()
    matrix: list[list[float]] = []
    losses: list[list[float]] = []
    diagonal: list[float] = []

    for seg in range(6):
        segment_losses = []
        for ex in segments[seg]["train"]:
            if ex.split != "train":
                raise AssertionError("held-out example entered adaptation")
            segment_losses.append(
                supervised_step(
                    model,
                    tokenizer,
                    [ex],
                    optimizer,
                    budget,
                    use_fast=True,
                    use_slow=False,
                    use_latent=False,
                    update_latent=False,
                )
            )
        losses.append(segment_losses)

        row = [float("nan")] * 6
        for old in range(seg + 1):
            row[old] = score_mappings(model, tokenizer, segments[old]["test"], candidates)
        matrix.append(row)
        diagonal.append(row[seg])

    expected_steps = 6 * 48
    if budget.optimizer_steps != expected_steps:
        raise AssertionError(f"expected {expected_steps} optimizer steps, got {budget.optimizer_steps}")
    if budget.training_examples_processed != expected_steps:
        raise AssertionError(
            f"expected {expected_steps} processed examples, got {budget.training_examples_processed}"
        )
    if budget.parameter_write_units != expected_steps * plastic_params:
        raise AssertionError("single-adapter write accounting mismatch")

    all_losses = [x for segment in losses for x in segment]
    finite_losses = all(torch.isfinite(torch.tensor(all_losses)).tolist())
    finite_params = all(torch.isfinite(p.detach()).all().item() for p in params)
    backbone_frozen = all(
        not p.requires_grad
        for name, p in model.named_parameters()
        if ".adapters." not in name
    )
    backbone_gradients_absent = all(
        p.grad is None
        for name, p in model.named_parameters()
        if ".adapters." not in name
    )

    result = {
        "protocol": "EXP-001R-sequential-interference-gate-v1",
        "status": "DEVELOPMENT_ONLY",
        "seed": seed,
        "selected_fast_rank": rank,
        "single_adapter_effective_rank": 2 * rank,
        "lr": lr,
        "plastic_parameter_count": plastic_params,
        "matrix": matrix,
        "diagonal": diagonal,
        "mean_diagonal_acquisition": sum(diagonal) / len(diagonal),
        "average_forgetting": average_forgetting(matrix),
        "pre_segment3_diagonal": diagonal[:3],
        "sequential_rules_out_all_arm_ceiling": any(v <= 0.95 for v in diagonal[:3]),
        "loss_trajectory_by_segment": losses,
        "finite_losses": finite_losses,
        "finite_adapter_parameters": finite_params,
        "backbone_frozen": backbone_frozen,
        "backbone_gradients_absent": backbone_gradients_absent,
        "budget": {
            "optimizer_steps": budget.optimizer_steps,
            "parameter_write_units": budget.parameter_write_units,
            "training_examples_processed": budget.training_examples_processed,
            "tokens_processed": budget.tokens_processed,
        },
        "model_name": cfg.model_name,
        "requested_model_revision": cfg.model_revision,
        "loaded_model_revision": getattr(model.base.config, "_commit_hash", None),
        "loaded_tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "model_init_seed": model_init_seed,
        "device": device,
        "device_numerics": device_report,
        "git_sha": git_sha(),
        "source_tree_sha256": source_tree_sha256(),
        "elapsed_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"seed-{seed}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    return result


def summarize(results: list[dict], *, rank: int, lr: float, representation_summary: dict) -> dict:
    mean_diagonal = sum(float(r["mean_diagonal_acquisition"]) for r in results) / len(results)
    forgetting = [float(r["average_forgetting"]) for r in results]
    acquisition_pass = mean_diagonal >= 0.667
    interference_pass = any(v >= 0.10 for v in forgetting)
    anti_ceiling_pass = any(bool(r["sequential_rules_out_all_arm_ceiling"]) for r in results)
    numerics_pass = all(
        r["finite_losses"]
        and r["finite_adapter_parameters"]
        and r["backbone_frozen"]
        and r["backbone_gradients_absent"]
        for r in results
    )
    both_gates_pass = acquisition_pass and interference_pass and anti_ceiling_pass and numerics_pass

    if not acquisition_pass:
        disposition = "return_to_representation_sufficiency"
    elif not interference_pass:
        disposition = "difficulty_or_interference_calibration_allowed"
    elif not anti_ceiling_pass:
        disposition = "ceiling_risk_requires_review"
    elif not numerics_pass:
        disposition = "implementation_or_numerics_failure"
    else:
        disposition = "resume_exp001_phase_b_multi_arm_development"

    return {
        "protocol": "EXP-001R-sequential-interference-gate-v1",
        "status": "DEVELOPMENT_ONLY",
        "representation_protocol": representation_summary.get("protocol"),
        "selected_fast_rank": rank,
        "single_adapter_effective_rank": 2 * rank,
        "selected_lr": lr,
        "seed_mean_diagonal_acquisition": [r["mean_diagonal_acquisition"] for r in results],
        "mean_diagonal_acquisition": mean_diagonal,
        "seed_average_forgetting": forgetting,
        "acquisition_rule": ">= 0.667 mean diagonal/new-task acquisition",
        "interference_rule": ">= 0.10 average forgetting in at least one development seed",
        "anti_ceiling_rule": "sequential provides a counterexample to all-adaptive-arms >0.95 before segment 3",
        "acquisition_pass": acquisition_pass,
        "interference_pass": interference_pass,
        "anti_ceiling_pass": anti_ceiling_pass,
        "numerics_and_freeze_pass": numerics_pass,
        "both_gates_pass": both_gates_pass,
        "disposition": disposition,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--representation-summary",
        default="results/exp001r-representation/summary.json",
        help="Must be the passing representation summary; this script refuses to run otherwise.",
    )
    ap.add_argument("--output-dir", default="results/exp001r-sequential")
    args = ap.parse_args()

    rank, lr, representation_summary = load_selected(ROOT / args.representation_summary)
    output_dir = ROOT / args.output_dir
    results = []
    for seed in DEV_SEEDS:
        print(f"[EXP-001R sequential] seed={seed} rank={rank} lr={lr:g}", flush=True)
        results.append(run_seed(seed=seed, rank=rank, lr=lr, device=args.device, output_dir=output_dir))

    summary = summarize(results, rank=rank, lr=lr, representation_summary=representation_summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
