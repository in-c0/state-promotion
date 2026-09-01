# State Promotion

**Working research project:** evidence-gated promotion between transient computation, persistent latent state, fast adaptation, and consolidated slow parameters for continual language models.

> Status: **pre-results / experiment scaffold.** No empirical claim is made yet.

## Research question

Can a frozen foundation model acquire new behavior online while retaining old behavior better than conventional continual adaptation when all methods are matched for trainable parameter budget, replay memory, parameter-write budget, and compute?

The specific hypothesis is that **where and when experience is committed** matters. A useful continual learner should not treat every observation as equally worthy of durable parameter change.

## Proposed state hierarchy

```text
observation
   |
   v
[ephemeral computation]
   |
   | evidence / recurrence / utility
   v
[persistent latent state]
   |
   | promotion gate
   v
[fast plastic parameters]
   |
   | retention-tested consolidation
   v
[slow modular parameters]
   |
   v
[frozen foundation core]
```

The initial experiment deliberately tests only the smallest falsifiable subset of this architecture.

## Experiment sequence

- **EXP-000 — smoke test:** tiny PyTorch continual classifier. Validates the experiment harness, retention matrix, rollback, and write accounting. **Not paper evidence.**
- **EXP-001 — small language model:** frozen ~0.5B model + matched adapter baselines on a controlled synthetic continual stream.
- **EXP-002 — established benchmarks:** CITB / TRACE-compatible evaluation.
- **EXP-003 — scale replication:** repeat the strongest preregistered comparison at ~7B.

See [`experiments/EXP-001-PREREG.md`](experiments/EXP-001-PREREG.md).

## Baselines

1. Frozen model.
2. Sequential plastic adapter.
3. Plastic adapter + replay.
4. Fixed-schedule fast/slow consolidation + persistent latent state.
5. **State Promotion:** evidence-gated fast/slow consolidation + the same persistent latent state.

The crucial comparisons must match:

- base model;
- trainable parameter count;
- online examples/tokens;
- allowed parameter writes;
- replay bytes;
- inference/training FLOPs as closely as practical.

## Primary metrics

- average accuracy / task score;
- average forgetting;
- backward transfer;
- forward transfer where defined;
- acquisition speed / plasticity;
- retention area under the learning curve;
- parameter writes and wall-clock/compute;
- memory footprint.

## Related work that constrains the novelty claim

This project is intentionally positioned as an extension/combination rather than claiming that multi-timescale learning itself is new.

- Behrouz et al., **Titans: Learning to Memorize at Test Time** (2025), arXiv:2501.00663.
- Behrouz et al., **Nested Learning / Hope** (NeurIPS 2025).
- Pattichis & Dovrolis, **Continual Knowledge Updating in LLM Systems: Learning Through Multi-Timescale Memory Dynamics** (2026), arXiv:2605.05097.
- Ding et al., **State commitment learning: training language models to distinguish computation from memory** (2026), arXiv:2606.05201.
- Lee et al., **SCALE: Upscaled Continual Learning of Large Language Models** (ACL Findings 2026), arXiv:2511.03270.
- Kang et al., **Harness Continual Learning: Continual Adaptation Beyond Model Parameters** (2026), arXiv:2608.19013.
- RightNow-AI, **Memoir** (2026), arXiv:2607.20792 — especially relevant because its preregistered fast-write coupling test produced a negative result and exposed write-volume/ceiling confounds.

## Reproducibility policy

- Preregister primary hypotheses and invalidation criteria before paper-result runs.
- Publish negative results.
- Keep pilot/smoke results separate from confirmatory runs.
- Match budgets before interpreting architecture effects.
- Treat ceiling effects, task leakage, and unequal write counts as invalidating confounds.
- Never use held-out evaluation labels for promotion/rollback decisions; B5 gates only on already-observed training/replay evidence.
- Count promotion-gate forward passes as decision-time inference compute.
- Preserve optimizer state whenever the corresponding continual parameter state persists; orchestration segment boundaries are not learning resets.
- Architecture-match persistent latent state across fixed, random, and learned-routing two-timescale controls.
- Pair model initialization seeds across methods within each lifetime seed.
- Record a deterministic source-tree fingerprint for archive pilots and require an actual code commit before confirmatory runs.
- Keep development/calibration seeds disjoint from confirmatory seeds.
- Record seeds and complete configs for every reported result.

## Local gates

```bash
make test
make pals
# Requires transformers + a locally available/downloadable Qwen2.5-0.5B-Instruct:
make lm-pilot
```

`make lm-pilot` runs the complete retention pilot (B0/B1/B2/B3/B5, derives the B4 commit-matched random control, then validates all manifests). `make lm-pilot-revision` runs the corresponding revision stream.

## Quick smoke test

```bash
python scripts/run_toy.py --seeds 0 1 2 3
```

The toy experiment uses only PyTorch and exists to validate mechanics, not to establish the research claim.

## License

Apache-2.0.
