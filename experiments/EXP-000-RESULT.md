# EXP-000 result — engineering smoke test

**Classification:** PILOT / engineering validation only. Not evidence for or against EXP-001 H1.

**Seeds:** 0–11

| Method | Final avg | Avg forgetting | Plasticity | Retention AUC |
|---|---:|---:|---:|---:|
| Sequential | 0.366 ± 0.060 | 0.761 ± 0.072 | 1.000 ± 0.000 | 0.548 ± 0.036 |
| Replay | 0.459 ± 0.058 | 0.251 ± 0.062 | 0.625 ± 0.054 | 0.616 ± 0.046 |
| Fixed | 0.459 ± 0.058 | 0.251 ± 0.062 | 0.625 ± 0.054 | 0.616 ± 0.046 |
| Promotion | 0.457 ± 0.057 | 0.255 ± 0.057 | 0.625 ± 0.054 | 0.614 ± 0.047 |

## Interpretation

1. **Sequential new-task performance is at ceiling (1.000).** Per the preregistered methodological principle, this task family cannot adjudicate adaptation-speed differences in its current form.
2. **Replay and fixed consolidation are identical by construction.** The toy uses additive linear fast/slow residuals; `slow <- slow + fast; fast <- 0` preserves the exact function. Therefore the fixed-consolidation arm is not a meaningful architectural intervention.
3. **Promotion is tied with replay within pilot variance.** There is no evidence here for the research hypothesis, and none should be claimed.
4. The harness successfully computes a retention matrix, forgetting, plasticity, replay behavior, write counts, and guarded commit/rollback paths.

## Changes required before EXP-001

- Use a non-ceiling stream whose acquisition speed remains measurable.
- Slow consolidation must involve a distinct update objective/timescale (e.g. replay/distillation into slow adapter parameters), not an exact algebraic merge.
- Explicitly match write volume as well as optimizer steps.
- Retain a fixed/random routing control matched to the promotion arm's commit count.
- Treat EXP-000 as a software test only; do not tune EXP-001 thresholds against its metric values.
