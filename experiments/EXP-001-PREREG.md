# EXP-001 preregistration — State Promotion on a small language model

**Status:** DRAFT LOCK candidate. This document must be committed before any confirmatory EXP-001 result is inspected.

## Question

Under matched resource budgets, does evidence-gated promotion across multiple state timescales improve the stability–plasticity trade-off of continual language-model adaptation?

## Architecture under test

Frozen base model (initial target: `Qwen/Qwen2.5-0.5B-Instruct`) plus:

1. **Persistent latent state** `z_t`: bounded-size recurrent state carried across stream items/sessions.
2. **Fast adapter** `F_t`: online plastic parameters trained on recent evidence.
3. **Slow adapter bank** `S`: durable parameterized knowledge.
4. **Episodic replay** `E`: bounded reservoir storing attributable training examples/outcomes.
5. **Promotion gate**: decides whether candidate fast-state learning is committed to slow parameters.
6. **Retention gate + rollback**: consolidation is rejected if protected behavior regresses beyond the preregistered tolerance.

The frozen foundation weights are never modified in EXP-001.

## Primary hypothesis H1

At an equalized trainable-parameter budget, replay-memory budget, online-token budget, optimizer-step/write budget, and approximately matched compute, **State Promotion** will achieve lower average forgetting than sequential adaptation and fixed-schedule consolidation while retaining at least 95% of the best baseline's new-task plasticity.

Primary success criterion:

- statistically lower mean average forgetting than the strongest budget-matched baseline; and
- mean diagonal/new-task score >= 0.95 × the strongest baseline diagonal/new-task score.

We will report effect sizes and confidence intervals, not only p-values.

## Secondary hypothesis H2

Persistent latent state contributes specifically to context-dependent exceptions/reversals. Removing or resetting `z_t` should reduce performance on the PALS context/reversal subset without materially changing the number of parameter writes.

## Baselines

B0. Frozen base.

B1. Sequential LoRA/adapter: one continuously updated adapter, no replay.

B2. Sequential adapter + bounded replay.

B3. Two-timescale adapter + replay + **fixed** consolidation schedule.

B4. Two-timescale adapter + replay + **random** consolidation schedule matched to B5's number of commits (post-hoc matched only for the routing ablation; not used for the primary online comparison).

B5. **State Promotion**: latent state + evidence-gated consolidation + retention test/rollback.

## Budget matching

Before a run can count as confirmatory:

- same frozen base and tokenizer;
- same trainable parameter count within ±2%, or report a parameter-efficiency curve;
- same number of stream examples/tokens;
- same replay capacity in bytes/examples;
- optimizer steps within ±2%;
- parameter-write events explicitly counted;
- total measured training wall time and estimated FLOPs reported;
- no method receives task IDs unless every method receives them.

A run violating one of these is **invalid for H1** and may only be reported as a pilot.

## Data

### PALS-v0: Persistent Adaptation Learning Stream

A procedurally generated stream of nonce-symbol associations, context-specific rules, exceptions, and explicit reversals. Generation seeds are held out from model development.

Design constraints:

- nonce terms minimize pretraining contamination;
- test examples are disjoint from update examples;
- tasks contain both related and conflicting mappings;
- some rules are global, some scoped to a context, and some later superseded;
- evaluation distinguishes recall, contextual exception handling, and reversal updating.

### External benchmark

After PALS-v0 is stable and non-ceiling, run a compatible subset of CITB and/or TRACE with documented preprocessing. These are secondary in EXP-001 if compute is restrictive; they become mandatory before a broad continual-learning claim.

## Evaluation matrix

After learning stream segment/task `i`, evaluate all tasks/segments `j <= i` and record `R[i,j]`.

Average forgetting for task `j`:

`F_j = max_{i >= j} R[i,j] - R[T,j]`

Report:

- final average score;
- average forgetting;
- diagonal/new-task plasticity;
- backward transfer;
- retention AUC across the stream;
- update sample efficiency;
- state/adaptor/replay memory bytes;
- optimizer steps, parameter writes, and measured runtime.

## Seeds and statistics

Pilot stage: unrestricted and explicitly labeled PILOT.

Confirmatory EXP-001:

- minimum 5 independent seeds for the 0.5B language model;
- use paired seeds/task streams across methods;
- bootstrap 95% CIs for primary differences;
- paired permutation or paired t-test only after inspecting distributional appropriateness;
- report all seeds, including failed but valid runs.

Scale to >=12 seeds if variance makes the primary interval inconclusive and compute permits.

## Invalidation / kill criteria

A benchmark configuration is invalid for the primary claim if any of the following occurs:

1. All adaptive arms exceed 95% on the primary score before the third stream segment (ceiling).
2. A method sees leaked task identity or held-out answers unavailable to others.
3. Parameter count, write count, optimizer steps, or replay budget violate matching tolerance.
4. Evaluation depends on non-deterministic generation without enough repeats to estimate variance.
5. The promotion arm's apparent advantage disappears when consolidation count is matched by a random/fixed routing control.
6. Results depend on a single task ordering.

If H1 is falsified under valid conditions, publish the negative result and proceed only with a revised hypothesis that is clearly marked post-hoc.

## Required ablations

- no persistent latent state;
- latent state reset at each task boundary;
- fixed consolidation instead of evidence gate;
- evidence gate without retention rollback;
- replay removed;
- slow consolidation removed;
- matched random consolidation schedule.

## Novelty boundary

We do **not** claim novelty for fast/slow weights, external memory, replay, state commitment, frozen backbones, modular adapters, or consolidation individually.

The intended contribution is the experimentally controlled question of whether **evidence-gated promotion across ephemeral/latent/fast/slow state, with guarded commitment and explicit write-budget matching, improves continual adaptation**.

Relevant adjacent work must include Titans, Nested Learning/Hope, Memini, State Commitment Learning, SCALE, Harness Continual Learning, and Memoir.

## Publication policy

- code and configs public before submission when feasible;
- every paper table traceable to machine-readable run manifests;
- no cherry-picking seeds;
- negative results published;
- claims restricted to tested scales and benchmarks;
- 7B results described as replication/scale evidence, not substituted for small-model ablations.

## PALS metric clarification (pre-run addendum)

PALS separates two questions that ordinary task-retention matrices conflate:

1. **Retention stream:** old mappings remain valid forever. Forgetting is harmful and contributes to H1.
2. **Revision stream:** a later event may explicitly supersede an earlier mapping for the same context/key pair. Retaining the obsolete answer is harmful, not a success.

Therefore H1 is computed on the stable retention stream. The revision stream reports separately:

- active-world accuracy (latest valid mapping);
- context-exception accuracy (simultaneously valid context-specific mappings);
- supersession accuracy (ability to replace obsolete state);
- stale-answer rate (probability/selection rate of the superseded answer).

We will not count desired forgetting of superseded facts as catastrophic forgetting.
