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
5. **Promotion gate**: decides whether candidate fast-state learning is committed to slow parameters using only already-observed attributable training/replay evidence.
6. **Retention gate + rollback**: consolidation is rejected if protected replay-derived probes regress beyond the preregistered tolerance. Held-out evaluation examples are never available to this decision.

The frozen foundation weights are never modified in EXP-001.

## Primary hypothesis H1

At an equalized trainable-parameter capacity, replay-memory budget, online-token budget, **parameter-write budget ceiling**, and approximately matched adaptation-token compute envelope, **State Promotion** will achieve lower average forgetting than sequential adaptation and fixed-schedule consolidation while retaining at least 95% of the best baseline's new-task plasticity.

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
- same total plastic parameter capacity within ±2%, or report a parameter-efficiency curve;
- same number of unique stream examples/tokens;
- same replay capacity in bytes/examples for replay-assisted arms;
- same hard **parameter-write budget ceiling** within ±2%, measured as the number of parameter elements exposed to optimizer writes;
- optimizer steps are reported but are not required to be identical when methods update different-sized parameter subsets per step;
- adaptation-token exposure is explicitly counted, and the strong replay baseline versus two-timescale arms must be compared under the same predeclared compute envelope or on a resource/performance curve;
- total measured training wall time and a documented compute proxy / FLOP estimate are reported;
- no method receives task IDs unless every method receives them.

A run violating one of these is **invalid for H1** and may only be reported as a pilot.

### Pre-result amendment A — write-budget normalization (2026-09-01)

This amendment was made **before any EXP-001 language-model score was inspected**. The original draft required both optimizer-step counts and parameter-write counts to match within ±2%. That is over-constrained when B1/B2 update a single adapter spanning the full plastic capacity while B3/B5 update only the fast or slow subset on a given step: equal optimizer steps would imply unequal parameter writes, while equal writes would imply unequal optimizer steps.

For EXP-001, the primary resource normalization is therefore:

1. Let one write unit equal the number of elements in the fast-state parameter subset.
2. Give every adaptive arm the same hard write ceiling, nominally **2T write units** for a stream containing `T` unique online update events.
3. B1/B2 may update the full single-adapter capacity (2 units) once per online event, consuming the full 2T ceiling.
4. B3/B5 update the fast subset (1 unit) on each online event, leaving at most T units for slow consolidation.
5. For the strong replay comparison, B2 uses replay in the online update batch, whereas B3/B5 reserve replay for slow consolidation; adaptation-token exposure is counted so that compute can be matched or plotted as a resource curve.
6. B5 is allowed to consume **less** than the write ceiling when its evidence gate declines promotion. That lower consumption is part of the mechanism under test, not a reason to add dummy writes.
7. B4 random routing is matched post-hoc to B5's accepted commit count and per-commit consolidation allocation, as preregistered.
8. Any method exceeding the write ceiling is invalid. Actual writes, optimizer steps, replay examples, processed adaptation tokens, and wall time are all reported.

This amendment changes only resource normalization; H1's stability/plasticity success criterion and the benchmark invalidation rules remain unchanged.

### Pre-result amendment B — no held-out evidence in promotion decisions (2026-09-01)

This amendment was made **before any EXP-001 language-model score was inspected**. An engineering audit found that an earlier scaffold passed held-out `test` examples into the promotion/retention gate. That would give B5 privileged access to evaluation labels and invalidate the comparison even if those examples were not used for gradient updates.

For every EXP-001 run after this amendment:

1. Held-out evaluation examples are used only for post-update scoring and never for routing, promotion, rollback, optimizer selection, or replay construction.
2. The current-task promotion probe is formed from already-observed training examples in the current segment, deduplicated by active context/key/target mapping.
3. The protected retention probe is formed only from the bounded replay reservoir. On revision streams, replay entries made obsolete by an observed superseding training event are excluded from protection.
4. Every promotion run records probe counts and hashes plus a machine-checkable `heldout_gate_example_count`, which must be zero.
5. No-grad forward passes used by the promotion gate are counted separately as **decision-time inference compute**. They are not hidden inside post-hoc evaluation cost.

This amendment fixes an information-leak confound in the scaffold. It does not alter H1, the parameter-write normalization, or the held-out evaluation metrics.

### Pre-result amendment C — continuous optimizer state (2026-09-01)

This amendment was made **before any EXP-001 language-model score was inspected**. An implementation audit found that the scaffold recreated AdamW at every stream-segment boundary. That makes a nominally continuous B1/B2 adapter weaker than intended because its first- and second-moment state is repeatedly discarded for orchestration reasons rather than for a mechanism defined by the condition.

For every EXP-001 run after this amendment:

1. B1/B2 preserve optimizer state across stream-segment boundaries for the continuously updated single adapter.
2. Fast-state conditions preserve the fast optimizer while the same fast parameter state remains active.
3. Whenever the fast parameter state is explicitly reset after accepted consolidation, its optimizer state is reset at the same boundary.
4. A rejected/declined promotion does not reset the fast optimizer, because the fast parameter state itself remains the active online state.
5. Model initialization is seeded per paired lifetime/condition so all methods in a paired seed begin from identical prompt-state initialization, while different confirmatory seeds remain independent.

This amendment strengthens fidelity to the preregistered baselines and removes a segment-boundary artifact. It does not alter H1 or any evaluation target.

### Pre-result amendment D — architecture-matched latent state for routing controls (2026-09-01)

This amendment was made **before any EXP-001 language-model score was inspected**. A static condition audit found that the recovered B3 fixed-routing implementation disabled persistent latent state while B5 enabled it. That would confound the principal B5-vs-B3 routing comparison: a difference could be attributed either to learned promotion or simply to B5 receiving an additional persistent state channel.

For every EXP-001 run after this amendment:

1. B3 fixed routing, B4 random routing, and B5 learned promotion all use the same persistent latent-state mechanism and update rule.
2. Their fast/slow prompt capacities, latent dimensionality, replay capacity, and initialization pairing are architecture matched.
3. The primary B5-vs-B3/B4 interpretation is therefore about **routing/consolidation policy**, subject to the remaining resource and compute controls.
4. `promotion-no-latent` remains the explicit H2 ablation for measuring the contribution of persistent latent state to the proposed method.
5. Run manifests record whether latent state is enabled, and the validator rejects a fixed/random/promotion comparison with a latent-state mismatch.

This amendment removes an architectural confound; it does not change H1's success criterion.

## Pre-confirmatory calibration boundary

EXP-001 must not move from PILOT to confirmatory status until a development-only calibration phase is complete and frozen. Development seeds and confirmatory seeds must be disjoint. The calibration phase may be used only to:

- verify that PALS is non-ceiling and produces measurable interference;
- choose/freeze the fixed-routing cadence from a small predeclared cadence set rather than retaining a knowingly weak fixed baseline;
- choose/freeze B5 gate thresholds and learning-rate values; and
- establish a common compute envelope that does not systematically handicap the strongest replay or fixed-routing baseline.

After this boundary is frozen, confirmatory seeds may not be used for threshold/cadence tuning. Any post-lock change creates a new protocol version and the affected runs are not confirmatory evidence for the previous version.

Before the first confirmatory run, also freeze and record:

- immutable model-weight and tokenizer revisions;
- an exact dependency/environment lock;
- a code commit SHA (the source-tree SHA-256 is a fallback pilot provenance fingerprint, not a substitute for a confirmatory commit); and
- the PALS generator revision plus the disjoint confirmatory seed list or a committed seed-selection rule.

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
3. The parameter-capacity, write-ceiling, data/replay-budget, or compute-envelope rules in **Budget matching** are violated.
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
