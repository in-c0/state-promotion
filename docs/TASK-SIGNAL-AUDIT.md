# EXP-001 task-signal audit

This note records what task/stream metadata is available to each method before any 0.5B result is inspected.

## Model-visible input

PALS prompts expose only the natural task content: a context nonce and a key nonce. They do **not** contain:

- stream name;
- segment/task index;
- train/test split;
- relation label (`durable`, `context_exception`, `supersedes`);
- version number;
- future target mappings.

The supervised update target is visible only as the label for the current training example. Held-out test answers are never included in adaptation batches. `tests/test_pals.py::test_prompt_does_not_expose_hidden_task_metadata` guards the prompt contract.

## Promotion-gate evidence

The promotion arm is **not** allowed to inspect held-out evaluation labels. This was explicitly corrected before any EXP-001 LM result was inspected.

- Current-task probes are deduplicated examples from the current segment's already-observed `train` split.
- Protected probes come from the bounded replay reservoir and therefore contain only previously observed training evidence.
- On revision streams, replay entries whose context/key mapping has been superseded by a later observed training event are not treated as protected knowledge.
- Every promotion result records probe hashes/counts and `heldout_gate_example_count`; confirmatory validation requires that count to be zero.
- Promotion-gate forward passes are separately counted as decision-time inference compute because they influence adaptation behavior.

`tests/test_promotion_probes.py` guards these properties at the stream/harness layer, and `guarded_promotion` rejects any non-training probe at runtime.

## Harness-visible boundary signal

The experiment runner iterates the same segment boundaries for every arm. Segment boundaries are therefore a **shared orchestration signal**, not a private task ID:

- B1 sequential: optimizer is recreated at each shared segment boundary.
- B2 replay: same boundary cadence as B1.
- B3 fixed: slow consolidation occurs at the shared boundary.
- B4 random: a preregistered/matched schedule chooses which shared boundaries consolidate.
- B5 promotion: the evidence gate is evaluated at the same shared boundaries.

The model itself is not passed the integer segment index. The random-control scheduler uses the index only outside the model to instantiate its matched commit schedule.

## Evaluation-only metadata

`stream`, `segment`, `relation`, and `version` are used by the evaluator to construct retention/revision metrics. They are not injected into model prompts. Revision evaluation uses them to determine which historical mapping is currently valid; this is scoring logic, not learning input.

## Remaining leakage checks before confirmatory lock

Before confirmatory seeds, inspect one serialized training batch and one held-out evaluation batch per arm and archive their hashes/manifests. Confirm that no method-specific code path adds segment IDs or held-out targets to the model input. Also verify from every B5 manifest that `heldout_gate_example_count == 0` and that decision-time inference counters are present.

## Continuous-state audit correction (2026-09-01)

The same pre-result audit found that the initial scaffold recreated AdamW at every synthetic segment boundary. That would weaken the preregistered continuously updated B1/B2 baselines for an implementation reason unrelated to continual learning. The runner now persists optimizer state whenever the corresponding parameter state persists, and resets optimizer state only when the fast parameters themselves are reset after an accepted consolidation. Model-side prompt-state initialization is also explicitly paired by seed across methods.

This is recorded as preregistration Amendment C. No EXP-001 language-model score was inspected before this correction.

## Routing-control architecture correction (2026-09-01)

B3 fixed routing and B4 matched-random routing now use the same persistent latent-state mechanism as B5. The earlier scaffold disabled latent state for those controls, which would confound routing effects with access to an extra persistent state channel. `promotion-no-latent` remains the explicit H2 ablation. The run manifest exposes `latent_state_enabled`, and the validator rejects a fixed/random/promotion latent mismatch.
