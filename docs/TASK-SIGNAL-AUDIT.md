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

Before confirmatory seeds, inspect one serialized training batch and one held-out evaluation batch per arm and archive their hashes/manifests. Confirm that no method-specific code path adds segment IDs or held-out targets to the model input.
