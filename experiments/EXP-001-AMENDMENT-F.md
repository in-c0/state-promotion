# EXP-001 pre-result Amendment F — immutable paired model snapshot

**Status:** locked before any EXP-001 language-model score is inspected.

## Problem discovered in execution review

The pilot runner named `Qwen/Qwen2.5-0.5B-Instruct` but did not require an immutable Hugging Face revision. Each arm could therefore resolve the moving repository ref independently. Even if unlikely over a short pilot, repository drift would break the paired-method assumption and weaken exact reproducibility.

## Locked rule

1. The full pilot orchestrator resolves the requested model ref to one immutable Hugging Face snapshot SHA **before the first arm runs**.
2. That exact SHA is passed to both tokenizer and model loading for every arm, including frozen and the post-hoc commit-count-matched random control.
3. Every run manifest records the pinned `snapshot_revision` plus any revision reported by the loaded model/tokenizer objects.
4. The validator rejects a missing snapshot pin, mixed snapshot SHAs across pilot arms, or a reported model/tokenizer revision that conflicts with the pin.
5. Development pilots may establish which immutable snapshot becomes the confirmatory model revision, but confirmatory protocol v1.0 must freeze it explicitly.

This amendment changes provenance control only. It does not alter PALS examples, adaptation rules, routing thresholds, resource budgets, or H1.
