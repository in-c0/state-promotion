# EXP-001 pre-result Amendment E — decision-time routing compute

**Status:** locked before any EXP-001 language-model score is inspected.

## Problem discovered in static review

B5 State Promotion performs no-gradient forward passes over already-observed evidence to decide whether to consolidate and whether a candidate slow-state update should be retained or rolled back. These forward passes are part of the algorithm, not post-hoc evaluation. Depending on probe size and candidate count, their cost can be material relative to ordinary adaptation training.

The pre-result scaffold already records this cost separately as `decision_forward_calls`, `decision_tokens_processed`, `decision_token_parameter_compute_proxy`, and `estimated_decision_flops_frozen_backbone`, but the phrase "approximately matched compute" in the original H1 could be misread as either ignoring this overhead or requiring unrelated baselines to execute dummy routing passes.

## Locked interpretation

For EXP-001:

1. **Adaptation/training resources are the matched primary budget.** The primary H1 comparison matches frozen base/tokenizer, total plastic capacity, replay capacity, unique online stream exposure, the hard parameter-element write ceiling, and the predeclared adaptation-token training envelope as specified in Amendment A.
2. **Decision-time routing inference is an explicit method overhead, not hidden evaluation.** Every forward pass whose output can change a consolidation/promotion/rollback decision must be counted separately. Post-hoc held-out evaluation remains excluded from algorithmic compute.
3. B1/B2/B3/B4 are not required to execute meaningless dummy decision passes merely to equal B5's routing overhead. Such dummy computation would not strengthen those methods and would obscure the actual systems trade-off.
4. **No B5 result may be described as compute-matched without qualification.** Primary tables must report both adaptation/training compute and decision-time routing compute. A total algorithmic compute view must add the two proxies (or measured equivalents when available).
5. Before confirmatory protocol v1.0 is frozen, development-only calibration must quantify B5 routing overhead. If decision-time compute is large enough to change the practical interpretation, the paper must include a performance-vs-total-compute/resource curve or a stronger compute-augmented baseline rather than relying only on the write-budget comparison.
6. B5 may not increase probe size, candidate count, or routing frequency on confirmatory seeds. Probe construction/routing frequency must be frozen using development seeds only.
7. The validator must require decision-time accounting for promotion-family methods and preserve the distinction between adaptation compute and total algorithmic compute. It should not silently treat zero routing overhead as matched compute.

## Why this does not change H1 after seeing results

No EXP-001 LM score had been generated or inspected when this ambiguity was identified. This amendment clarifies resource accounting; it does not change the stability/plasticity success criterion, benchmark labels, write ceiling, replay capacity, or held-out evaluation metrics.
