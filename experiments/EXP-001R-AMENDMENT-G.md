# EXP-001R Amendment G — gate runners must use the protocol's presentation order

**Status:** locked before any EXP-001R held-out or comparative arm score is inspected.
The only scores that motivated this amendment are development-seed acquisition
numbers from the EXP-001R representation gate itself, which is a development
instrument by construction.

## Defect

`scripts/run_exp001r_representation.py` and `scripts/run_exp001r_sequential.py`
iterated each segment's train examples in PALS generation order. PALS emits
`train_repeats` consecutive copies of each mapping, so generation order is
maximally blocked: for retention segment 0 the key order is

    [(PORIN,8), (LISIV,8), (DALOR,8), (RUQEN,8), (SUPAX,8), (KIDUL,8)]

The protocol runner does not train in that order. `scripts/run_lm_pals.py`
constructs `rng = random.Random(seed)` once per run and calls `rng.shuffle(train)`
on a copy of each segment's train list, in ascending segment order.

Both gate runners therefore measured acquisition under a training regime the
protocol never uses.

The sequential gate carried the identical defect at the time this amendment was
locked, and had never been executed. Its interference number would have been
inflated in the same direction, because blocked order maximises interference.
The defect is therefore repaired in both runners before either produces a result
that is relied upon.

## Evidence that this is a confound, not a capacity result

This failure mode was diagnosed and quantified independently, before the gate
grid was executed, in commit `f53d402` on `exp001/dynamic-input-and-snapshot-pin`:
a `--no-shuffle` control isolated it exactly, with prompt-8tok @5e-3 scoring
1.00/6 under blocked order against 5.67/6 shuffled. `f53d402` is not an ancestor
of `47d2576`; EXP-001R branched from `5d66242`, one commit earlier, and never
inherited the correction.

The first EXP-001R grid run reproduced the same signature mechanically. In every
cell scoring exactly 1/6, the single correct mapping was the last block trained
(`KIDUL` for seed 20260901, `SUSIV` for seed 20260902) — recency collapse onto
the most recently seen mapping, not a failure to represent the mapping set.

That run is preserved unmodified at
`results/exp001r-representation-blocked-order-defect/` and
`logs/exp001r-representation-blocked-order-defect.log`, with its summary annotated
`superseded: true`. It is not deleted, not rerun in place, and not reinterpreted
as evidence about LoRA capacity.

## Locked rule

1. Every EXP-001R gate runner that adapts on a PALS segment must present that
   segment's train examples in the same order discipline as the protocol runner:
   one `random.Random(seed)` per run, `shuffle` applied to a copy of each
   segment's train list, segments consumed in ascending order.
2. A regression test pins this: gate-runner presentation order must equal the
   order `run_lm_pals.py` produces for the same seed. A runner that trains in
   raw generation order fails the suite.
3. The representation grid's predeclared cells (ranks 1/2/4 x LRs 3e-4/1e-3/3e-3),
   development seeds (20260901/02/03), pass rule (mean >= 5/6 and worst-seed
   >= 4/6, finite), and selection rule (smallest passing rank; within rank
   highest worst-seed; ties to smaller LR) are **unchanged** by this amendment.
   No threshold is loosened. A failing gate is not made to pass by weakening its
   criterion; it is re-measured under the presentation order the protocol
   actually specifies. The sequential gate's acquisition, interference, anti-ceiling and
   numerics rules and its disposition mapping are **unchanged**.
4. The grid is re-run in full after the fix. The superseded run remains on record.

This amendment changes presentation order only, to match the preregistered
protocol. It does not alter PALS examples, adaptation rules, promotion
thresholds, routing, resource budgets, gate thresholds, or H1.

## Scope note

The score-free structural smoke (`scripts/smoke_exp001r_qwen.py`) could not have
detected this: it verifies structural invariants and scores zero held-out
examples, so it never exercises the acquisition path. That remains the correct
design for the smoke; this class of defect is caught by rule 2 above instead.
