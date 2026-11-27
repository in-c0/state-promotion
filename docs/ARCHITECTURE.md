# Architecture note — commit state, don't accumulate everything

The design separates four persistence levels.

| Level | Lifetime | Example | Update mechanism |
|---|---|---|---|
| Ephemeral | one reasoning step/session | scratch computation | normal activations/context |
| Latent | sessions/days | salience, unresolved context | bounded recurrent state |
| Fast parametric | examples/hours | rapidly learned mapping | online adapter updates |
| Slow parametric | weeks/long-lived | stable reusable behavior | gated consolidation |

External episodic memory remains attributable and inspectable; it is not treated as equivalent to latent or parametric memory.

## Promotion rule

An experience is a candidate for deeper persistence only when evidence supports future utility. EXP-001 starts with a deliberately simple gate based on recurrence, prediction error/surprise, current-task acquisition, and protected-task retention. A learned router is later work.

The critical safety/reliability boundary is **candidate != committed state**:

1. Learn/update fast state.
2. Form a consolidation candidate.
3. Evaluate current-task gain and protected-task retention.
4. Commit only if the preregistered gate passes.
5. Otherwise roll back and retain the evidence episodically.

This separates the semantic question "should the system learn this?" from the mechanical fact "a gradient was available."
