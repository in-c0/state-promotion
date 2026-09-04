"""Deterministic development-seed derivation for EXP-001 v2.

Issue #6 section 1 retires the v1 development seeds 20260901/02/03 from tuning,
because B5-vs-baseline ordering has already been inspected on them. The v2 seeds
are derived from a fixed label so the list in the issue is reproducible rather
than asserted.

Each seed is one 4-byte big-endian word of SHA-256(label), mapped into the
8-digit range [10_000_000, 99_999_999]. The offset keeps every seed the same
width, so no seed can collapse to a short value by accident.
"""
from __future__ import annotations

import hashlib

V2_DEVELOPMENT_LABEL = "EXP-001-v2-consolidation-development-v1"
V2_DEVELOPMENT_SEED_COUNT = 5

# v3 (issue #7): the v2 seeds are retired too -- latent diagnostics were
# inspected on three of them, so they are no longer clean for tuning.
V3_DEVELOPMENT_LABEL = "EXP-001-v3-embedding-latent-development-v1"
V3_DEVELOPMENT_SEED_COUNT = 5

SEED_SPAN = 90_000_000
SEED_FLOOR = 10_000_000

# Retired from v2 tuning: arm ordering was inspected on these in Phase-B v1.
V1_DEVELOPMENT_SEEDS = (20260901, 20260902, 20260903)


def derive_seeds(label: str, count: int) -> tuple[int, ...]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    if count * 4 > len(digest):
        raise ValueError(f"cannot derive {count} seeds from a {len(digest)}-byte digest")
    return tuple(
        (int.from_bytes(digest[i * 4:(i + 1) * 4], "big") % SEED_SPAN) + SEED_FLOOR
        for i in range(count)
    )


def v2_development_seeds() -> tuple[int, ...]:
    return derive_seeds(V2_DEVELOPMENT_LABEL, V2_DEVELOPMENT_SEED_COUNT)


def v3_development_seeds() -> tuple[int, ...]:
    return derive_seeds(V3_DEVELOPMENT_LABEL, V3_DEVELOPMENT_SEED_COUNT)


V2_DEVELOPMENT_SEEDS = v2_development_seeds()
V3_DEVELOPMENT_SEEDS = v3_development_seeds()

# Every seed already burned on inspected performance or diagnostics.
RETIRED_DEVELOPMENT_SEEDS = V1_DEVELOPMENT_SEEDS + V2_DEVELOPMENT_SEEDS
