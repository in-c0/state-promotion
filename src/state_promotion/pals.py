from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable


SYLLABLES_A = ["za", "ve", "ki", "mo", "ru", "ta", "ne", "po", "li", "su", "da", "fi"]
SYLLABLES_B = ["rin", "vek", "tal", "mos", "qen", "pax", "dul", "zim", "lor", "fen", "gur", "siv"]


@dataclass(frozen=True)
class Example:
    stream: str
    segment: int
    context: str
    key: str
    target: str
    split: str
    relation: str = "durable"
    version: int = 1

    @property
    def prompt(self) -> str:
        return (
            "You are maintaining a learned codebook. "
            f"In context {self.context}, what is the current code for {self.key}? "
            "Answer with only the code.\nAnswer:"
        )

    def record(self) -> dict:
        d = asdict(self)
        d["prompt"] = self.prompt
        return d


def _nonce(rng: random.Random, used: set[str]) -> str:
    while True:
        word = (rng.choice(SYLLABLES_A) + rng.choice(SYLLABLES_B)).upper()
        if word not in used:
            used.add(word)
            return word


def _codes(rng: random.Random, n: int) -> list[str]:
    used: set[str] = set()
    return [_nonce(rng, used) for _ in range(n)]


def generate_retention_stream(seed: int, segments: int = 6, items: int = 6,
                              train_repeats: int = 8, test_repeats: int = 2) -> tuple[list[Example], list[str]]:
    """Stable context-conditioned mappings used for the primary retention test.

    The same nonce keys recur across contexts with independent code permutations,
    creating interference without making old mappings obsolete.
    """
    rng = random.Random(seed)
    used: set[str] = set()
    keys = [_nonce(rng, used) for _ in range(items)]
    codes = _codes(rng, items)
    contexts = [_nonce(rng, used) for _ in range(segments)]
    out: list[Example] = []
    for seg, context in enumerate(contexts):
        mapping = codes.copy()
        rng.shuffle(mapping)
        for key, code in zip(keys, mapping):
            for _ in range(train_repeats):
                out.append(Example("retention", seg, context, key, code, "train"))
            for _ in range(test_repeats):
                out.append(Example("retention", seg, context, key, code, "test"))
    return out, codes


def generate_revision_stream(seed: int, items: int = 6, train_repeats: int = 8,
                             test_repeats: int = 2) -> tuple[list[Example], list[str]]:
    """Context exceptions plus explicit supersession.

    Segments 0 and 1 establish two simultaneously valid contexts. Segment 2
    revises half of context 0. Old labels for those exact context/key pairs become
    obsolete and should *not* count as knowledge worth retaining.
    """
    rng = random.Random(seed + 99173)
    used: set[str] = set()
    keys = [_nonce(rng, used) for _ in range(items)]
    codes = _codes(rng, items)
    ctx_a, ctx_b = _nonce(rng, used), _nonce(rng, used)
    map_a = codes.copy(); rng.shuffle(map_a)
    map_b = codes.copy(); rng.shuffle(map_b)
    out: list[Example] = []

    def add_segment(seg: int, context: str, mapping: list[str], relation: str, version: int,
                    selected: Iterable[int] | None = None) -> None:
        indices = list(range(items)) if selected is None else list(selected)
        for i in indices:
            for _ in range(train_repeats):
                out.append(Example("revision", seg, context, keys[i], mapping[i], "train", relation, version))
            for _ in range(test_repeats):
                out.append(Example("revision", seg, context, keys[i], mapping[i], "test", relation, version))

    add_segment(0, ctx_a, map_a, "durable", 1)
    add_segment(1, ctx_b, map_b, "context_exception", 1)

    revised = map_a.copy()
    subset = list(range(items // 2))
    rotated = [map_a[i] for i in subset]
    rotated = rotated[1:] + rotated[:1]
    for i, code in zip(subset, rotated):
        revised[i] = code
    add_segment(2, ctx_a, revised, "supersedes", 2, subset)

    return out, codes


def write_jsonl(path: Path, examples: Iterable[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.record(), ensure_ascii=False) + "\n")
