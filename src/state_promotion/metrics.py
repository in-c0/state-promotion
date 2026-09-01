from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ContinualMetrics:
    final_average: float
    average_forgetting: float
    average_plasticity: float
    retention_auc: float


def summarize(score_matrix: Sequence[Sequence[float]]) -> ContinualMetrics:
    """Summarize a lower-triangular continual-learning score matrix.

    score_matrix[i][j] is performance on task j after learning task i.
    Entries for j > i may be NaN.
    """
    r = np.asarray(score_matrix, dtype=float)
    n = r.shape[0]
    final = r[n - 1, :n]
    final_average = float(np.nanmean(final))

    forgetting = []
    for j in range(n - 1):
        history = r[j:, j]
        history = history[~np.isnan(history)]
        if len(history):
            forgetting.append(float(np.max(history) - final[j]))
    average_forgetting = float(np.mean(forgetting)) if forgetting else 0.0

    diagonal = np.diag(r)
    average_plasticity = float(np.nanmean(diagonal))

    per_step = []
    for i in range(n):
        per_step.append(float(np.nanmean(r[i, : i + 1])))
    retention_auc = float(np.mean(per_step))

    return ContinualMetrics(
        final_average=final_average,
        average_forgetting=average_forgetting,
        average_plasticity=average_plasticity,
        retention_auc=retention_auc,
    )
