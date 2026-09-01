from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import random
from typing import Iterable, Literal

import torch
from torch import nn
from torch.nn import functional as F

from .metrics import ContinualMetrics, summarize

Method = Literal["sequential", "replay", "fixed", "promotion"]


@dataclass
class ToyConfig:
    input_dim: int = 24
    hidden_dim: int = 48
    classes: int = 4
    tasks: int = 6
    train_per_task: int = 96
    test_per_task: int = 160
    steps_per_example: int = 1
    lr: float = 0.035
    noise: float = 0.55
    task_shift: float = 1.6
    replay_capacity: int = 128
    replay_batch: int = 8
    latent_decay: float = 0.90
    consolidate_min_current_acc: float = 0.63
    consolidate_max_retention_drop: float = 0.025


@dataclass
class TaskData:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


class ReplayBuffer:
    """Deterministic reservoir replay buffer."""

    def __init__(self, capacity: int, rng: random.Random):
        self.capacity = capacity
        self.rng = rng
        self.items: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.seen = 0

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.seen += 1
        item = (x.detach().cpu().clone(), y.detach().cpu().clone())
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        j = self.rng.randrange(self.seen)
        if j < self.capacity:
            self.items[j] = item

    def sample(self, n: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if not self.items:
            return []
        n = min(n, len(self.items))
        return self.rng.sample(self.items, n)


class StatefulClassifier(nn.Module):
    """Frozen core with additive fast/slow plastic heads and persistent latent state."""

    def __init__(self, cfg: ToyConfig, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.cfg = cfg
        self.core = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.base_head = nn.Linear(cfg.hidden_dim, cfg.classes)
        self.slow = nn.Linear(cfg.hidden_dim, cfg.classes)
        self.fast = nn.Linear(cfg.hidden_dim, cfg.classes)
        self.latent_head = nn.Linear(cfg.hidden_dim, cfg.classes, bias=False)

        for module in (self.core, self.base_head):
            for p in module.parameters():
                p.requires_grad_(False)
        with torch.no_grad():
            self.core.weight.copy_(torch.randn(self.core.weight.shape, generator=g) / math.sqrt(cfg.input_dim))
            self.core.bias.zero_()
            self.base_head.weight.copy_(torch.randn(self.base_head.weight.shape, generator=g) / math.sqrt(cfg.hidden_dim))
            self.base_head.bias.zero_()
            self.slow.weight.zero_(); self.slow.bias.zero_()
            self.fast.weight.zero_(); self.fast.bias.zero_()
            self.latent_head.weight.copy_(torch.randn(self.latent_head.weight.shape, generator=g) * 0.01)

        self.register_buffer("latent", torch.zeros(cfg.hidden_dim))

    def reset_latent(self) -> None:
        self.latent.zero_()

    @torch.no_grad()
    def update_latent(self, h: torch.Tensor) -> None:
        obs = h.detach().mean(dim=0)
        self.latent.mul_(self.cfg.latent_decay).add_(obs * (1.0 - self.cfg.latent_decay))

    def forward(self, x: torch.Tensor, use_latent: bool = True, update_latent: bool = False) -> torch.Tensor:
        h = torch.tanh(self.core(x))
        logits = self.base_head(h) + self.slow(h) + self.fast(h)
        if use_latent:
            logits = logits + self.latent_head(self.latent).unsqueeze(0)
        if update_latent:
            self.update_latent(h)
        return logits

    @torch.no_grad()
    def consolidate_fast_into_slow(self) -> None:
        self.slow.weight.add_(self.fast.weight)
        self.slow.bias.add_(self.fast.bias)
        self.fast.weight.zero_()
        self.fast.bias.zero_()

    def fast_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.fast.parameters()
        yield from self.latent_head.parameters()


def make_tasks(cfg: ToyConfig, seed: int) -> list[TaskData]:
    """Create distinguishable but conflicting tasks."""
    g = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(cfg.classes, cfg.input_dim, generator=g) * 1.4
    tasks: list[TaskData] = []
    for _ in range(cfg.tasks):
        shift = torch.randn(cfg.input_dim, generator=g)
        shift = shift / (shift.norm() + 1e-8) * cfg.task_shift
        perm = torch.randperm(cfg.classes, generator=g)

        def sample(n: int) -> tuple[torch.Tensor, torch.Tensor]:
            latent_y = torch.randint(0, cfg.classes, (n,), generator=g)
            x = prototypes[latent_y] + shift + torch.randn(n, cfg.input_dim, generator=g) * cfg.noise
            y = perm[latent_y]
            return x, y

        xtr, ytr = sample(cfg.train_per_task)
        xte, yte = sample(cfg.test_per_task)
        tasks.append(TaskData(xtr, ytr, xte, yte))
    return tasks


@torch.no_grad()
def accuracy(model: StatefulClassifier, task: TaskData, use_latent: bool) -> float:
    logits = model(task.x_test, use_latent=use_latent, update_latent=False)
    return float((logits.argmax(dim=-1) == task.y_test).float().mean().item())


@torch.no_grad()
def replay_accuracy(model: StatefulClassifier, replay: ReplayBuffer, use_latent: bool) -> float:
    if not replay.items:
        return 1.0
    xs = torch.stack([x for x, _ in replay.items])
    ys = torch.stack([y for _, y in replay.items]).view(-1)
    logits = model(xs, use_latent=use_latent, update_latent=False)
    return float((logits.argmax(-1) == ys).float().mean().item())


def train_one(model: StatefulClassifier, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor,
              replay: ReplayBuffer | None, cfg: ToyConfig, use_replay: bool, use_latent: bool) -> None:
    model.train()
    xbatch = x.unsqueeze(0)
    ybatch = y.view(1)
    if use_replay and replay is not None:
        samples = replay.sample(cfg.replay_batch)
        if samples:
            rx = torch.stack([a for a, _ in samples])
            ry = torch.stack([b for _, b in samples]).view(-1)
            xbatch = torch.cat([xbatch, rx], dim=0)
            ybatch = torch.cat([ybatch, ry], dim=0)

    for _ in range(cfg.steps_per_example):
        optimizer.zero_grad(set_to_none=True)
        logits = model(xbatch, use_latent=use_latent, update_latent=False)
        loss = F.cross_entropy(logits, ybatch)
        loss.backward()
        optimizer.step()
    if use_latent:
        with torch.no_grad():
            _ = model(x.unsqueeze(0), use_latent=True, update_latent=True)


def attempt_guarded_consolidation(model: StatefulClassifier, current_task: TaskData, replay: ReplayBuffer,
                                   cfg: ToyConfig, use_latent: bool) -> bool:
    current = accuracy(model, current_task, use_latent)
    if current < cfg.consolidate_min_current_acc:
        return False
    before = replay_accuracy(model, replay, use_latent)
    snapshot = copy.deepcopy(model.state_dict())
    model.consolidate_fast_into_slow()
    after = replay_accuracy(model, replay, use_latent)
    if before - after > cfg.consolidate_max_retention_drop:
        model.load_state_dict(snapshot)
        return False
    return True


@dataclass
class ToyRun:
    method: str
    seed: int
    score_matrix: list[list[float]]
    metrics: ContinualMetrics
    consolidations: int
    rejected_consolidations: int
    optimizer_steps: int


def run_method(method: Method, seed: int, cfg: ToyConfig | None = None) -> ToyRun:
    cfg = cfg or ToyConfig()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tasks = make_tasks(cfg, seed + 1000)
    model = StatefulClassifier(cfg, seed + 2000)
    replay = ReplayBuffer(cfg.replay_capacity, rng)

    use_replay = method in {"replay", "fixed", "promotion"}
    use_latent = method == "promotion"
    optimizer = torch.optim.SGD(list(model.fast_parameters()), lr=cfg.lr)
    score_matrix: list[list[float]] = []
    consolidations = 0
    rejected = 0
    opt_steps = 0

    for ti, task in enumerate(tasks):
        order = list(range(cfg.train_per_task))
        rng.shuffle(order)
        for idx in order:
            train_one(model, optimizer, task.x_train[idx], task.y_train[idx], replay, cfg, use_replay, use_latent)
            opt_steps += cfg.steps_per_example
            replay.add(task.x_train[idx], task.y_train[idx])

        if method == "fixed":
            model.consolidate_fast_into_slow()
            consolidations += 1
        elif method == "promotion":
            if attempt_guarded_consolidation(model, task, replay, cfg, use_latent):
                consolidations += 1
            else:
                rejected += 1

        row = [float("nan")] * cfg.tasks
        for tj in range(ti + 1):
            row[tj] = accuracy(model, tasks[tj], use_latent)
        score_matrix.append(row)

    return ToyRun(
        method=method,
        seed=seed,
        score_matrix=score_matrix,
        metrics=summarize(score_matrix),
        consolidations=consolidations,
        rejected_consolidations=rejected,
        optimizer_steps=opt_steps,
    )
