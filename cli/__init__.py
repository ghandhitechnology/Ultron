"""Live simulation view for Ultron rollouts. Framework-free surface only.

Importing this package never imports Textual.
"""

from __future__ import annotations

from .demo import make_demo
from .model import (
    AttackingSnapshot,
    CompleteSnapshot,
    DefendingSnapshot,
    EpisodeCase,
    EpisodeSummary,
    FailedSnapshot,
    InvalidTransition,
    JobEvent,
    JobProgress,
    JobSnapshot,
    JobSpec,
    Phase,
    ProbingSnapshot,
    RestoringSnapshot,
    active_role,
    apply,
    estimate_eta_s,
    initial_snapshot,
    progress,
)
from .observe import instrument_runner, run_job

__all__ = [
    "AttackingSnapshot",
    "CompleteSnapshot",
    "DefendingSnapshot",
    "EpisodeCase",
    "EpisodeSummary",
    "FailedSnapshot",
    "InvalidTransition",
    "JobEvent",
    "JobProgress",
    "JobSnapshot",
    "JobSpec",
    "Phase",
    "ProbingSnapshot",
    "RestoringSnapshot",
    "active_role",
    "apply",
    "estimate_eta_s",
    "initial_snapshot",
    "instrument_runner",
    "make_demo",
    "progress",
    "run_job",
]
