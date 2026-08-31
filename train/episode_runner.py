from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from .adjudicator import ProbeResult, adjudicate
from .rewards import assign_gen01_attacker_turn_rewards, assign_terminal_rtg
from .schema_v1 import (
    SCHEMA_VERSION,
    Role,
    TerminalOutcome,
    TrajectoryStep,
    TrajectoryV1,
)


class GuestVm(Protocol):
    vm_id: str
    vsock_cid: int
    host_address: str
    snapshot_path: Path


RestoreFn = Callable[[GuestVm, str], None]
TurnExecutor = Callable[[GuestVm, Role, dict[str, Any], int], list[TrajectoryStep]]
FinalProbe = Callable[[GuestVm, dict[str, Any]], ProbeResult]
ProfileLoader = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class EpisodeConfig:
    profile_id: str
    generation: int
    group_id: str
    opponent_checkpoint_id: str
    attacker_ckpt: str
    defender_ckpt: str


class EpisodeRunner:
    def __init__(
        self,
        *,
        snapshot_sha256: str,
        load_profile: ProfileLoader,
        run_turn: TurnExecutor,
        final_probe: FinalProbe,
        restore: RestoreFn,
        turns_per_side: int = 8,
    ) -> None:
        self.snapshot_sha256 = snapshot_sha256
        self.load_profile = load_profile
        self.run_turn = run_turn
        self.final_probe = final_probe
        self.restore = restore
        self.turns_per_side = turns_per_side

    def run(self, cfg: EpisodeConfig, vm: GuestVm) -> list[TrajectoryV1]:
        self.restore(vm, self.snapshot_sha256)
        profile = self.load_profile(cfg.profile_id)
        attacker_steps: list[TrajectoryStep] = []
        defender_steps: list[TrajectoryStep] = []
        for turn in range(self.turns_per_side * 2):
            side = Role.ATTACKER if turn % 2 == 0 else Role.DEFENDER
            steps = self.run_turn(vm, side, profile, turn)
            target = attacker_steps if side == Role.ATTACKER else defender_steps
            target.extend(steps)

        probe = self.final_probe(vm, profile)
        reason, attacker_reward, defender_reward = adjudicate(probe)
        if cfg.generation <= 1:
            assign_gen01_attacker_turn_rewards(attacker_steps)
        assign_terminal_rtg(attacker_steps, attacker_reward, generation=cfg.generation)
        assign_terminal_rtg(defender_steps, defender_reward, generation=cfg.generation)

        terminal = TerminalOutcome(
            reason_code=reason,
            attacker_euid=probe.guest_attacker_euid,
            host_confirmed_root=probe.host_confirmed_root,
            availability_ok=probe.availability_ok,
            attacker_reward=attacker_reward,
            defender_reward=defender_reward,
        )
        episode_id = str(uuid4())
        return [
            self._trajectory(cfg, episode_id, Role.ATTACKER, attacker_steps, terminal),
            self._trajectory(cfg, episode_id, Role.DEFENDER, defender_steps, terminal),
        ]

    @staticmethod
    def _trajectory(
        cfg: EpisodeConfig,
        episode_id: str,
        role: Role,
        steps: list[TrajectoryStep],
        terminal: TerminalOutcome,
    ) -> TrajectoryV1:
        return TrajectoryV1(
            schema_version=SCHEMA_VERSION,
            episode_id=episode_id,
            generation=cfg.generation,
            profile_id=cfg.profile_id,
            role=role,
            adapter_id=f"{role.value}_lora",
            opponent_checkpoint_id=cfg.opponent_checkpoint_id,
            group_id=cfg.group_id,
            steps=steps,
            terminal=terminal,
        )
