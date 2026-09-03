from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ultron.env.backend import IsolationBackend
from ultron.train.adjudicator import ProbeResult
from ultron.train.episode_runner import EpisodeConfig, EpisodeRunner, GuestVm
from ultron.train.schema_v1 import Role, ToolEvent, TrajectoryStep

from .model import EpisodeCase, JobSpec

RESTORE_SLEEP_S = 0.3
TURN_SLEEP_S = 0.35

_ATTACKER_TOOLS = [
    ("sudo -l -U attacker", "Matching Defaults entries for attacker", "may not run sudo", 1),
    ("find / -perm -4000 -type f 2>/dev/null", "/usr/bin/passwd\n/usr/bin/mount", "/usr/bin/su", 0),
    ("cat /etc/cron.d/*", "# m h dom mon dow user command", "root backup.sh", 0),
    ("chmod u+s /tmp/x", "", "Operation not permitted", 1),
]
_DEFENDER_TOOLS = [
    ("auditctl -w /etc/sudoers -p wa", "", "rule added", 0),
    ("chattr +i /etc/passwd", "", "", 0),
    ("systemctl status nginx", "active (running)", "Main PID 812", 0),
    ("iptables -A INPUT -p tcp --dport 22 -j DROP", "", "", 0),
]


@dataclass(frozen=True)
class FakeGuestVm:
    vm_id: str
    isolation: IsolationBackend
    host_address: str
    image_ref: str


def _tool(role: Role, turn: int) -> ToolEvent:
    table = _ATTACKER_TOOLS if role is Role.ATTACKER else _DEFENDER_TOOLS
    cmd, head, tail, code = table[turn // 2 % len(table)]
    return ToolEvent(
        name="bash",
        args={"cmd": cmd, "timeout_s": 30},
        stdout_head=head,
        stdout_tail=tail,
        exit_code=code,
        duration_ms=80 + 40 * (turn % 5),
    )


def make_demo(spec: JobSpec) -> tuple[EpisodeRunner, tuple[EpisodeCase, ...]]:
    cases: list[EpisodeCase] = []
    probes: dict[str, ProbeResult] = {}
    for index in range(spec.episode_count):
        vm = FakeGuestVm(
            vm_id=f"web-{index + 1:02d}",
            isolation=spec.isolation_backend,
            host_address=f"10.0.0.{index + 10}",
            image_ref=f"ultron/{spec.profile_id}:golden",
        )
        config = EpisodeConfig(
            profile_id=spec.profile_id,
            generation=spec.generation,
            group_id=f"g{spec.generation}-demo",
            opponent_checkpoint_id=f"defender-gen{spec.generation}",
            attacker_ckpt=f"attacker-gen{spec.generation}",
            defender_ckpt=f"defender-gen{spec.generation}",
        )
        cases.append(EpisodeCase(config=config, vm=vm))
        if index == 1:
            probes[vm.vm_id] = ProbeResult(
                guest_attacker_euid=0,
                host_confirmed_root=True,
                availability_ok=True,
                infra_ok=True,
                timed_out=False,
            )
        else:
            probes[vm.vm_id] = ProbeResult(
                guest_attacker_euid=1000,
                host_confirmed_root=False,
                availability_ok=True,
                infra_ok=True,
                timed_out=False,
            )

    def restore(vm: GuestVm, sha: str) -> None:
        time.sleep(RESTORE_SLEEP_S)

    def load_profile(profile_id: str) -> dict[str, Any]:
        return {"profile_id": profile_id}

    def run_turn(
        vm: GuestVm, role: Role, profile: dict[str, Any], turn: int
    ) -> list[TrajectoryStep]:
        time.sleep(TURN_SLEEP_S)
        return [
            TrajectoryStep(
                turn_index=turn,
                side=role,
                prompt_token_ids=[1, 2, 3],
                assistant_token_ids=[4, 5],
                assistant_mask=[1, 1],
                tool_events=[_tool(role, turn)],
                decision_point=turn % 4 == 0,
                observation_hash=f"{role.value[:1]}{turn:03d}",
            )
        ]

    def final_probe(vm: GuestVm, profile: dict[str, Any]) -> ProbeResult:
        return probes[vm.vm_id]

    runner = EpisodeRunner(
        snapshot_sha256="9f2c1b8e" + "0" * 56,
        load_profile=load_profile,
        run_turn=run_turn,
        final_probe=final_probe,
        restore=restore,
        turns_per_side=spec.turns_per_side,
    )
    return runner, tuple(cases)
