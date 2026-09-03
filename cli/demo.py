from __future__ import annotations

from ultron.env.backend import GuestHandle, IsolationBackend
from ultron.train.adjudicator import ProbeResult
from ultron.train.episode_runner import GuestVm
from ultron.train.schema_v1 import Role, ToolEvent, TrajectoryStep
from ultron.train.turn_record import build_trajectory_step

from .fold import BoardFold
from .spec import JobSpec, Seams

SCRIPTS = ("full-16", "hold-at-turn-7", "attacker-root", "infra-fail")


def default_spec() -> JobSpec:
    return JobSpec(
        episode_count=1,
        generation=3,
        profile_id="web",
        group_id="g-12",
        opponent_checkpoint_id="defender-gen2",
        snapshot_sha256="9f2c" + "a" * 60,
        turns_per_side=8,
    )


def scripted_guest() -> GuestHandle:
    return GuestHandle(
        guest_id="guest-03",
        isolation=IsolationBackend.DOCKER,
        host_address="ultron-isolated",
        image_ref="ubuntu-demo",
    )


def _step(turn: int, role: Role, events: list[ToolEvent]) -> TrajectoryStep:
    return build_trajectory_step(
        turn_index=turn,
        side=role,
        prompt_token_ids=[1],
        assistant_token_ids=[2],
        assistant_mask=[1],
        tool_events=events,
    )


def _tool(name: str, cmd: str, exit_code: int, duration_ms: int) -> ToolEvent:
    return ToolEvent(
        name=name,
        args={"cmd": cmd},
        stdout_head="",
        stdout_tail="",
        exit_code=exit_code,
        duration_ms=duration_ms,
    )


def _turn_events(turn: int) -> list[ToolEvent]:
    if turn % 2 == 0:
        cmds = (
            ("bash", "cat /etc/shadow", 1, 412),
            ("bash", "find / -perm -4000 -type f", 0, 2204),
        )
    else:
        cmds = (
            ("bash", "chmod 600 /etc/shadow", 0, 96),
            ("bash", "systemctl restart nginx", 0, 742),
        )
    return [_tool(name, cmd, code, dur) for name, cmd, code, dur in cmds]


def play_script(fold: BoardFold, script: str, guest: GuestVm, spec: JobSpec) -> None:
    if script not in SCRIPTS:
        raise ValueError(f"unknown script {script}")
    sha = spec.snapshot_sha256
    freeze_after = 7 if script == "hold-at-turn-7" else None
    fold.apply_restore_start(guest, sha)
    fold.apply_restore_done(guest)
    budget = spec.turns_per_side * 2
    for turn in range(budget):
        role = Role.ATTACKER if turn % 2 == 0 else Role.DEFENDER
        fold.apply_turn_start(role, turn)
        if freeze_after is not None and turn == freeze_after:
            return
        fold.apply_turn_done(role, turn, [_step(turn, role, _turn_events(turn))])
    fold.apply_probe_start()
    if script == "attacker-root":
        probe = ProbeResult(
            guest_attacker_euid=0,
            host_confirmed_root=True,
            availability_ok=True,
            infra_ok=True,
            timed_out=False,
        )
    elif script == "infra-fail":
        probe = ProbeResult(
            guest_attacker_euid=1000,
            host_confirmed_root=False,
            availability_ok=True,
            infra_ok=False,
            timed_out=False,
        )
    else:
        probe = ProbeResult(
            guest_attacker_euid=1000,
            host_confirmed_root=False,
            availability_ok=True,
            infra_ok=True,
            timed_out=False,
        )
    fold.apply_probe_done(probe)
    fold.apply_episode_done()
    fold.apply_job_done()


def scripted_seams(script: str) -> Seams:
    guest = scripted_guest()

    def restore(vm: GuestVm, sha256: str) -> None:
        return None

    def run_turn(vm: GuestVm, role: Role, profile: dict, turn: int) -> list[TrajectoryStep]:
        return [_step(turn, role, _turn_events(turn))]

    def final_probe(vm: GuestVm, profile: dict) -> ProbeResult:
        if script == "attacker-root":
            return ProbeResult(0, True, True, True, False)
        if script == "infra-fail":
            return ProbeResult(1000, False, True, False, False)
        return ProbeResult(1000, False, True, True, False)

    def load_profile(profile_id: str) -> dict:
        return {"id": profile_id}

    _ = guest
    return Seams(restore, run_turn, final_probe, load_profile)
