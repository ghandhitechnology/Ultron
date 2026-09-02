from __future__ import annotations

import time
from collections.abc import Callable

from ultron.cli.model import JobMeta
from ultron.cli.observe import EpisodeCase
from ultron.env.backend import GuestHandle, IsolationBackend
from ultron.train.adjudicator import ProbeResult
from ultron.train.episode_runner import EpisodeConfig, EpisodeRunner
from ultron.train.schema_v1 import Role, ToolEvent
from ultron.train.turn_record import build_trajectory_step

Sleep = Callable[[float], None]


def origin_clock() -> Callable[[], float]:
    started = time.monotonic()
    return lambda: time.monotonic() - started


def make_demo(
    meta: JobMeta,
    *,
    delay_s: float = 0.08,
    sleep: Sleep = time.sleep,
) -> tuple[EpisodeRunner, tuple[EpisodeCase, ...]]:
    def restore(vm, sha: str) -> None:
        sleep(delay_s)

    def load_profile(profile_id: str) -> dict:
        return {"profile_id": profile_id}

    def run_turn(vm, role: Role, profile: dict, turn: int):
        sleep(delay_s)
        tool = _demo_tool(role, turn)
        return [
            build_trajectory_step(
                turn_index=turn,
                side=role,
                prompt_token_ids=[1],
                assistant_token_ids=[2],
                assistant_mask=[1],
                tool_events=[tool],
            )
        ]

    def final_probe(vm, profile: dict) -> ProbeResult:
        sleep(delay_s)
        episode = int(vm.vm_id.rsplit("-", 1)[-1])
        rooted = episode == meta.episodes_planned - 1 and meta.episodes_planned > 1
        return ProbeResult(
            guest_attacker_euid=0 if rooted else 1000,
            host_confirmed_root=rooted,
            availability_ok=True,
            infra_ok=True,
            timed_out=False,
        )

    runner = EpisodeRunner(
        snapshot_sha256=meta.snapshot_sha256 or "demo-sha",
        load_profile=load_profile,
        run_turn=run_turn,
        final_probe=final_probe,
        restore=restore,
        turns_per_side=meta.turns_per_side,
    )
    cases = tuple(_case(meta, index) for index in range(meta.episodes_planned))
    return runner, cases


def _case(meta: JobMeta, index: int) -> EpisodeCase:
    vm = GuestHandle(
        guest_id=f"ultron-guest-{index}",
        isolation=meta.isolation,
        host_address=f"10.0.0.{10 + index}",
        image_ref="ultron/golden:demo",
    )
    cfg = EpisodeConfig(
        profile_id=meta.profile_id,
        generation=meta.generation,
        group_id=meta.group_id,
        opponent_checkpoint_id="defender-frozen",
        attacker_ckpt="attacker_lora",
        defender_ckpt="defender_lora",
    )
    return EpisodeCase(config=cfg, vm=vm)


def _demo_tool(role: Role, turn: int) -> ToolEvent:
    if role is Role.ATTACKER:
        scripts = (
            ("bash", "id"),
            ("bash", "cat /etc/shadow"),
            ("bash", "find / -perm -4000"),
            ("bash", "curl -s http://127.0.0.1/admin"),
        )
    else:
        scripts = (
            ("bash", "nft add rule inet filter input drop"),
            ("bash", "systemctl restart sshd"),
            ("bash", "chmod 600 /etc/crontab"),
            ("bash", "ss -lntp"),
        )
    name, cmd = scripts[turn % len(scripts)]
    denied = "permission denied" in cmd or cmd.startswith("cat /etc/shadow")
    stdout = "uid=1000(attacker) gid=1000\n" if cmd == "id" else f"{cmd}\n"
    if denied:
        stdout = "cat: /etc/shadow: Permission denied\n"
    return ToolEvent(
        name=name,
        args={"cmd": cmd},
        stdout_head=stdout,
        stdout_tail="",
        exit_code=1 if denied else 0,
        duration_ms=80 + turn * 17,
    )
