from dataclasses import dataclass
from pathlib import Path

from ultron.train.adjudicator import ProbeResult
from ultron.train.episode_runner import EpisodeConfig, EpisodeRunner


@dataclass(frozen=True)
class FakeVm:
    vm_id: str
    vsock_cid: int
    host_address: str
    snapshot_path: Path


def test_run_restores_with_expected_sha() -> None:
    calls: list[tuple[str, str]] = []
    probe = ProbeResult(
        guest_attacker_euid=1000,
        host_confirmed_root=False,
        availability_ok=True,
        infra_ok=True,
        timed_out=False,
    )
    runner = EpisodeRunner(
        snapshot_sha256="a" * 64,
        load_profile=lambda profile_id: {},
        run_turn=lambda *args: [],
        final_probe=lambda vm, profile: probe,
        restore=lambda vm, sha: calls.append((vm.vm_id, sha)),
        turns_per_side=0,
    )
    cfg = EpisodeConfig(
        profile_id="web",
        generation=0,
        group_id="group-1",
        opponent_checkpoint_id="defender-gen0",
        attacker_ckpt="attacker-gen0",
        defender_ckpt="defender-gen0",
    )
    vm = FakeVm(vm_id="vm-1", vsock_cid=3, host_address="10.0.0.2", snapshot_path=Path("/snap"))

    trajectories = runner.run(cfg, vm)

    assert calls == [("vm-1", "a" * 64)]
    assert len(trajectories) == 2
