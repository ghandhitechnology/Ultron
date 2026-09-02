from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from subprocess import CompletedProcess
from threading import Thread
from typing import Any

from ultron.env.availability import ServiceProbe, check_availability, tcp_probe
from ultron.env.backend import GuestHandle, IsolationBackend
from ultron.env.docker_backend import DockerBackend
from ultron.env.snapshot import SnapshotDriftError
from ultron.env.vm_pool import VmPool, restore_verified
from ultron.train.adjudicator import ProbeResult, adjudicate
from ultron.train.convert_verl import trajectory_to_verl_records
from ultron.train.episode_runner import EpisodeConfig, EpisodeRunner
from ultron.train.schema_v1 import ReasonCode, Role, ToolEvent, TrajectoryV1
from ultron.train.turn_record import build_trajectory_step

ROOT = Path(__file__).resolve().parents[1]
BABY_IMAGE = "ultron/baby:smoke"
BABY_CONTEXT = ROOT / "env" / "baby"
BABY_GUEST_ID = "ultron-baby-smoke"
ATTACKER = "attacker"
HTTP_PORT = 8080
NETWORK = "ultron-isolated"
WRONG_IMAGE_ID = "sha256:" + ("0" * 64)


class SpyRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], **kwargs: Any) -> CompletedProcess[str]:
        argv = list(args)
        self.calls.append(argv)
        return subprocess.run(argv, **kwargs)


class _LoopbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args: Any) -> None:
        return


def docker_daemon_ok() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True, text=True).returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Broad, short Docker guest smoke for a cloud GPU server."
    )
    parser.add_argument("--image", default=BABY_IMAGE)
    parser.add_argument("--guest-id", default=BABY_GUEST_ID)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    require_docker()
    check_nvidia(required=args.require_gpu)
    ensure_network()
    build_image(args.image)

    spy = SpyRunner()
    backend = DockerBackend(run=spy, network=NETWORK, cpus="0.5", memory="128m")
    image_id = inspect_image_id(backend, args.image)
    backend.verify_image(args.image, image_id)
    print(f"ok  verify_image match ({image_id[:19]}…)")
    try:
        backend.verify_image(args.image, WRONG_IMAGE_ID)
    except SnapshotDriftError:
        print("ok  verify_image rejects drift")
    else:
        raise SystemExit("FAIL: verify_image accepted a mismatched id")

    placeholder = GuestHandle(
        guest_id=args.guest_id,
        isolation=IsolationBackend.DOCKER,
        host_address="",
        image_ref=args.image,
    )
    pool = VmPool([placeholder], backend, restore_timeout_s=20)
    handle: GuestHandle | None = None
    loopback: HTTPServer | None = None
    try:
        handle = pool.restore_verified(placeholder, image_id)
        print(f"ok  restore {handle.guest_id} at {handle.host_address}")
        wait_tcp(handle.host_address, HTTP_PORT)

        stdout, code = backend.exec_as_user(handle, ATTACKER, "id -un && id -u")
        if code != 0 or ATTACKER not in stdout or "1000" not in stdout:
            raise SystemExit(f"FAIL: exec_as_user: code={code} stdout={stdout!r}")
        print(f"ok  exec_as_user ({' '.join(stdout.split())})")

        before = len(spy.calls)
        claimed_root = backend.confirm_root(handle, ATTACKER)
        confirm_calls = spy.calls[before:]
        if any(len(cmd) >= 2 and cmd[1] == "exec" for cmd in confirm_calls):
            raise SystemExit("FAIL: confirm_root invoked docker exec")
        if claimed_root:
            raise SystemExit("FAIL: confirm_root true before any euid-0 process")
        print("ok  confirm_root false without euid 0, and no docker exec")

        availability = check_availability(
            handle.host_address, [ServiceProbe("http", HTTP_PORT)]
        )
        if not availability.ok:
            raise SystemExit(f"FAIL: availability {availability.details}")
        print("ok  host TCP availability")

        loopback = start_loopback()
        loopback_port = int(loopback.server_address[1])
        if not tcp_probe("127.0.0.1", loopback_port, timeout_s=1.0):
            raise SystemExit("FAIL: host cannot reach 127.0.0.1 stand-in")
        print(f"ok  host loopback stand-in :{loopback_port}")
        check_guest_isolation(backend, handle, loopback_port)

        trajectories = run_baby_episode(backend, handle, image_id)
        check_trajectories(trajectories)
        print("ok  baby EpisodeRunner + schema + veRL convert")

        lying = adjudicate(
            ProbeResult(
                guest_attacker_euid=0,
                host_confirmed_root=False,
                availability_ok=True,
                infra_ok=True,
                timed_out=False,
            )
        )
        if lying[0] != ReasonCode.DEFENDER_HOLD:
            raise SystemExit(f"FAIL: guest uid 0 without host confirm scored {lying[0]}")
        print("ok  guest uid 0 without host confirm is not a win")

        spawn_euid0(backend, handle)
        time.sleep(0.5)
        if not backend.confirm_root(handle, ATTACKER):
            raise SystemExit(confirm_root_permission_hint(backend, handle))
        print("ok  confirm_root true for attacker real=1000 euid=0")
    finally:
        if loopback is not None:
            loopback.shutdown()
        if not args.keep:
            if handle is not None:
                pool.quarantine(handle, "baby-smoke-done")
            else:
                backend.stop(args.guest_id)
            print(f"ok  stop {args.guest_id}")

    print("Baby cloud smoke passed.")
    return 0


def require_docker() -> None:
    if not docker_daemon_ok():
        raise SystemExit("FAIL: docker daemon is not available")
    print("ok  docker info")


def check_nvidia(*, required: bool) -> None:
    if shutil.which("nvidia-smi") is None:
        if required:
            raise SystemExit("FAIL: nvidia-smi missing")
        print("skip nvidia-smi")
        return
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode != 0:
        if required:
            raise SystemExit("FAIL: nvidia-smi")
        print("skip nvidia-smi (command failed)")
        return
    print("ok  nvidia-smi")


def ensure_network() -> None:
    inspect = subprocess.run(
        ["docker", "network", "inspect", NETWORK],
        capture_output=True,
        text=True,
    )
    if inspect.returncode == 0:
        print(f"ok  docker network {NETWORK}")
        return
    created = subprocess.run(
        ["docker", "network", "create", "--internal", NETWORK],
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        raise SystemExit(f"FAIL: docker network create: {created.stderr}")
    print(f"ok  docker network create --internal {NETWORK}")


def build_image(image: str) -> None:
    if not BABY_CONTEXT.is_dir():
        raise SystemExit(f"FAIL: missing baby context {BABY_CONTEXT}")
    result = subprocess.run(
        ["docker", "build", "-t", image, str(BABY_CONTEXT)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout or "FAIL: docker build")
    print(f"ok  docker build {image}")


def inspect_image_id(backend: DockerBackend, image: str) -> str:
    result = backend._invoke(["docker", "inspect", "--format", "{{.Id}}", image])
    image_id = (result.stdout or "").strip()
    if result.returncode != 0 or not image_id:
        raise SystemExit(f"FAIL: docker inspect image id: {result.stderr}")
    return image_id


def wait_tcp(host: str, port: int, timeout_s: float = 8.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if tcp_probe(host, port, timeout_s=0.4):
            return
        time.sleep(0.2)
    raise SystemExit(f"FAIL: {host}:{port} did not accept connections")


def start_loopback() -> HTTPServer:
    server = HTTPServer(("127.0.0.1", 0), _LoopbackHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server


def check_guest_isolation(
    backend: DockerBackend, handle: GuestHandle, loopback_port: int
) -> None:
    internet = _python_connect_cmd("1.1.1.1", 443)
    _, internet_code = backend.exec_as_user(handle, ATTACKER, internet)
    if internet_code == 0:
        raise SystemExit("FAIL: guest reached 1.1.1.1:443 on an internal network")
    print("ok  guest cannot reach the internet")

    loopback = _python_connect_cmd("127.0.0.1", loopback_port)
    _, loopback_code = backend.exec_as_user(handle, ATTACKER, loopback)
    if loopback_code == 0:
        raise SystemExit("FAIL: guest reached the host loopback stand-in via 127.0.0.1")
    print("ok  guest 127.0.0.1 is not the host vLLM bind")

    gateway = network_gateway()
    if gateway is None:
        print("skip host-gateway isolation (no gateway on network)")
        return
    _, gateway_code = backend.exec_as_user(
        handle, ATTACKER, _python_connect_cmd(gateway, loopback_port)
    )
    if gateway_code == 0:
        raise SystemExit(
            f"FAIL: guest reached host 127.0.0.1 stand-in via gateway {gateway}"
        )
    print(f"ok  guest cannot reach loopback stand-in via {gateway}")


def network_gateway() -> str | None:
    result = subprocess.run(
        [
            "docker",
            "network",
            "inspect",
            "--format",
            "{{(index .IPAM.Config 0).Gateway}}",
            NETWORK,
        ],
        capture_output=True,
        text=True,
    )
    value = (result.stdout or "").strip()
    if result.returncode != 0 or not value or value in {"<no value>", "<nil>"}:
        return None
    return value


def _python_connect_cmd(host: str, port: int) -> str:
    return (
        "python3 -c "
        f"'import socket,sys;"
        f"s=socket.socket();s.settimeout(2);"
        f"e=s.connect_ex((\"{host}\",{port}));"
        f"s.close();sys.exit(0 if e==0 else 1)'"
    )


def run_baby_episode(
    backend: DockerBackend, handle: GuestHandle, image_id: str
) -> list[TrajectoryV1]:
    live: dict[str, GuestHandle] = {"handle": handle}

    def restore(vm: GuestHandle, expected_id: str) -> None:
        live["handle"] = restore_verified(
            vm, expected_id, backend, restore_timeout_s=20
        )
        wait_tcp(live["handle"].host_address, HTTP_PORT)

    def run_turn(
        vm: GuestHandle, side: Role, profile: dict[str, Any], turn: int
    ) -> list:
        guest = live["handle"]
        stdout, code = backend.exec_as_user(guest, ATTACKER, "id -un")
        text = stdout.strip()[:32]
        return [
            build_trajectory_step(
                turn_index=turn,
                side=side,
                prompt_token_ids=[1, 2],
                assistant_token_ids=[3],
                assistant_mask=[1],
                tool_events=[
                    ToolEvent(
                        name="bash",
                        args={"command": "id -un"},
                        stdout_head=text,
                        stdout_tail=text,
                        exit_code=code,
                        duration_ms=1,
                    )
                ],
                observation_hash="baby",
            )
        ]

    def final_probe(vm: GuestHandle, profile: dict[str, Any]) -> ProbeResult:
        guest = live["handle"]
        stdout, code = backend.exec_as_user(guest, ATTACKER, "id -u")
        try:
            euid = int((stdout or "").strip()) if code == 0 else -1
        except ValueError:
            euid = -1
        availability = check_availability(
            guest.host_address, [ServiceProbe("http", HTTP_PORT)]
        )
        return ProbeResult(
            guest_attacker_euid=euid,
            host_confirmed_root=backend.confirm_root(guest, ATTACKER),
            availability_ok=availability.ok,
            infra_ok=True,
            timed_out=False,
        )

    runner = EpisodeRunner(
        snapshot_sha256=image_id,
        load_profile=lambda profile_id: {"id": profile_id},
        run_turn=run_turn,
        final_probe=final_probe,
        restore=restore,
        turns_per_side=1,
    )
    cfg = EpisodeConfig(
        profile_id="web",
        generation=0,
        group_id="baby-group",
        opponent_checkpoint_id="defender-baby",
        attacker_ckpt="attacker-baby",
        defender_ckpt="defender-baby",
    )
    return runner.run(cfg, handle)


def check_trajectories(trajectories: list[TrajectoryV1]) -> None:
    if len(trajectories) != 2:
        raise SystemExit(f"FAIL: expected 2 trajectories, got {len(trajectories)}")
    roles = {traj.role for traj in trajectories}
    if roles != {Role.ATTACKER, Role.DEFENDER}:
        raise SystemExit(f"FAIL: roles {roles}")
    for traj in trajectories:
        payload = traj.to_dict()
        roundtrip = TrajectoryV1.from_dict(json.loads(json.dumps(payload)))
        if roundtrip.isolation_backend is not IsolationBackend.DOCKER:
            raise SystemExit("FAIL: isolation_backend is not docker")
        if roundtrip.terminal.reason_code != ReasonCode.DEFENDER_HOLD:
            raise SystemExit(
                f"FAIL: baby episode reason {roundtrip.terminal.reason_code}"
            )
        records = trajectory_to_verl_records(roundtrip, generation=0)
        if len(records) != len(roundtrip.steps):
            raise SystemExit("FAIL: veRL convert dropped steps")


def spawn_euid0(backend: DockerBackend, handle: GuestHandle) -> None:
    result = backend._invoke(
        [
            "docker",
            "exec",
            "-d",
            "-u",
            "root",
            handle.guest_id,
            "python3",
            "-c",
            "import os,time; os.setresuid(1000,0,0); time.sleep(20)",
        ]
    )
    if result.returncode != 0:
        raise SystemExit(f"FAIL: spawn euid0: {result.stderr}")


def confirm_root_permission_hint(backend: DockerBackend, handle: GuestHandle) -> str:
    result = backend._invoke(
        ["docker", "inspect", "--format", "{{.State.Pid}}", handle.guest_id]
    )
    pid = (result.stdout or "").strip()
    guest_root = backend.proc_root / pid / "root" / "etc" / "passwd"
    try:
        guest_root.read_text()
    except OSError as exc:
        return (
            "FAIL: confirm_root stayed false after attacker euid 0. "
            f"Cannot read {guest_root}: {exc}. Rerun as root so host /proc "
            "confirm can see the guest."
        )
    return (
        "FAIL: confirm_root stayed false after spawning real=1000 euid=0 "
        f"(guest /proc readable via {guest_root})"
    )


if __name__ == "__main__":
    sys.exit(main())
