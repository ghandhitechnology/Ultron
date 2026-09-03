import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ultron.train.capability import (
    CapabilityError,
    Fit,
    GpuSnapshot,
    HostSnapshot,
    assess,
    main,
    parse_family_name,
    parse_nvidia_smi,
    parse_physical_cores,
    parse_ram_gb,
    read_host,
    render,
    report_ok,
    to_payload,
)
from ultron.train.family import FamilyName

ROOT = Path(__file__).resolve().parents[2]

CPUINFO_TWO_CORE_HT = """
processor	: 0
physical id	: 0
core id		: 0

processor	: 1
physical id	: 0
core id		: 0

processor	: 2
physical id	: 0
core id		: 1

processor	: 3
physical id	: 0
core id		: 1
"""

H100 = GpuSnapshot(0, "NVIDIA H100 80GB HBM3", 81559)
H100_B = GpuSnapshot(1, "NVIDIA H100 80GB HBM3", 81559)
L40S = GpuSnapshot(0, "NVIDIA L40S", 46068)
L40S_B = GpuSnapshot(1, "NVIDIA L40S", 46068)
T4 = GpuSnapshot(0, "Tesla T4", 15360)
T4_B = GpuSnapshot(1, "Tesla T4", 15360)


def host(
    *,
    arch: str = "x86_64",
    physical_cores: int | None = 64,
    logical_cpus: int | None = 128,
    ram_gb: float | None = 256.0,
    gpus: tuple[GpuSnapshot, ...] = (H100, H100_B),
    nvidia_error: str | None = None,
) -> HostSnapshot:
    return HostSnapshot(
        arch=arch,
        physical_cores=physical_cores,
        logical_cpus=logical_cpus,
        ram_gb=ram_gb,
        gpus=gpus,
        nvidia_error=nvidia_error,
    )


def test_requirement_names_match_family_enum() -> None:
    from ultron.train.capability import REQUIREMENTS

    assert {item.name for item in REQUIREMENTS} == {item.value for item in FamilyName}


def test_parse_physical_cores_counts_unique_sockets() -> None:
    assert parse_physical_cores(CPUINFO_TWO_CORE_HT) == 2
    assert parse_physical_cores("processor : 0\nprocessor : 1\n") == 2
    assert parse_physical_cores("") is None


def test_parse_ram_and_nvidia_csv() -> None:
    assert parse_ram_gb("MemTotal:       264134656 kB\n") == pytest.approx(251.9, abs=0.1)
    gpus = parse_nvidia_smi("0, NVIDIA H100 80GB HBM3, 81559\n1, NVIDIA H100 80GB HBM3, 81559\n")
    assert len(gpus) == 2
    assert gpus[0].name.startswith("NVIDIA H100")
    assert gpus[0].memory_gb == pytest.approx(79.6, abs=0.1)
    assert parse_nvidia_smi("0, broken, N/A\n") == ()


def test_full_h100_host_runs_every_family() -> None:
    report = assess(host())
    assert report.runnable == ("qwen-4b", "qwen-8b", "gemma")
    assert all(item.fit is Fit.RUN for item in report.verdicts)
    assert report_ok(report)
    text = render(report)
    assert "Runnable: qwen-4b, qwen-8b, gemma" in text
    assert "GPU 0" in text


def test_l40s_is_caution_for_8b_and_gemma() -> None:
    report = assess(host(gpus=(L40S, L40S_B)))
    by_name = {item.requirement.name: item for item in report.verdicts}
    assert by_name["qwen-4b"].fit is Fit.SKIP
    assert by_name["qwen-8b"].fit is Fit.CAUTION
    assert by_name["gemma"].fit is Fit.CAUTION
    assert report.runnable == ("qwen-8b", "gemma")
    assert "cut guest concurrency" in "; ".join(by_name["qwen-8b"].reasons)


def test_24gb_cards_and_one_gpu_are_skips() -> None:
    small = assess(host(gpus=(T4, T4_B)))
    assert small.runnable == ()
    assert all(item.fit is Fit.SKIP for item in small.verdicts)
    single = assess(host(gpus=(H100,)))
    assert all(any("1 GPUs < 2" in reason for reason in item.reasons) for item in single.verdicts)


def test_ram_and_arch_gates() -> None:
    tight = assess(host(ram_gb=128.0))
    by_name = {item.requirement.name: item for item in tight.verdicts}
    assert by_name["qwen-4b"].fit is Fit.RUN
    assert by_name["qwen-8b"].fit is Fit.SKIP
    assert by_name["gemma"].fit is Fit.SKIP
    arm = assess(host(arch="aarch64"))
    assert arm.runnable == ()
    assert "not x86-64" in arm.verdicts[0].reasons[0]


def test_missing_nvidia_skips() -> None:
    report = assess(host(gpus=(), nvidia_error="nvidia-smi not found"))
    assert report.runnable == ()
    assert "nvidia-smi not found" in render(report)


def test_selected_family_exit_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ultron.train.capability.read_host",
        lambda: host(ram_gb=128.0),
    )
    assert main([]) == 0
    assert main(["--family", "qwen-4b"]) == 0
    assert main(["--family", "gemma"]) == 1
    assert main(["--family", "llama-8b"]) == 2


def test_json_payload_and_unknown_family(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("ultron.train.capability.read_host", lambda: host())
    assert main(["--json", "--family", "qwen-8b"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["selected"] == "qwen-8b"
    assert payload["runnable"] == ["qwen-4b", "qwen-8b", "gemma"]
    with pytest.raises(CapabilityError, match="unknown model family"):
        parse_family_name("llama-8b")


def test_read_host_accepts_injected_inventory() -> None:
    snapshot = read_host(
        cpuinfo=CPUINFO_TWO_CORE_HT,
        meminfo="MemTotal: 132120576 kB\n",
        nvidia_csv="0, NVIDIA A100-SXM4-80GB, 81920\n1, NVIDIA A100-SXM4-80GB, 81920\n",
        arch="x86_64",
        logical_cpus=4,
    )
    assert snapshot.physical_cores == 2
    assert snapshot.ram_gb == pytest.approx(126.0, abs=0.2)
    assert len(snapshot.gpus) == 2
    report = assess(snapshot)
    assert report.runnable == ()
    assert "2 physical cores < 32" in report.verdicts[0].reasons[0]


def test_module_and_file_entrypoints_agree() -> None:
    file_run = subprocess.run(
        [sys.executable, str(ROOT / "train" / "capability.py"), "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    module_run = subprocess.run(
        [sys.executable, "-m", "ultron.train.capability", "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert file_run.returncode in {0, 1}, file_run.stderr
    assert module_run.returncode == file_run.returncode, module_run.stderr
    assert json.loads(file_run.stdout)["verdicts"] == json.loads(module_run.stdout)["verdicts"]


def test_lib_capability_runs_before_host_gates() -> None:
    script = """
    set -euo pipefail
    source scripts/lib_capability.sh
    type ultron_check_model_capability
    printf 'wired\\n'
    """
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ultron_check_model_capability" in result.stdout
    assert "wired" in result.stdout
    bm = (ROOT / "scripts" / "bootstrap_bm.sh").read_text()
    cloud = (ROOT / "scripts" / "bootstrap_cloud.sh").read_text()
    for text in (bm, cloud):
        assert "lib_capability.sh" in text
        assert "ultron_check_model_capability" in text
        assert text.index("ultron_check_model_capability") < text.index("nvidia-smi")


def test_skip_env_bypasses_the_check() -> None:
    result = subprocess.run(
        ["bash", "-c", "set -euo pipefail; source scripts/lib_capability.sh; ultron_check_model_capability"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "ULTRON_SKIP_MODEL_CAPABILITY": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "Skipping model capability check" in result.stdout


def test_bootstrap_cloud_stops_before_host_gates_when_unfit() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "bootstrap_cloud.sh")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={key: value for key, value in os.environ.items() if key != "ULTRON_SKIP_MODEL_CAPABILITY"},
    )
    text = result.stdout + result.stderr
    assert "=== Model capability ===" in text
    if "No supported family fits this host." in text:
        assert result.returncode == 1
        assert "Cloud host gates failed" not in text
        assert "MISSING: docker" not in text
        assert "MISSING: nvidia-smi" not in text


def test_payload_marks_selected_skip() -> None:
    report = assess(host(gpus=()), family="qwen-4b")
    payload = to_payload(report)
    assert payload["ok"] is False
    assert payload["selected"] == "qwen-4b"
    assert payload["runnable"] == []
