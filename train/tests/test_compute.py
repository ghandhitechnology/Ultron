import json
import sys
from dataclasses import replace

import pytest

from ultron.train.compute import AllocationLedger, compare, load_hardware, run_command, summarize, verify_hardware, gpu_lease


def allocation(path, *, start, seconds, identifier="one", status=0, profile=None):
    ledger = AllocationLedger(path)
    profile = profile or replace(load_hardware(), instance_hourly_rate=8520,
                                 rate_source="test fixture", rate_observed_at="2026-09-11")
    ledger.begin(profile, run_id="test", phase="whole-run", scope_key="same-scope",
                 allocation_id=identifier, resource_id="test-host")
    ledger.end(identifier, status=status, elapsed_seconds=seconds)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-2]["time"], rows[-1]["time"] = start, start + seconds
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_cost_counts_entire_pair_including_failed_idle_attempt(tmp_path):
    path = tmp_path / "ledger.jsonl"
    allocation(path, start=0, seconds=3600, status=1)
    allocation(path, start=3600, seconds=3600, identifier="retry")
    report = summarize(path, extra_cost=1000)
    assert report["gpu_hours"] == 4
    assert report["elapsed_hours"] == 2
    assert report["total_cost"] == 18040
    assert report["failed_attempts"] == 1


def test_open_allocation_counts_time_and_cannot_pass_target(tmp_path):
    path = tmp_path / "ledger.jsonl"
    allocation(path, start=0, seconds=3600)
    baseline = summarize(path)
    rows = path.read_text().splitlines()
    path.write_text(rows[0] + "\n")
    report = summarize(path, now=900)
    assert report["gpu_hours"] == 0.5
    assert report["open_allocations"] == ["one"]
    assert not compare(baseline, report)["meets_half_resource_target"]


def test_duplicate_allocation_of_same_devices_rejected(tmp_path):
    path = tmp_path / "ledger.jsonl"
    allocation(path, start=0, seconds=3600)
    allocation(path, start=1, seconds=1000, identifier="overlap")
    with pytest.raises(ValueError, match="overlapping allocations"):
        summarize(path)


def test_half_gpu_time_with_fixed_cost_misses_half_cost(tmp_path):
    baseline, candidate = tmp_path / "before", tmp_path / "after"
    allocation(baseline, start=0, seconds=3600)
    allocation(candidate, start=0, seconds=1800)
    report = compare(summarize(baseline, extra_cost=1000), summarize(candidate, extra_cost=1000))
    assert report["ratios"]["gpu_hours"] == 0.5
    assert report["ratios"]["total_cost"] > 0.5
    assert not report["meets_half_resource_target"]


def test_command_failure_is_durable_and_rate_is_unknown_by_default(tmp_path):
    path = tmp_path / "ledger"
    status = run_command(AllocationLedger(path), load_hardware(),
                         [sys.executable, "-c", "raise SystemExit(7)"],
                         run_id="fixture", phase="allocation", scope_key="fixture", resource_id="fixture")
    assert status == 7
    report = summarize(path)
    assert report["failed_attempts"] == 1
    assert report["open_allocations"] == []
    assert report["total_cost"] is None


def test_h100_profile_rejects_different_physical_gpu(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("ultron.train.compute.subprocess.run", lambda *args, **kwargs: SimpleNamespace(
        stdout="0, GPU-one, NVIDIA H100 80GB HBM3, 81559\n1, GPU-two, NVIDIA A100-SXM4-80GB, 81920\n"))
    with pytest.raises(ValueError, match="not an H100"):
        verify_hardware(load_hardware())


def test_gpu_lease_rejects_overlapping_jobs_and_releases_on_failure(tmp_path):
    profile = load_hardware()
    with pytest.raises(RuntimeError):
        with gpu_lease(profile, directory=tmp_path):
            with pytest.raises(ValueError, match="another Ultron allocation"):
                with gpu_lease(profile, directory=tmp_path):
                    pytest.fail("overlapping allocation was admitted")
            raise RuntimeError("interrupted allocation")
    with gpu_lease(profile, directory=tmp_path):
        pass


def test_observed_uuid_prevents_index_alias_and_resource_label_double_count(tmp_path):
    path = tmp_path / "ledger"
    observed = tuple({"uuid": f"GPU-{index}", "index": str(index),
                      "name": "NVIDIA H100 80GB HBM3", "memory_mib": "81559"} for index in range(2))
    profile = replace(load_hardware(), observed_devices=observed)
    allocation(path, start=0, seconds=100, profile=profile)
    allocation(path, start=1, seconds=100, identifier="overlap",
               profile=replace(profile, devices=("GPU-0", "GPU-1")))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[2]["resource_id"] = "another-billing-label"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="overlapping allocations"):
        summarize(path)


def test_unverified_hardware_is_labelled_as_declared(tmp_path):
    path = tmp_path / "ledger"
    allocation(path, start=0, seconds=100)
    report = summarize(path)
    assert not report["hardware_observed"]
    assert report["gpu_models"] == []
    assert report["declared_gpu_models"] == ["NVIDIA H100 80GB"]
    assert not compare(report, report)["meets_half_resource_target"]
