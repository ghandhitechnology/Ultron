from ultron.train.adjudicator import ProbeResult
from ultron.train.gates import (
    dummy_service_is_not_availability,
    false_root_is_rejected,
    known_patched_holds,
    known_solvable_holds,
    observation_leaks_metadata,
    timeout_is_unresolved,
)


def test_false_root_and_timeout_gates() -> None:
    false_root = ProbeResult(0, False, True, True, False)
    assert false_root_is_rejected(false_root) is True
    timeout = ProbeResult(1000, False, True, True, True)
    assert timeout_is_unresolved(timeout) is True
    hang = ProbeResult(1000, False, True, True, False, hung=True)
    assert timeout_is_unresolved(hang) is True


def test_metadata_dummy_and_known_tasks() -> None:
    assert observation_leaks_metadata("ls /tmp", frozenset({"suid_helper_ground_truth"})) is False
    assert observation_leaks_metadata("suid_helper_ground_truth", frozenset({"suid_helper_ground_truth"}))
    assert dummy_service_is_not_availability(real_ok=False, dummy_ok=True) is False
    assert dummy_service_is_not_availability(real_ok=True, dummy_ok=True) is True
    assert known_solvable_holds(launcher_ran=True, attacker_won=True)
    assert known_patched_holds(launcher_ran=True, attacker_won=False)
    assert known_solvable_holds(launcher_ran=False, attacker_won=True) is False
