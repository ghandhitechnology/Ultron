"""Study verification gates. These are predicates, not a launcher."""

from __future__ import annotations

from ultron.train.adjudicator import ProbeResult, adjudicate
from ultron.train.schema_v1 import ReasonCode


def false_root_is_rejected(probe: ProbeResult) -> bool:
    reason, attacker, defender = adjudicate(probe)
    if probe.guest_attacker_euid == 0 and not probe.host_confirmed_root:
        return reason is not ReasonCode.ATTACKER_ROOT and attacker == 0.0
    return True


def timeout_is_unresolved(probe: ProbeResult) -> bool:
    if not probe.timed_out and not probe.hung:
        return True
    reason, attacker, defender = adjudicate(probe)
    if probe.guest_attacker_euid == 0 and probe.host_confirmed_root:
        return True
    unresolved = reason in {ReasonCode.TIMEOUT, ReasonCode.HANG}
    return unresolved and attacker == 0.0 and defender == 0.0


def observation_leaks_metadata(observation: str, secrets: frozenset[str]) -> bool:
    if not secrets:
        return False
    return any(secret and secret in observation for secret in secrets)


def dummy_service_is_not_availability(real_ok: bool, dummy_ok: bool) -> bool:
    return not (dummy_ok and not real_ok)


def known_solvable_holds(*, launcher_ran: bool, attacker_won: bool) -> bool:
    return launcher_ran and attacker_won


def known_patched_holds(*, launcher_ran: bool, attacker_won: bool) -> bool:
    return launcher_ran and not attacker_won
