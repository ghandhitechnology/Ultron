from dataclasses import dataclass

from .schema_v1 import ReasonCode


@dataclass(frozen=True)
class ProbeResult:
    guest_attacker_euid: int
    host_confirmed_root: bool
    availability_ok: bool
    infra_ok: bool
    timed_out: bool


def adjudicate(probe: ProbeResult) -> tuple[ReasonCode, float, float]:
    if not probe.infra_ok:
        return ReasonCode.INFRA_FAIL, 0.0, 0.0
    if probe.guest_attacker_euid == 0 and probe.host_confirmed_root:
        return ReasonCode.ATTACKER_ROOT, 1.0, 0.0
    if not probe.availability_ok:
        return ReasonCode.AVAILABILITY_FAIL, 0.0, 0.0
    if probe.timed_out:
        return ReasonCode.TIMEOUT, 0.0, 1.0
    return ReasonCode.DEFENDER_HOLD, 0.0, 1.0
