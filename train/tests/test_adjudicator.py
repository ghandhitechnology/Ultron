import pytest

from ultron.train.adjudicator import ProbeResult, adjudicate
from ultron.train.schema_v1 import ReasonCode


@pytest.mark.parametrize(
    ("probe", "expected"),
    [
        (
            ProbeResult(0, True, False, False, True),
            (ReasonCode.INFRA_FAIL, 0.0, 0.0),
        ),
        (
            ProbeResult(0, True, False, True, True),
            (ReasonCode.ATTACKER_ROOT, 1.0, 0.0),
        ),
        (
            ProbeResult(1000, False, False, True, True),
            (ReasonCode.AVAILABILITY_FAIL, 0.0, 0.0),
        ),
        (
            ProbeResult(1000, False, True, True, True),
            (ReasonCode.TIMEOUT, 0.0, 1.0),
        ),
        (
            ProbeResult(1000, False, True, True, False),
            (ReasonCode.DEFENDER_HOLD, 0.0, 1.0),
        ),
    ],
)
def test_adjudicator_precedence(
    probe: ProbeResult, expected: tuple[ReasonCode, float, float]
) -> None:
    assert adjudicate(probe) == expected


def test_guest_root_without_host_confirmation_does_not_win() -> None:
    probe = ProbeResult(0, False, True, True, False)
    assert adjudicate(probe)[0] == ReasonCode.DEFENDER_HOLD
