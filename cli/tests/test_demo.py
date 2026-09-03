from ultron.cli.board import Done, Trading
from ultron.cli.job import LiveJob
from ultron.train.schema_v1 import ReasonCode


def test_attacker_root_settles() -> None:
    board = LiveJob.demo("attacker-root").board_at_end()
    assert isinstance(board.phase, Done)
    assert board.phase.last.terminal.reason_code is ReasonCode.ATTACKER_ROOT
    assert board.phase.last.terminal.attacker_euid == 0


def test_hold_at_turn_7_is_trading() -> None:
    board = LiveJob.demo("hold-at-turn-7").board_at_end()
    assert isinstance(board.phase, Trading)
    assert board.phase.cursor.index == 7
    assert board.phase.cursor.side.value == "defender"
    assert board.reason_code is None


def test_infra_fail_reason() -> None:
    board = LiveJob.demo("infra-fail").board_at_end()
    assert isinstance(board.phase, Done)
    assert board.phase.last.terminal.reason_code.value == "INFRA_FAIL"
