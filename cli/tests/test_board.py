from ultron.cli.board import TurnCursor, Trading
from ultron.cli.job import LiveJob
from ultron.train.schema_v1 import Role


def test_even_turn_is_attacker() -> None:
    cursor = TurnCursor.of(0, 8)
    assert cursor.side is Role.ATTACKER
    assert TurnCursor.of(1, 8).side is Role.DEFENDER


def test_trading_has_no_terminal() -> None:
    board = LiveJob.demo("hold-at-turn-7").board_at_end()
    assert isinstance(board.phase, Trading)
    assert board.reason_code is None
