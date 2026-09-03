from ultron.cli.ascii import render
from ultron.cli.job import LiveJob


def test_trading_ascii_shows_colosseum_and_pending_probe() -> None:
    board = LiveJob.demo("hold-at-turn-7").board_at_end()
    text = render(board)
    assert "ATTACKER" in text
    assert "DEFENDER" in text
    assert "GUEST" in text
    assert "PROCESS" in text
    assert "——" in text
    assert "ETA" in text


def test_settled_ascii_shows_euid() -> None:
    board = LiveJob.demo("attacker-root").board_at_end()
    text = render(board)
    assert "ATTACKER_ROOT" in text or "COMPROMISED" in text
    assert "euid 0" in text
