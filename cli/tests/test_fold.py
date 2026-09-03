import pytest

from ultron.cli.demo import default_spec, scripted_guest
from ultron.cli.fold import BoardFold, IllegalTransition
from ultron.train.schema_v1 import Role


def test_illegal_turn_before_restore() -> None:
    fold = BoardFold(default_spec(), clock=lambda: 0.0)
    with pytest.raises(IllegalTransition):
        fold.apply_turn_start(Role.ATTACKER, 0)


def test_even_turn_rejects_defender_role() -> None:
    fold = BoardFold(default_spec(), clock=lambda: 0.0)
    guest = scripted_guest()
    fold.apply_restore_start(guest, "a" * 64)
    fold.apply_restore_done(guest)
    with pytest.raises(IllegalTransition):
        fold.apply_turn_start(Role.DEFENDER, 0)
