from ultron.train.rae import RoleAwareBaseline, group_centered_advantages
from ultron.train.schema_v1 import Role


def test_default_group_center_does_not_use_role_baseline() -> None:
    rewards = [1.0, 0.0, 0.5]
    centered = group_centered_advantages(rewards)
    assert centered == [0.5, -0.5, 0.0]


def test_role_baseline_cancels_under_group_centering() -> None:
    baseline = RoleAwareBaseline(alpha=0.0)
    baseline.update(Role.ATTACKER, 1.0)
    with_baseline = group_centered_advantages(
        [1.0, 0.0], role=Role.ATTACKER, baseline=baseline
    )
    without = group_centered_advantages([1.0, 0.0])
    assert with_baseline == without
