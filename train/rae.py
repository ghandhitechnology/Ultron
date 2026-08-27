from dataclasses import dataclass, field

from .schema_v1 import Role


@dataclass
class RoleAwareBaseline:
    alpha: float = 0.95
    mu: dict[Role, float] = field(
        default_factory=lambda: {Role.ATTACKER: 0.0, Role.DEFENDER: 0.0}
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")

    def update(self, role: Role, reward: float) -> None:
        self.mu[role] = self.alpha * self.mu[role] + (1.0 - self.alpha) * reward

    def advantage(self, role: Role, reward: float) -> float:
        return reward - self.mu[role]


def group_centered_advantages(
    rewards: list[float], role: Role, baseline: RoleAwareBaseline
) -> list[float]:
    if not rewards:
        return []
    adjusted = [baseline.advantage(role, reward) for reward in rewards]
    mean = sum(adjusted) / len(adjusted)
    return [value - mean for value in adjusted]
