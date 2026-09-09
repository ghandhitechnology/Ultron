import random

import pytest

from ultron.train.pfsp import PoolEntry, sample_adaptive_history, sample_study_opponent
from ultron.train.schema_v1 import Role
from ultron.train.study import AdaptiveMix, Method


def _entry(checkpoint_id: str, win: float = 0.5) -> PoolEntry:
    return PoolEntry(checkpoint_id, f"/{checkpoint_id}", Role.DEFENDER, win)


MIX = AdaptiveMix(0.5, 0.25, 0.25)


def test_fixed_and_latest_methods() -> None:
    single = _entry("single")
    latest = _entry("latest")
    historical = [_entry("h1"), _entry("h2")]
    reference = [_entry("r1"), _entry("r2")]
    rng = random.Random(0)
    kwargs = dict(
        single=single,
        latest=latest,
        historical=historical,
        reference_pool=reference,
        learner_relative={"h1": 0.9, "h2": 0.1},
        rng=rng,
        mix=MIX,
    )
    assert sample_study_opponent(Method.FIXED_SINGLE, **kwargs).checkpoint_id == "single"
    assert sample_study_opponent(Method.ADAPTIVE_LATEST, **kwargs).checkpoint_id == "latest"
    diverse = {sample_study_opponent(Method.FIXED_DIVERSE, **kwargs).checkpoint_id for _ in range(20)}
    assert diverse <= {"r1", "r2"}
    with pytest.raises(ValueError, match="untrained"):
        sample_study_opponent(Method.UNTRAINED, **kwargs)


def test_adaptive_history_mix_is_fifty_twenty_five_twenty_five() -> None:
    latest = _entry("latest")
    historical = [_entry("hard", 0.9), _entry("easy", 0.1)]
    counts = {"latest": 0, "hard": 0, "easy": 0}
    rng = random.Random(1)
    for _ in range(4000):
        chosen = sample_adaptive_history(
            latest=latest,
            historical=historical,
            learner_relative={"hard": 0.9, "easy": 0.1},
            rng=rng,
            mix=MIX,
        )
        counts[chosen.checkpoint_id] += 1
    total = sum(counts.values())
    assert abs(counts["latest"] / total - 0.5) < 0.05
    assert counts["hard"] > counts["easy"]
