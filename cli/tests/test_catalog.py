from pathlib import Path

import pytest

from ultron.cli.catalog import (
    ActionId,
    CatalogError,
    ForegroundPlan,
    GymPlan,
    TmuxPlan,
    all_actions,
    parse_values,
    plan,
    spec_for,
)
from ultron.env.backend import IsolationBackend


ROOT = Path(__file__).resolve().parents[2]


def test_catalog_covers_every_action() -> None:
    ids = {spec.id for spec in all_actions(root=ROOT)}
    assert ids == set(ActionId)


def test_default_plans_cover_every_action() -> None:
    kinds = []
    for spec in all_actions(root=ROOT):
        raw = {field.key: field.default for field in spec.fields}
        built = plan(spec.id, raw, root=ROOT)
        kinds.append(built.kind)
        if spec.id is ActionId.DEMO:
            assert isinstance(built, GymPlan)
            assert built.meta.isolation is IsolationBackend.DOCKER
        elif spec.id in {
            ActionId.GENERATION,
            ActionId.ROLLOUT,
            ActionId.GRPO,
            ActionId.DPO,
            ActionId.SERVE_ATTACKER,
            ActionId.SERVE_DEFENDER,
        }:
            assert isinstance(built, TmuxPlan)
            assert built.session
        else:
            assert isinstance(built, ForegroundPlan)
            assert built.argv[0]
    assert "gym" in kinds
    assert "tmux" in kinds
    assert "foreground" in kinds


def test_generation_plan_sets_episode_env() -> None:
    built = plan(ActionId.GENERATION, {"generation": "3", "episodes": "16"}, root=ROOT)
    assert isinstance(built, TmuxPlan)
    assert built.session == "ultron-gen-3"
    assert built.argv[-1] == "3"
    assert ("ULTRON_EPISODES", "16") in built.env


def test_dpo_rejects_early_generation() -> None:
    with pytest.raises(CatalogError, match="generation 2"):
        plan(ActionId.DPO, {"role": "attacker", "generation": "1"}, root=ROOT)


def test_invalid_choice_is_rejected() -> None:
    spec = spec_for(ActionId.TESTS, root=ROOT)
    with pytest.raises(CatalogError, match="Suite"):
        parse_values(spec, {"suite": "gpu"})


def test_review_defaults_to_generation_traces() -> None:
    built = plan(ActionId.REVIEW, {"generation": "4", "phase": "complete"}, root=ROOT)
    assert isinstance(built, ForegroundPlan)
    assert "data/traces/gen4" in built.argv[3]
    assert "--eval-dir" in built.argv
    assert "--archive-dir" in built.argv
    assert "--pfsp" in built.argv


def test_tests_plan_selects_suite_paths() -> None:
    built = plan(ActionId.TESTS, {"suite": "cli"}, root=ROOT)
    assert isinstance(built, ForegroundPlan)
    assert built.argv[-2:] == ("cli/tests", "-q")
