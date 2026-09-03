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
    assert ("ULTRON_MODEL_FAMILY", "qwen-4b") in built.env
    assert built.argv[1:3] == ("--family", "qwen-4b")


def test_family_pin_follows_the_selector() -> None:
    built = plan(
        ActionId.GENERATION,
        {"generation": "1", "episodes": "8"},
        root=ROOT,
        family="gemma",
    )
    assert isinstance(built, TmuxPlan)
    assert ("ULTRON_MODEL_FAMILY", "gemma") in built.env
    assert built.argv[1:3] == ("--family", "gemma")
    review = plan(ActionId.REVIEW, {"generation": "1", "phase": "complete"}, root=ROOT, family="gemma")
    assert isinstance(review, ForegroundPlan)
    assert ("ULTRON_MODEL_FAMILY", "gemma") in review.env
    assert any(str(part).endswith("data/families/gemma/archives") for part in review.argv)
    assert any(str(part).endswith("data/families/gemma/checkpoints/pfsp_pool.json") for part in review.argv)


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(CatalogError, match="unknown model family"):
        plan(ActionId.TESTS, {"suite": "cli"}, root=ROOT, family="llama-8b")


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


def test_all_tests_score_archives_after_pytest() -> None:
    built = plan(ActionId.TESTS, {"suite": "all"}, root=ROOT)
    assert isinstance(built, ForegroundPlan)
    assert built.argv[0].endswith("run_tests.sh")
    assert built.argv[-1] == "all"
    assert ("ULTRON_MODEL_FAMILY", "qwen-4b") in built.env


def test_benchmarks_plan_reads_archives_only() -> None:
    built = plan(ActionId.BENCHMARKS, {}, root=ROOT)
    assert isinstance(built, ForegroundPlan)
    assert "--archive-dir" in built.argv
    assert "--checkpoint-root" not in built.argv
    assert "--all" in built.argv
    assert any(str(part).endswith("data/archives") for part in built.argv)
    execute = plan(ActionId.BENCHMARKS, {"execute": "true", "all_generations": "false", "generation": "3"}, root=ROOT)
    assert isinstance(execute, TmuxPlan)
    assert execute.session == "ultron-bench-gen3"
    assert "--execute" in execute.argv
    assert "--generation" in execute.argv
