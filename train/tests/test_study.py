import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ultron.env.backend import IsolationBackend
from ultron.train.family import FamilyName, resolve
from ultron.train.study import (
    PROTOCOL_VERSION,
    Ablation,
    EvalSplit,
    ExecutionGateError,
    Method,
    ResultSource,
    RunIdentity,
    ScoreIdentity,
    StudyError,
    assert_comparable,
    assert_paid_execution_allowed,
    check_configs,
    load_protocol,
    main,
    study_matrix,
)

ROOT = Path(__file__).resolve().parents[2]


def test_matrix_matches_the_locked_plan() -> None:
    protocol = load_protocol(repo_root=ROOT)
    eight = protocol.cells_for(FamilyName.QWEN_8B)
    four = protocol.cells_for(FamilyName.QWEN_4B)
    assert [cell.variant for cell in eight] == [
        "fixed-single",
        "fixed-diverse",
        "adaptive-latest",
        "adaptive-history",
        "untrained",
        "fixed-diverse+shaping-off",
        "fixed-diverse+added-dpo",
        "adaptive-history+shaping-off",
        "adaptive-history+added-dpo",
    ]
    assert [cell.variant for cell in four] == [
        "fixed-diverse",
        "adaptive-history",
        "untrained",
    ]
    assert protocol.default_family is FamilyName.QWEN_8B
    assert protocol.secondary_family is FamilyName.QWEN_4B
    assert protocol.target_delta_pp == 5
    assert protocol.isolation is IsolationBackend.KVM
    assert protocol.splits == (
        EvalSplit.UNSEEN_INSTANCE,
        EvalSplit.UNSEEN_COMBINATION,
        EvalSplit.HELD_OUT_MECHANISM,
    )
    primary = [cell for cell in eight if cell.primary]
    assert len(primary) == 1
    assert primary[0].method is Method.ADAPTIVE_HISTORY
    assert primary[0].ablation is Ablation.NONE
    assert primary[0].shaping is True
    assert primary[0].dpo is False
    shaping_off = next(cell for cell in eight if cell.variant == "adaptive-history+shaping-off")
    assert shaping_off.shaping is False
    added_dpo = next(cell for cell in eight if cell.variant == "fixed-diverse+added-dpo")
    assert added_dpo.dpo is True
    assert all(cell.ablation is Ablation.NONE for cell in four)


def test_final_test_tasks_are_sealed_and_development_is_separate() -> None:
    protocol = load_protocol(repo_root=ROOT)
    train_ids = {task.task_id for task in protocol.tasks if task.split == "train"}
    dev_ids = {task.task_id for task in protocol.tasks if task.development}
    sealed = [task for task in protocol.tasks if task.sealed]
    assert train_ids == {"suid_helper", "writable_cron", "suid_and_cron"}
    assert "dev_writable_tmp" in dev_ids
    assert train_ids.isdisjoint(dev_ids)
    assert protocol.final_test_sealed is True
    assert {task.split for task in sealed} == {
        "unseen_instance",
        "unseen_combination",
        "held_out_mechanism",
    }


def test_paid_execution_requires_a_positive_kvm_cap() -> None:
    protocol = load_protocol(repo_root=ROOT)
    with pytest.raises(ExecutionGateError, match="spending cap"):
        assert_paid_execution_allowed(None, protocol)
    with pytest.raises(ExecutionGateError, match="positive"):
        assert_paid_execution_allowed(0, protocol)
    with pytest.raises(ExecutionGateError, match="kvm"):
        assert_paid_execution_allowed(100, protocol, isolation=IsolationBackend.DOCKER)
    assert_paid_execution_allowed(100, protocol, isolation=IsolationBackend.KVM)


def test_mock_scores_cannot_be_compared_with_measured() -> None:
    run = RunIdentity(PROTOCOL_VERSION, "qwen-8b", "adaptive-history", 0, "r1")
    measured = ScoreIdentity(run, "unseen_instance", "attacker", ResultSource.MEASURED)
    mocked = ScoreIdentity(run, "unseen_instance", "attacker", ResultSource.MOCK)
    assert measured.key() != mocked.key()
    assert run.namespace() == Path("data/study/qwen-8b/adaptive-history/seed0/r1")
    with pytest.raises(StudyError, match="mock"):
        assert_comparable(measured, mocked)


def test_check_and_matrix_cli() -> None:
    assert main(["matrix"]) == 0
    failures = check_configs(repo_root=ROOT)
    assert failures == []
    env = {k: v for k, v in os.environ.items() if k != "ULTRON_MODEL_FAMILY"}
    result = subprocess.run(
        [sys.executable, "-m", "ultron.train.study", "check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert PROTOCOL_VERSION in result.stdout
    listed = subprocess.run(
        [sys.executable, "-m", "ultron.train.study", "matrix"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    rows = json.loads(listed.stdout)
    assert len(rows) == 12
    assert resolve(environ={}).name is FamilyName.QWEN_8B


def test_study_matrix_helper_matches_load() -> None:
    assert study_matrix() == load_protocol().cells
