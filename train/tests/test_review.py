import json
from pathlib import Path

import pytest

from ultron.env.backend import IsolationBackend
from ultron.train.review import (
    Phase,
    Verdict,
    review_traces,
    write_review,
)
from ultron.train.schema_v1 import (
    SCHEMA_VERSION,
    ReasonCode,
    Role,
    TerminalOutcome,
    ToolEvent,
    TrajectoryStep,
    TrajectoryV1,
)


def terminal(
    reason: ReasonCode,
    *,
    attacker_reward: float | None = None,
    defender_reward: float | None = None,
    euid: int | None = None,
    host_root: bool | None = None,
    availability_ok: bool | None = None,
) -> TerminalOutcome:
    if reason is ReasonCode.ATTACKER_ROOT:
        attacker_reward = 1.0 if attacker_reward is None else attacker_reward
        defender_reward = 0.0 if defender_reward is None else defender_reward
        euid = 0 if euid is None else euid
        host_root = True if host_root is None else host_root
        availability_ok = True if availability_ok is None else availability_ok
    elif reason is ReasonCode.INFRA_FAIL:
        attacker_reward = 0.0 if attacker_reward is None else attacker_reward
        defender_reward = 0.0 if defender_reward is None else defender_reward
        euid = 1000 if euid is None else euid
        host_root = False if host_root is None else host_root
        availability_ok = True if availability_ok is None else availability_ok
    elif reason is ReasonCode.AVAILABILITY_FAIL:
        attacker_reward = 0.0 if attacker_reward is None else attacker_reward
        defender_reward = 0.0 if defender_reward is None else defender_reward
        euid = 1000 if euid is None else euid
        host_root = False if host_root is None else host_root
        availability_ok = False if availability_ok is None else availability_ok
    else:
        attacker_reward = 0.0 if attacker_reward is None else attacker_reward
        defender_reward = 1.0 if defender_reward is None else defender_reward
        euid = 1000 if euid is None else euid
        host_root = False if host_root is None else host_root
        availability_ok = True if availability_ok is None else availability_ok
    return TerminalOutcome(
        reason_code=reason,
        attacker_euid=euid,
        host_confirmed_root=host_root,
        availability_ok=availability_ok,
        attacker_reward=attacker_reward,
        defender_reward=defender_reward,
    )


def step(
    turn_index: int,
    *,
    role: Role = Role.ATTACKER,
    valid: bool = True,
    hits: list[str] | None = None,
    tools: list[ToolEvent] | None = None,
    decision_point: bool = False,
) -> TrajectoryStep:
    return TrajectoryStep(
        turn_index=turn_index,
        side=role,
        prompt_token_ids=[turn_index],
        assistant_token_ids=[turn_index],
        assistant_mask=[1],
        tool_events=tools or [],
        subgoal_hits=hits or [],
        format_valid=valid,
        decision_point=decision_point,
        observation_hash=f"obs-{turn_index}",
    )


def trajectory(
    *,
    episode_id: str,
    role: Role,
    reason: ReasonCode = ReasonCode.DEFENDER_HOLD,
    generation: int = 0,
    profile_id: str = "web",
    group_id: str = "group-1",
    opponent: str = "defender-gen0",
    steps: list[TrajectoryStep] | None = None,
    outcome: TerminalOutcome | None = None,
) -> TrajectoryV1:
    side_steps = steps or [
        step(
            0,
            role=role,
            tools=[
                ToolEvent(
                    name="bash",
                    args={"command": "id"},
                    stdout_head="uid=1000",
                    stdout_tail="uid=1000",
                    exit_code=0,
                    duration_ms=5,
                )
            ],
        )
    ]
    return TrajectoryV1(
        schema_version=SCHEMA_VERSION,
        episode_id=episode_id,
        generation=generation,
        profile_id=profile_id,
        role=role,
        adapter_id=f"{role.value}_lora",
        opponent_checkpoint_id=opponent,
        group_id=group_id,
        steps=side_steps,
        terminal=outcome or terminal(reason),
        isolation_backend=IsolationBackend.KVM,
    )


def write_roles(traces: Path, episodes: list[tuple[TrajectoryV1, TrajectoryV1]]) -> None:
    traces.mkdir(parents=True, exist_ok=True)
    attacker = traces / "attacker.jsonl"
    defender = traces / "defender.jsonl"
    with attacker.open("w") as atk, defender.open("w") as dfn:
        for attacker_traj, defender_traj in episodes:
            atk.write(json.dumps(attacker_traj.to_dict()) + "\n")
            dfn.write(json.dumps(defender_traj.to_dict()) + "\n")


def pair(
    episode_id: str,
    reason: ReasonCode,
    **kwargs: object,
) -> tuple[TrajectoryV1, TrajectoryV1]:
    attacker = trajectory(episode_id=episode_id, role=Role.ATTACKER, reason=reason, **kwargs)
    defender = trajectory(episode_id=episode_id, role=Role.DEFENDER, reason=reason, **kwargs)
    return attacker, defender


def finding_codes(review) -> set[str]:
    return {item.code for item in review.findings}


def test_empty_traces_are_unusable(tmp_path: Path) -> None:
    traces = tmp_path / "gen0"
    traces.mkdir()
    review = review_traces(traces, phase=Phase.ROLLOUT, generation=0)
    assert review.verdict is Verdict.UNUSABLE
    assert "empty_traces" in finding_codes(review)


def test_paired_asr_and_reason_mix(tmp_path: Path) -> None:
    traces = tmp_path / "gen0"
    write_roles(
        traces,
        [
            pair("ep-1", ReasonCode.ATTACKER_ROOT),
            pair("ep-2", ReasonCode.DEFENDER_HOLD),
        ],
    )
    (traces / "metrics.json").write_text(json.dumps({"asr": 0.5}) + "\n")
    review = review_traces(traces, phase=Phase.ROLLOUT, generation=0)
    assert review.outcomes.episode_count == 2
    assert review.outcomes.asr == pytest.approx(0.5)
    assert review.outcomes.reason_counts[ReasonCode.ATTACKER_ROOT.value] == 1
    assert review.outcomes.reason_counts[ReasonCode.DEFENDER_HOLD.value] == 1
    assert review.verdict is Verdict.USABLE
    assert review.flow.sparkline


def test_windows_and_late_asr_drop(tmp_path: Path) -> None:
    traces = tmp_path / "gen1"
    episodes = [
        pair(f"win-{index}", ReasonCode.ATTACKER_ROOT, generation=1)
        for index in range(12)
    ] + [
        pair(f"lose-{index}", ReasonCode.DEFENDER_HOLD, generation=1)
        for index in range(12)
    ]
    write_roles(traces, episodes)
    (traces / "metrics.json").write_text(json.dumps({"asr": 0.5}) + "\n")
    review = review_traces(traces, phase=Phase.ROLLOUT, generation=1, window_count=4)
    assert len(review.flow.windows) == 4
    assert review.flow.first_half.asr == pytest.approx(1.0)
    assert review.flow.second_half.asr == pytest.approx(0.0)
    assert review.flow.asr_delta == pytest.approx(-1.0)
    assert "late_asr_drop" in finding_codes(review)


def test_availability_defender_win_is_block(tmp_path: Path) -> None:
    traces = tmp_path / "gen0"
    bad = terminal(
        ReasonCode.AVAILABILITY_FAIL,
        attacker_reward=0.0,
        defender_reward=1.0,
        availability_ok=False,
    )
    write_roles(
        traces,
        [
            (
                trajectory(
                    episode_id="bad",
                    role=Role.ATTACKER,
                    outcome=bad,
                ),
                trajectory(
                    episode_id="bad",
                    role=Role.DEFENDER,
                    outcome=bad,
                ),
            )
        ],
    )
    review = review_traces(traces, phase=Phase.ROLLOUT, generation=0)
    assert review.verdict is Verdict.UNUSABLE
    assert "availability_defender_win" in finding_codes(review)


def test_unconfirmed_root_and_infra_reward_are_blocks(tmp_path: Path) -> None:
    traces = tmp_path / "gen0"
    unconfirmed = terminal(
        ReasonCode.ATTACKER_ROOT,
        host_root=False,
        euid=0,
        attacker_reward=1.0,
        defender_reward=0.0,
    )
    infra = terminal(ReasonCode.INFRA_FAIL, attacker_reward=1.0, defender_reward=0.0)
    write_roles(
        traces,
        [
            (
                trajectory(episode_id="root", role=Role.ATTACKER, outcome=unconfirmed),
                trajectory(episode_id="root", role=Role.DEFENDER, outcome=unconfirmed),
            ),
            (
                trajectory(episode_id="infra", role=Role.ATTACKER, outcome=infra),
                trajectory(episode_id="infra", role=Role.DEFENDER, outcome=infra),
            ),
        ],
    )
    review = review_traces(traces, phase=Phase.ROLLOUT, generation=0)
    codes = finding_codes(review)
    assert "unconfirmed_root_win" in codes
    assert "infra_nonzero_reward" in codes
    assert review.verdict is Verdict.UNUSABLE


def test_parse_error_and_unpaired_role(tmp_path: Path) -> None:
    traces = tmp_path / "gen0"
    write_roles(traces, [pair("ok", ReasonCode.DEFENDER_HOLD)])
    with (traces / "attacker.jsonl").open("a") as handle:
        handle.write("{not json\n")
        handle.write(
            json.dumps(
                trajectory(episode_id="solo", role=Role.ATTACKER).to_dict()
            )
            + "\n"
        )
    review = review_traces(traces, phase=Phase.ROLLOUT, generation=0)
    codes = finding_codes(review)
    assert "parse_errors" in codes
    assert "unpaired_episodes" in codes
    assert review.outcomes.episode_count == 2
    assert review.verdict is Verdict.CAUTION


def test_group_opponent_drift_and_terminal_mismatch(tmp_path: Path) -> None:
    traces = tmp_path / "gen1"
    attacker_a, defender_a = pair("a", ReasonCode.DEFENDER_HOLD, generation=1, group_id="g")
    attacker_b, defender_b = pair(
        "b", ReasonCode.DEFENDER_HOLD, generation=1, group_id="g", opponent="other"
    )
    mismatch_def = trajectory(
        episode_id="a",
        role=Role.DEFENDER,
        generation=1,
        group_id="g",
        reason=ReasonCode.ATTACKER_ROOT,
    )
    write_roles(traces, [(attacker_a, mismatch_def), (attacker_b, defender_b)])
    (traces / "metrics.json").write_text(json.dumps({"asr": 0.0}) + "\n")
    review = review_traces(traces, phase=Phase.ROLLOUT, generation=1)
    codes = finding_codes(review)
    assert "group_opponent_drift" in codes
    assert "paired_terminal_mismatch" in codes
    assert review.verdict is Verdict.UNUSABLE


def test_asr_stuck_after_gen0(tmp_path: Path) -> None:
    traces = tmp_path / "gen2"
    write_roles(traces, [pair("ep", ReasonCode.DEFENDER_HOLD, generation=2)])
    (traces / "metrics.json").write_text(json.dumps({"asr": 0.0}) + "\n")
    review = review_traces(traces, phase=Phase.ROLLOUT, generation=2)
    assert "asr_stuck" in finding_codes(review)
    assert review.verdict is Verdict.UNUSABLE


def test_format_collapse_and_tool_errors(tmp_path: Path) -> None:
    traces = tmp_path / "gen0"
    failed = [
        step(
            0,
            valid=False,
            tools=[
                ToolEvent(
                    name="bash",
                    args={"command": "false"},
                    stdout_head="",
                    stdout_tail="",
                    exit_code=1,
                    duration_ms=3,
                )
            ],
        )
    ]
    episodes = [
        (
            trajectory(episode_id=f"e{i}", role=Role.ATTACKER, steps=failed),
            trajectory(
                episode_id=f"e{i}",
                role=Role.DEFENDER,
                steps=[step(0, role=Role.DEFENDER, valid=False)],
            ),
        )
        for i in range(20)
    ]
    write_roles(traces, episodes)
    (traces / "metrics.json").write_text(json.dumps({"asr": 0.0}) + "\n")
    review = review_traces(traces, phase=Phase.ROLLOUT, generation=0)
    codes = finding_codes(review)
    assert "format_collapse" in codes
    assert "high_tool_error_rate" in codes


def test_complete_phase_records_eval_and_archive_gaps(tmp_path: Path) -> None:
    traces = tmp_path / "gen2"
    write_roles(traces, [pair("ep", ReasonCode.ATTACKER_ROOT, generation=2)])
    (traces / "metrics.json").write_text(json.dumps({"asr": 1.0}) + "\n")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "tier3_light_plan.json").write_text(json.dumps({"held_out_episodes_per_profile": 50}) + "\n")
    review = review_traces(
        traces,
        phase=Phase.COMPLETE,
        generation=2,
        eval_dir=eval_dir,
        archive_dir=tmp_path / "archives",
        pfsp_path=tmp_path / "pfsp.json",
    )
    codes = finding_codes(review)
    assert "eval_results_missing" in codes
    assert "archive_missing" in codes
    assert "pfsp_missing" in codes
    assert "asr_stuck" in codes


def test_eval_incomplete_arms_and_bandpass_profiles(tmp_path: Path) -> None:
    traces = tmp_path / "gen2"
    write_roles(
        traces,
        [
            pair("web-1", ReasonCode.ATTACKER_ROOT, generation=2, profile_id="web"),
            pair("web-2", ReasonCode.DEFENDER_HOLD, generation=2, profile_id="web"),
            pair("ws-1", ReasonCode.DEFENDER_HOLD, generation=2, profile_id="workstation"),
            pair("ws-2", ReasonCode.DEFENDER_HOLD, generation=2, profile_id="workstation"),
        ],
    )
    (traces / "metrics.json").write_text(json.dumps({"asr": 0.25}) + "\n")
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "tier3_light_plan.json").write_text(json.dumps({"mode": "light"}) + "\n")
    (eval_dir / "tier3_light_results.json").write_text(
        json.dumps({"incomplete_arms": ["debian12"], "rows": []}) + "\n"
    )
    archives = tmp_path / "archives" / "gen2"
    archives.mkdir(parents=True)
    (archives / "manifest.json").write_text(
        json.dumps({"roles": {"attacker": {}, "defender": {}}, "checkpoints": [1, 2]}) + "\n"
    )
    pfsp = tmp_path / "pfsp.json"
    pfsp.write_text(
        json.dumps(
            {
                "generation": 2,
                "entries": [
                    {
                        "checkpoint_id": "attacker-gen2",
                        "path": "x",
                        "role": "attacker",
                        "win_rate_vs_live": 0.4,
                    }
                ],
            }
        )
        + "\n"
    )
    review = review_traces(
        traces,
        phase=Phase.COMPLETE,
        generation=2,
        eval_dir=eval_dir,
        archive_dir=tmp_path / "archives",
        pfsp_path=pfsp,
    )
    assert "eval_arm_incomplete" in finding_codes(review)
    assert "workstation" in review.gates.bandpass_profiles
    assert review.artifacts.archive.checkpoint_count == 2
    assert review.artifacts.pfsp.size == 1
    assert review.artifacts.eval.incomplete_arms == ("debian12",)


def test_complete_phase_records_benchmark_score_gaps(tmp_path: Path) -> None:
    traces = tmp_path / "gen2"
    write_roles(traces, [pair("ep", ReasonCode.ATTACKER_ROOT, generation=2)])
    (traces / "metrics.json").write_text(json.dumps({"asr": 0.4}) + "\n")
    eval_dir = tmp_path / "eval"
    bench = eval_dir / "benchmarks"
    bench.mkdir(parents=True)
    (bench / "plan.json").write_text(json.dumps({"job_count": 2}) + "\n")
    (eval_dir / "tier3_light_plan.json").write_text(json.dumps({"mode": "light"}) + "\n")
    (eval_dir / "tier3_light_results.json").write_text(json.dumps({"rows": []}) + "\n")
    review = review_traces(
        traces,
        phase=Phase.COMPLETE,
        generation=2,
        eval_dir=eval_dir,
    )
    codes = finding_codes(review)
    assert "benchmark_scores_missing" in codes
    markdown = write_review(review, traces)[1].read_text()
    assert "External benchmarks scored 0" in markdown


def test_metrics_mismatch_and_markdown_write(tmp_path: Path) -> None:
    traces = tmp_path / "gen0"
    write_roles(
        traces,
        [
            pair("a", ReasonCode.ATTACKER_ROOT),
            pair("b", ReasonCode.DEFENDER_HOLD),
        ],
    )
    (traces / "metrics.json").write_text(json.dumps({"asr": 0.9}) + "\n")
    review = review_traces(traces, phase=Phase.ROLLOUT, generation=0)
    assert "metrics_asr_mismatch" in finding_codes(review)
    json_path, markdown_path = write_review(review, traces)
    payload = json.loads(json_path.read_text())
    markdown = markdown_path.read_text()
    assert payload["verdict"] == "caution"
    assert payload["outcomes"]["asr"] == pytest.approx(0.5)
    assert "## How the work went" in markdown
    assert "**Verdict.**" in markdown
    assert "ASR flow" in markdown
    assert "`metrics_asr_mismatch`" in markdown


def test_cli_writes_review_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    traces = tmp_path / "gen0"
    write_roles(traces, [pair("a", ReasonCode.DEFENDER_HOLD), pair("b", ReasonCode.ATTACKER_ROOT)])
    (traces / "metrics.json").write_text(json.dumps({"asr": 0.5}) + "\n")
    from ultron.train import review as review_mod
    import sys

    argv = ["review", str(traces), "--phase", "rollout", "--generation", "0"]
    old = sys.argv
    sys.argv = argv
    try:
        review_mod._main()
    finally:
        sys.argv = old
    printed = json.loads(capsys.readouterr().out)
    assert printed["verdict"] == "usable"
    assert printed["episodes"] == 2
    assert (traces / "review.md").is_file()
    assert (traces / "review.json").is_file()
