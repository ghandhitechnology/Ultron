from __future__ import annotations

from ultron.cli.model import JobProgress, JobSnapshot, Phase, estimate_eta_s, progress
from ultron.cli.pixel import PixelStyle
from ultron.train.schema_v1 import ReasonCode, Role

BLOCK = "█"
EMPTY = "░"


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "ETA --"
    total = int(round(seconds))
    return f"ETA {total // 60:02d}:{total % 60:02d}"


def format_bar(ratio: float, width: int) -> str:
    ratio = min(1.0, max(0.0, ratio))
    filled = int(round(ratio * width))
    return BLOCK * filled + EMPTY * (width - filled)


def header_line(snapshot: JobSnapshot) -> str:
    prog = progress(snapshot)
    done = prog.completed_episodes
    total = prog.total_episodes
    if snapshot.phase is Phase.COMPLETE:
        ratio = 1.0
        ep_label = f"episode {total}/{total}"
    else:
        ratio = (done + _turn_ratio(snapshot, prog)) / total
        ep_label = f"episode {min(done + 1, total)}/{total}"
    turn = "—"
    if snapshot.turn_index is not None:
        turn = f"turn {snapshot.turn_index + 1}/{prog.total_turns}"
    elif snapshot.phase is Phase.PROBING:
        turn = "probe"
    elif snapshot.phase is Phase.RESTORING:
        turn = "restore"
    elif snapshot.phase is Phase.COMPLETE:
        turn = "done"
    pct = f"{int(ratio * 100):3d}%"
    return (
        f"  LIVE GUEST GYM   gen {snapshot.meta.generation}   {ep_label}   "
        f"profile {snapshot.meta.profile_id}     {pct} / {format_eta(estimate_eta_s(snapshot))}"
        f"     {turn}"
    )


def footer_line(
    snapshot: JobSnapshot,
    *,
    sim: bool = True,
    pixel_style: PixelStyle | None = None,
) -> str:
    mode = "SIM MODE" if sim else "LIVE"
    pixel = "" if pixel_style is None else f"   pixel {pixel_style.value}"
    return (
        f"  ultron v{snapshot.meta.version}   generation {snapshot.meta.generation}   "
        f"{snapshot.meta.profile_id}   {snapshot.meta.isolation.value}{pixel}"
        f"                {mode}"
    )


def progress_block(snapshot: JobSnapshot) -> str:
    prog = progress(snapshot)
    ep_ratio = prog.completed_episodes / prog.total_episodes
    if snapshot.phase is Phase.COMPLETE:
        ep_ratio = 1.0
    turn_ratio = _turn_ratio(snapshot, prog)
    mean = ""
    if snapshot.completed:
        avg = sum(item.duration_s for item in snapshot.completed) / len(snapshot.completed)
        mean = f"   mean {avg:.1f}s"
    return (
        f"  EPISODES  {prog.completed_episodes:>2} / {prog.total_episodes:<2}   "
        f"{format_bar(ep_ratio, 28)}{mean}\n"
        f"  TURNS     {_turn_display(snapshot, prog):>2} / {prog.total_turns:<2}   "
        f"{format_bar(turn_ratio, 28)}   {snapshot.phase.value}\n"
        f"  {_outcome_strip(snapshot)}   click a/s/d/t · p pixel · esc fold · q quit"
    )


def attacker_pane(snapshot: JobSnapshot) -> str:
    return "\n".join(
        [
            "ATTACKER",
            "attacker_lora",
            _side_state(snapshot, Role.ATTACKER),
            f"turn {_turn_label(snapshot, Role.ATTACKER)}",
            f"tools {snapshot.attacker_tools}",
            f"last {snapshot.last_attacker}",
            "════ exploit ════▶",
        ]
    )


def defender_pane(snapshot: JobSnapshot) -> str:
    return "\n".join(
        [
            "DEFENDER",
            "defender_lora",
            _side_state(snapshot, Role.DEFENDER),
            f"turn {_turn_label(snapshot, Role.DEFENDER)}",
            f"tools {snapshot.defender_tools}",
            f"last {snapshot.last_defender}",
            "◀════ policy ════",
        ]
    )


def sandbox_pane(snapshot: JobSnapshot) -> str:
    guest = snapshot.guest_id or "—"
    host = snapshot.host_address or "—"
    image = snapshot.image_ref or "—"
    euid = "euid ?"
    avail = "avail ?"
    root = "root ?"
    if snapshot.probe is not None:
        euid = f"euid {snapshot.probe.guest_attacker_euid}"
        avail = "avail ✓" if snapshot.probe.availability_ok else "avail ✗"
        root = "root ✓" if snapshot.probe.host_confirmed_root else "root ✗"
    isolation = snapshot.meta.isolation.value
    return "\n".join(
        [
            "SANDBOX",
            "┌────────────────────┐",
            f"│ {isolation:<18} │",
            f"│ {guest:<18} │",
            f"│ {host:<18} │",
            f"│ {image[:18]:<18} │",
            f"│ {euid:<8} {avail:<9} │",
            f"│ {root:<18} │",
            "└────────────────────┘",
        ]
    )


def arena_block(snapshot: JobSnapshot) -> str:
    left = attacker_pane(snapshot).splitlines()
    mid = sandbox_pane(snapshot).splitlines()
    right = defender_pane(snapshot).splitlines()
    height = max(len(left), len(mid), len(right))
    rows = []
    for i in range(height):
        a = left[i] if i < len(left) else ""
        s = mid[i] if i < len(mid) else ""
        d = right[i] if i < len(right) else ""
        rows.append(f"{a:<28}{s:<28}{d}")
    rows.append(_outcome_strip(snapshot))
    return "\n".join(rows)


def detail_block(snapshot: JobSnapshot, focus: str) -> str:
    if focus == "sandbox":
        probe = snapshot.probe
        lines = [
            "SANDBOX",
            f"  guest        {snapshot.guest_id or '—'}",
            f"  host         {snapshot.host_address or '—'}",
            f"  image        {snapshot.image_ref or '—'}",
            f"  isolation    {snapshot.meta.isolation.value}",
            f"  snapshot     {snapshot.meta.snapshot_sha256 or '—'}",
            f"  phase        {snapshot.phase.value}",
        ]
        if probe is not None:
            lines.extend(
                [
                    f"  euid         {probe.guest_attacker_euid}",
                    f"  host_root    {probe.host_confirmed_root}",
                    f"  availability {probe.availability_ok}",
                    f"  infra_ok     {probe.infra_ok}",
                    f"  timed_out    {probe.timed_out}",
                ]
            )
        else:
            lines.append("  probe        pending until episode end")
        return "\n".join(lines)
    if focus == "attacker":
        return _side_detail(snapshot, Role.ATTACKER)
    if focus == "defender":
        return _side_detail(snapshot, Role.DEFENDER)
    if focus == "tool":
        tool = snapshot.last_tool
        if tool is None:
            return "TOOL\n  no tool event yet"
        cmd = tool.args.get("cmd", "")
        return "\n".join(
            [
                f"TOOL  {tool.name}",
                f"  role     {snapshot.last_tool_role.value if snapshot.last_tool_role else '—'}",
                f"  cmd      {cmd}",
                f"  exit     {tool.exit_code}",
                f"  duration {tool.duration_ms}ms",
                "  stdout_head",
                *(f"    {line}" for line in (tool.stdout_head or "").splitlines()[:12]),
                "  stdout_tail",
                *(f"    {line}" for line in (tool.stdout_tail or "").splitlines()[:8]),
            ]
        )
    return "PROCESS\n  click a pane or a tool to expand\n  esc folds"


def _side_detail(snapshot: JobSnapshot, role: Role) -> str:
    active = snapshot.active_role is role
    last = snapshot.last_attacker if role is Role.ATTACKER else snapshot.last_defender
    tools = snapshot.attacker_tools if role is Role.ATTACKER else snapshot.defender_tools
    return "\n".join(
        [
            role.value.upper(),
            f"  adapter   {role.value}_lora",
            f"  state     {'ACTING' if active else 'waiting'}",
            f"  tools     {tools}",
            f"  last      {last}",
            f"  turn      {snapshot.turn_index if snapshot.turn_index is not None else '—'}",
        ]
    )


def _side_state(snapshot: JobSnapshot, role: Role) -> str:
    if snapshot.active_role is role:
        return "● ACTING"
    if snapshot.phase is Phase.COMPLETE and snapshot.last_terminal is not None:
        if role is Role.ATTACKER and snapshot.last_terminal.reason_code is ReasonCode.ATTACKER_ROOT:
            return "● ROOT"
        if role is Role.DEFENDER and snapshot.last_terminal.reason_code is ReasonCode.DEFENDER_HOLD:
            return "● HOLD"
    return "○ wait"


def _outcome_strip(snapshot: JobSnapshot) -> str:
    counts: dict[str, int] = {}
    for item in snapshot.completed:
        key = item.terminal.reason_code.value
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "outcomes  (none yet)"
    parts = [f"{name} {counts[name]}" for name in counts]
    return "outcomes  " + "  ".join(parts)


def _turn_ratio(snapshot: JobSnapshot, prog: JobProgress) -> float:
    if snapshot.phase is Phase.COMPLETE:
        return 1.0
    if snapshot.turn_index is None:
        if snapshot.phase is Phase.PROBING:
            return 1.0
        return 0.0
    return min(1.0, (snapshot.turn_index + 1) / prog.total_turns)


def _turn_display(snapshot: JobSnapshot, prog: JobProgress) -> int:
    if snapshot.phase is Phase.COMPLETE:
        return prog.total_turns
    if snapshot.turn_index is None:
        return 0 if snapshot.phase is Phase.RESTORING else prog.total_turns
    return snapshot.turn_index + 1


def _turn_label(snapshot: JobSnapshot, role: Role) -> str:
    if snapshot.active_role is role and snapshot.turn_index is not None:
        return str(snapshot.turn_index)
    return "—"
