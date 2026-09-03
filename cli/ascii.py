from __future__ import annotations

from ultron.train.schema_v1 import Role

from .board import (
    Board,
    Done,
    Failed,
    Probing,
    Restoring,
    Settled,
    Trading,
)


def _fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text + " " * (width - len(text))
    if width == 1:
        return text[:1]
    return text[: width - 1] + "…"


def _bar(fraction: float, width: int) -> str:
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "█" * filled + "░" * (width - filled)


def _clock(seconds: float) -> str:
    total = int(seconds)
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"


def _side_label(board: Board) -> str:
    phase = board.phase
    if isinstance(phase, Restoring):
        return "RST"
    if isinstance(phase, Probing):
        return "PRB"
    if isinstance(phase, (Settled, Done)):
        return "END"
    if isinstance(phase, Failed):
        return "FAIL"
    if board.acting_role is Role.ATTACKER:
        return "ATK"
    if board.acting_role is Role.DEFENDER:
        return "DEF"
    return "—"


def _eta(board: Board) -> str:
    eta = board.progress.eta_s
    if eta is None:
        return "—"
    return _clock(eta)


def _guest_euid(board: Board) -> str:
    if isinstance(board.phase, Settled):
        return str(board.phase.terminal.attacker_euid)
    if isinstance(board.phase, Done):
        return str(board.phase.last.terminal.attacker_euid)
    return "——"


def _guest_avail(board: Board) -> str:
    if isinstance(board.phase, Settled):
        return "ok" if board.phase.terminal.availability_ok else "fail"
    if isinstance(board.phase, Done):
        return "ok" if board.phase.last.terminal.availability_ok else "fail"
    return "——"


def _guest_title(board: Board) -> str:
    if isinstance(board.phase, Restoring):
        return "GUEST  RESTORING"
    if isinstance(board.phase, Probing):
        return "GUEST  UNDER PROBE"
    if isinstance(board.phase, Settled) and board.phase.terminal.reason_code.value == "ATTACKER_ROOT":
        return "GUEST  COMPROMISED"
    if isinstance(board.phase, Done) and board.phase.last.terminal.reason_code.value == "ATTACKER_ROOT":
        return "GUEST  COMPROMISED"
    return "GUEST"


def _quarantine_line(board: Board) -> str:
    if board.quarantine is None:
        return "quarantine  not instrumented"
    return f"quarantine  {len(board.quarantine)}"


def _agent_block(board: Board, role: Role, width: int) -> list[str]:
    face = board.cast.attacker if role is Role.ATTACKER else board.cast.defender
    title = "ATTACKER" if role is Role.ATTACKER else "DEFENDER"
    stance = face.stance.upper()
    last = face.last.verb if face.last else "—"
    detail = face.last.detail if face.last else ""
    turn = f"turn {face.turns_done}/{board.spec.turns_per_side}"
    lines = [
        f"{title}  {stance}",
        turn,
        f"▸ {last}",
        _fit(detail, width),
    ]
    return [_fit(line, width) for line in lines]


def render(board: Board, *, cols: int = 100, rows: int = 28) -> str:
    cols = max(72, cols)
    inner = cols - 2
    guest = board.cast.guest
    guest_id = guest.guest_id if guest else "no guest"
    isolation = guest.isolation.value if guest else "—"
    host = guest.host_address if guest else ""
    turn_part = ""
    if board.progress.cursor is not None:
        turn_part = f"  t {board.progress.cursor.index}/{board.progress.cursor.budget}"
    header = _fit(
        f" ULTRON  gen {board.spec.generation}  {board.spec.profile_id}  {isolation}"
        f"{'':4}elapsed {_clock(board.progress.elapsed_s)}",
        cols,
    )
    bar_w = max(10, cols - 48)
    progress = (
        f" {_bar(board.progress.fraction, bar_w)}  "
        f"ep {board.progress.episodes_done}/{board.progress.episode_count}"
        f"{turn_part}  {_side_label(board)}  ETA {_eta(board)}"
    )
    progress = _fit(progress, cols)
    flank = max(22, (inner - 24) // 2)
    guest_w = inner - 2 * flank - 2
    atk = _agent_block(board, Role.ATTACKER, flank - 2)
    dfn = _agent_block(board, Role.DEFENDER, flank - 2)
    euid = _guest_euid(board)
    avail = _guest_avail(board)
    guest_lines = [
        _fit(_guest_title(board), guest_w - 2),
        _fit(f"{guest_id}  {isolation}", guest_w - 2),
        _fit(host, guest_w - 2),
        _fit(f"euid {euid}  avail {avail}", guest_w - 2),
        _fit(_quarantine_line(board), guest_w - 2),
    ]
    while len(atk) < 5:
        atk.append(" " * (flank - 2))
    while len(dfn) < 5:
        dfn.append(" " * (flank - 2))
    top = (
        "┌─ "
        + _fit("ATTACKER", flank - 3)
        + "┬─ "
        + _fit("GUEST", guest_w - 3)
        + "┬─ "
        + _fit("DEFENDER", flank - 3)
        + "┐"
    )
    if len(top) != cols:
        top = _fit(top, cols)
    body = []
    for i in range(5):
        body.append(
            "│ "
            + atk[i]
            + " │ "
            + guest_lines[i]
            + " │ "
            + dfn[i]
            + " │"
        )
        body[-1] = _fit(body[-1], cols)
    bot = "└" + "─" * (flank) + "┴" + "─" * (guest_w) + "┴" + "─" * flank + "┘"
    bot = _fit(bot, cols)
    proc_top = "┌─ PROCESS " + "─" * (cols - 13) + "┐"
    proc_top = _fit(proc_top, cols)
    strokes = list(board.process.strokes[-6:])
    proc = [_fit(f"│ {_fit(board.process.headline, inner - 2)} │", cols)]
    for stroke in strokes:
        line = (
            f"{stroke.actor[0].upper()}  {stroke.verb}  {stroke.detail}  "
            f"{'' if stroke.exit_code is None else 'e' + str(stroke.exit_code)}"
        )
        proc.append(_fit(f"│ {_fit(line, inner - 2)} │", cols))
    while len(proc) < 7:
        proc.append(_fit("│" + " " * inner + "│", cols))
    proc_bot = "└" + "─" * inner + "┘"
    proc_bot = _fit(proc_bot, cols)
    lines = [header, progress, "", top, *body, bot, proc_top, *proc, proc_bot]
    if len(lines) < rows:
        lines.extend([""] * (rows - len(lines)))
    return "\n".join(lines[:rows])
