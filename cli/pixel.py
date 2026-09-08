from __future__ import annotations

import re
from enum import Enum

SPRITE_WIDTH = 16
SPRITE_HEIGHT = 7
FRAME_RATE = 30
FRAME_HOLD = 3
FRAME_COUNT = 8 * FRAME_HOLD

_MARKUP_RE = re.compile(r"\[(?:/|[a-zA-Z#][^\]]*)\]")
_BLANK = " " * SPRITE_WIDTH


class SpriteId(str, Enum):
    VISION = "vision"
    EXPLOITER = "exploiter"


def visible_text(row: str) -> str:
    return _MARKUP_RE.sub("", row)


def frame_index(tick: int) -> int:
    return tick % FRAME_COUNT


def frames_for(sprite: SpriteId) -> tuple[tuple[str, ...], ...]:
    match sprite:
        case SpriteId.VISION:
            return VISION_FRAMES
        case SpriteId.EXPLOITER:
            return EXPLOITER_FRAMES
        case _:
            _assert_never(sprite)


def sprite_rows(sprite: SpriteId, tick: int) -> tuple[str, ...]:
    return frames_for(sprite)[frame_index(tick)]


def sprite_block(sprite: SpriteId, tick: int) -> str:
    return "\n".join(sprite_rows(sprite, tick))


def mascot_strip(tick: int) -> str:
    left = sprite_rows(SpriteId.EXPLOITER, tick)
    right = sprite_rows(SpriteId.VISION, tick)
    core = _core_rows(tick)
    left_link, right_link = _link_rows(tick)
    gap = "  "
    header = f"{'exploiter':<16}{gap}{'uplink':^6}{gap}{'core':^16}{gap}{'link':^6}{gap}{'vision':>16}"
    body = [
        f"{a}{gap}{left_link[index]}{gap}{core[index]}{gap}{right_link[index]}{gap}{b}"
        for index, (a, b) in enumerate(zip(left, right))
    ]
    return "\n".join((header, *body))


def signal_meter(level: int, *, width: int = 4) -> str:
    """Return a fixed-width terminal signal meter."""
    if width < 1:
        raise ValueError("signal meter width must be positive")
    level = max(0, min(width, level))
    return "▮" * level + "▯" * (width - level)


def telemetry_badge(
    label: str,
    *,
    level: int = 4,
    pending: int = 0,
    tick: int | None = None,
) -> str:
    color = "#f9e2af" if level < 3 else "#a6e3a1"
    queue = f"  Q {pending}" if pending else ""
    marker = "●" if tick is None else ("·", "•", "●", "•")[(tick // 4) % 4]
    return (
        f"[bold {color}]{marker} {label.upper()}[/]  "
        f"[{color}]{signal_meter(level)}[/]  "
        f"[dim #7f849c]{FRAME_RATE} FPS{queue}[/]"
    )


def _core_rows(tick: int) -> tuple[str, ...]:
    pulse = ("·", "•", "●", "•")[tick % 4]
    signal = ("  ·  ", "  )  ", " ))  ", ")))  ")[tick % 4]
    rows = (
        "╭──╮",
        f"{signal}│◆◆│{signal[::-1]}",
        "┌───┴──┴───┐",
        "│ ▣  ULTRON │",
        "├──────────┤",
        f"│ {pulse}  READY  │",
        "└──────────┘",
    )
    colors = ("#89b4fa", "#89b4fa", "#6c7086", "#cba6f7", "#45475a", "#a6e3a1", "#6c7086")
    return tuple(f"[{color}]{row:^{SPRITE_WIDTH}}[/]" for row, color in zip(rows, colors))


def _link_rows(tick: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    position = tick % 6
    reverse = 5 - position
    left: list[str] = []
    right: list[str] = []
    for row in range(SPRITE_HEIGHT):
        if row == 3:
            left_line = "".join("●" if index == position else "─" for index in range(6))
            right_line = "".join("●" if index == reverse else "─" for index in range(6))
            left.append(f"[#f38ba8]{left_line}[/]")
            right.append(f"[#a6e3a1]{right_line}[/]")
        else:
            left.append(" " * 6)
            right.append(" " * 6)
    return tuple(left), tuple(right)


def _assert_never(value: object) -> None:
    raise ValueError(f"unhandled {value!r}")


def _place(content: tuple[str, ...], pad_top: int) -> tuple[str, ...]:
    pad_bottom = SPRITE_HEIGHT - pad_top - len(content)
    if pad_bottom < 0:
        raise ValueError("sprite content taller than the frame")
    return (_BLANK,) * pad_top + content + (_BLANK,) * pad_bottom


def _vision(gem: str, *, blink: bool = False) -> tuple[str, ...]:
    brows = f" [#94e2d5]▄▀▀▀▄[/]  [#f9e2af]{gem}[/] [#94e2d5]▄▀▀▀▄[/] "
    if blink:
        eyes = (
            "                ",
            "  [#94e2d5]▀▀▀[/]      [#94e2d5]▀▀▀[/]  ",
            "                ",
        )
    else:
        eyes = (
            "  [#cdd6f4]▄▄▄[/]      [#cdd6f4]▄▄▄[/]  ",
            "  [#cdd6f4]█[/][#181825]█[/][#cdd6f4]█[/]      [#cdd6f4]█[/][#181825]█[/][#cdd6f4]█[/]  ",
            "  [#cdd6f4]▀▀▀[/]      [#cdd6f4]▀▀▀[/]  ",
        )
    smile = " [#a6e3a1]●[/] [#a6e3a1]▀▄▄▄▄▄▄▄▄▀[/] [#a6e3a1]●[/] "
    return (brows, *eyes, smile)


def _exploiter(*, nails: str, lobe: str, tip: str, oval: str) -> tuple[str, ...]:
    return (
        f"    {nails}    {nails}    ",
        f"    [#f38ba8]█[/][#eba0ac]▓[/]    [#f38ba8]█[/][#eba0ac]▓[/] {oval} ",
        f"{lobe} [#f38ba8]█[/][#eba0ac]▓[/]    [#f38ba8]█[/][#eba0ac]▓[/] [#fab387]█[/] [#fab387]█[/] ",
        f"[#eba0ac]▀▀▀[/] [#f38ba8]█▄[/]    [#f38ba8]▄█[/]    ",
        f"    [#f38ba8]▀█[/][#cdd6f4]{tip}[/][#f38ba8]█▀[/]       ",
    )


_VISION_KEYFRAMES: tuple[tuple[str, ...], ...] = (
    _place(_vision("◆"), pad_top=1),
    _place(_vision("◇"), pad_top=1),
    _place(_vision("◆"), pad_top=0),
    _place(_vision("◆", blink=True), pad_top=0),
    _place(_vision("◇"), pad_top=0),
    _place(_vision("◆"), pad_top=1),
    _place(_vision("◇"), pad_top=2),
    _place(_vision("◆"), pad_top=2),
)

_NAIL_UP = "[#cdd6f4]▄▄[/]"
_NAIL_DOWN = "[#cdd6f4]▄▀[/]"
_LOBE_SOFT = "[#eba0ac]▄▄[/]"
_LOBE_FULL = "[#eba0ac]██[/]"
_OVAL_OPEN = "[#fab387]▄▄[/]"
_OVAL_FULL = "[#fab387]●█[/]"

_EXPLOITER_KEYFRAMES: tuple[tuple[str, ...], ...] = (
    _place(_exploiter(nails=_NAIL_UP, lobe=_LOBE_SOFT, tip="▄", oval=_OVAL_OPEN), pad_top=1),
    _place(_exploiter(nails=_NAIL_UP, lobe=_LOBE_FULL, tip="▼", oval=_OVAL_OPEN), pad_top=0),
    _place(_exploiter(nails=_NAIL_UP, lobe=_LOBE_FULL, tip="▼", oval=_OVAL_FULL), pad_top=0),
    _place(_exploiter(nails=_NAIL_DOWN, lobe=_LOBE_SOFT, tip="▼", oval=_OVAL_FULL), pad_top=1),
    _place(_exploiter(nails=_NAIL_DOWN, lobe=_LOBE_SOFT, tip="▄", oval=_OVAL_OPEN), pad_top=1),
    _place(_exploiter(nails=_NAIL_UP, lobe=_LOBE_SOFT, tip="▄", oval=_OVAL_OPEN), pad_top=1),
    _place(_exploiter(nails=_NAIL_UP, lobe=_LOBE_FULL, tip="▼", oval=_OVAL_OPEN), pad_top=2),
    _place(_exploiter(nails=_NAIL_UP, lobe=_LOBE_SOFT, tip="▄", oval=_OVAL_OPEN), pad_top=2),
)


def _hold_frames(frames: tuple[tuple[str, ...], ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(frame for frame in frames for _ in range(FRAME_HOLD))


VISION_FRAMES = _hold_frames(_VISION_KEYFRAMES)
EXPLOITER_FRAMES = _hold_frames(_EXPLOITER_KEYFRAMES)
