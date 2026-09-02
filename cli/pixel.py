from __future__ import annotations

import re
from enum import Enum

SPRITE_WIDTH = 16
SPRITE_HEIGHT = 7
FRAME_COUNT = 8

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
    gap = "      "
    header = f"{'exploiter':<16}{gap}{'':16}{gap}{'vision':<16}"
    body = [f"{a}{gap}{' ' * 16}{gap}{b}" for a, b in zip(left, right)]
    return "\n".join((header, *body))


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


VISION_FRAMES: tuple[tuple[str, ...], ...] = (
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

EXPLOITER_FRAMES: tuple[tuple[str, ...], ...] = (
    _place(_exploiter(nails=_NAIL_UP, lobe=_LOBE_SOFT, tip="▄", oval=_OVAL_OPEN), pad_top=1),
    _place(_exploiter(nails=_NAIL_UP, lobe=_LOBE_FULL, tip="▼", oval=_OVAL_OPEN), pad_top=0),
    _place(_exploiter(nails=_NAIL_UP, lobe=_LOBE_FULL, tip="▼", oval=_OVAL_FULL), pad_top=0),
    _place(_exploiter(nails=_NAIL_DOWN, lobe=_LOBE_SOFT, tip="▼", oval=_OVAL_FULL), pad_top=1),
    _place(_exploiter(nails=_NAIL_DOWN, lobe=_LOBE_SOFT, tip="▄", oval=_OVAL_OPEN), pad_top=1),
    _place(_exploiter(nails=_NAIL_UP, lobe=_LOBE_SOFT, tip="▄", oval=_OVAL_OPEN), pad_top=1),
    _place(_exploiter(nails=_NAIL_UP, lobe=_LOBE_FULL, tip="▼", oval=_OVAL_OPEN), pad_top=2),
    _place(_exploiter(nails=_NAIL_UP, lobe=_LOBE_SOFT, tip="▄", oval=_OVAL_OPEN), pad_top=2),
)
