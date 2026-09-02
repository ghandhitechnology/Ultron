from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

SPRITE_WIDTH = 16
SPRITE_HEIGHT = 7
FRAME_COUNT = 6
STYLE_HOLD_LOOPS = 2

_MARKUP_RE = re.compile(r"\[(?:/|[a-zA-Z#][^\]]*)\]")


class PixelStyle(str, Enum):
    SKETCH = "sketch"
    OPUS = "opus"
    GROK = "grok"


class SpriteId(str, Enum):
    VISION = "vision"
    EXPLOITER = "exploiter"


@dataclass(frozen=True)
class SpritePack:
    vision: tuple[tuple[str, ...], ...]
    exploiter: tuple[tuple[str, ...], ...]


def visible_text(row: str) -> str:
    return _MARKUP_RE.sub("", row)


def frame_index(tick: int) -> int:
    return tick % FRAME_COUNT


def style_at(tick: int, *, pinned: PixelStyle | None = None) -> PixelStyle:
    if pinned is not None:
        return pinned
    styles = tuple(PixelStyle)
    hold = FRAME_COUNT * STYLE_HOLD_LOOPS
    return styles[(tick // hold) % len(styles)]


def advance_style_tick(tick: int) -> int:
    hold = FRAME_COUNT * STYLE_HOLD_LOOPS
    return ((tick // hold) + 1) * hold


def pack_for(style: PixelStyle) -> SpritePack:
    match style:
        case PixelStyle.SKETCH:
            return SKETCH_PACK
        case PixelStyle.OPUS:
            return OPUS_PACK
        case PixelStyle.GROK:
            return GROK_PACK
        case _:
            _assert_never(style)


def frames_for(style: PixelStyle, sprite: SpriteId) -> tuple[tuple[str, ...], ...]:
    pack = pack_for(style)
    match sprite:
        case SpriteId.VISION:
            return pack.vision
        case SpriteId.EXPLOITER:
            return pack.exploiter
        case _:
            _assert_never(sprite)


def sprite_rows(style: PixelStyle, sprite: SpriteId, tick: int) -> tuple[str, ...]:
    frames = frames_for(style, sprite)
    return frames[frame_index(tick)]


def sprite_block(style: PixelStyle, sprite: SpriteId, tick: int) -> str:
    return "\n".join(sprite_rows(style, sprite, tick))


def mascot_strip(tick: int, *, style: PixelStyle | None = None) -> str:
    resolved = style_at(tick, pinned=style)
    left = sprite_rows(resolved, SpriteId.EXPLOITER, tick)
    right = sprite_rows(resolved, SpriteId.VISION, tick)
    gap = "      "
    label = f"pixel  {resolved.value}"
    header = f"{'exploiter':<16}{gap}{label:^16}{gap}{'vision':<16}"
    body = [f"{a}{gap}{' ' * 16}{gap}{b}" for a, b in zip(left, right)]
    return "\n".join((header, *body))


def _assert_never(value: object) -> None:
    raise ValueError(f"unhandled {value!r}")


SKETCH_VISION: tuple[tuple[str, ...], ...] = (
    (
        "[#a6adc8]▄▀▀▀▄[/]  [#f9e2af]◇[/]  [#a6adc8]▄▀▀▀▄[/] ",
        "                ",
        " [#cdd6f4]▄▄[/]          [#cdd6f4]▄▄[/] ",
        " [#cdd6f4]█[/][#181825]█[/]          [#cdd6f4]█[/][#181825]█[/] ",
        " [#cdd6f4]▀▀[/]          [#cdd6f4]▀▀[/] ",
        "                ",
        "  [#cdd6f4]▀▄▄▄▄▄▄▄▀[/]     ",
    ),
    (
        "                ",
        "[#a6adc8]▄▀▀▀▄[/]  [#f9e2af]◆[/]  [#a6adc8]▄▀▀▀▄[/] ",
        " [#cdd6f4]▄▄[/]          [#cdd6f4]▄▄[/] ",
        " [#cdd6f4]█[/][#181825]█[/]          [#cdd6f4]█[/][#181825]█[/] ",
        " [#cdd6f4]▀▀[/]          [#cdd6f4]▀▀[/] ",
        "  [#cdd6f4]▀▄▄▄▄▄▄▄▀[/]     ",
        "                ",
    ),
    (
        "                ",
        "[#a6adc8]▄▀▀▀▄[/]  [#f9e2af]◆[/]  [#a6adc8]▄▀▀▀▄[/] ",
        "                ",
        " [#a6adc8]▀▀[/]          [#a6adc8]▀▀[/] ",
        "                ",
        "  [#cdd6f4]▀▄▄▄▄▄▄▄▄▀[/]    ",
        "                ",
    ),
    (
        "                ",
        "[#a6adc8]▄▀▀▀▄[/]  [#f9e2af]◇[/]  [#a6adc8]▄▀▀▀▄[/] ",
        " [#cdd6f4]▄▄[/]          [#cdd6f4]▄▄[/] ",
        " [#cdd6f4]█[/][#181825]█[/]          [#cdd6f4]█[/][#181825]█[/] ",
        " [#cdd6f4]▀▀[/]          [#cdd6f4]▀▀[/] ",
        "  [#cdd6f4]▀▄▄▄▄▄▄▄▀[/]     ",
        "                ",
    ),
    (
        "[#a6adc8]▄▀▀▀▄[/]  [#f9e2af]◇[/]  [#a6adc8]▄▀▀▀▄[/] ",
        " [#cdd6f4]▄▄[/]          [#cdd6f4]▄▄[/] ",
        " [#cdd6f4]█[/][#181825]█[/]          [#cdd6f4]█[/][#181825]█[/] ",
        " [#cdd6f4]▀▀[/]          [#cdd6f4]▀▀[/] ",
        "                ",
        "  [#cdd6f4]▀▄▄▄▄▄▄▄▀[/]     ",
        "                ",
    ),
    (
        "                ",
        "[#a6adc8]▄▀▀▀▄[/]  [#f9e2af]◆[/]  [#a6adc8]▄▀▀▀▄[/] ",
        " [#cdd6f4]▄▄[/]          [#cdd6f4]▄▄[/] ",
        " [#cdd6f4]█[/][#181825]█[/]          [#cdd6f4]█[/][#181825]█[/] ",
        " [#cdd6f4]▀▀[/]          [#cdd6f4]▀▀[/] ",
        "                ",
        "  [#cdd6f4]▀▄▄▄▄▄▄▄▀[/]     ",
    ),
)

SKETCH_EXPLOITER: tuple[tuple[str, ...], ...] = (
    (
        "   [#cdd6f4]▒█[/]    [#cdd6f4]▒█[/]     ",
        "   [#cdd6f4]▒█[/]    [#cdd6f4]▒█[/]  [#cdd6f4]▄▄[/] ",
        " [#fab387]●[/] [#cdd6f4]▒█[/]  [#f9e2af]▲[/]  [#cdd6f4]▒█[/] [#cdd6f4]▓█[/] ",
        "[#cdd6f4]████[/]  [#fab387]▼▶◀○[/] [#f9e2af]▼[/][#cdd6f4]██[/]  ",
        "[#cdd6f4]███[/]       [#f9e2af]▼[/][#cdd6f4]██[/]   ",
        " [#cdd6f4]▀▀▀[/]       [#cdd6f4]▀▀[/]   ",
        "                ",
    ),
    (
        "  [#cdd6f4]▒█[/]    [#cdd6f4]▒█[/]      ",
        "  [#cdd6f4]▒█[/]    [#cdd6f4]▒█[/]   [#cdd6f4]▄▄[/] ",
        " [#fab387]●[/] [#cdd6f4]▒█[/] [#f9e2af]▲▲[/]  [#cdd6f4]▒█[/] [#cdd6f4]▓█[/] ",
        "[#cdd6f4]████[/]  [#fab387]▼▶◀○[/] [#f9e2af]▼[/][#cdd6f4]██[/]  ",
        "[#cdd6f4]███[/]       [#f9e2af]▼[/][#cdd6f4]██[/]   ",
        " [#cdd6f4]▀▀▀[/]       [#cdd6f4]▀▀[/]   ",
        "                ",
    ),
    (
        "                ",
        "   [#cdd6f4]▄█[/]    [#cdd6f4]█▄[/]  [#cdd6f4]▄▄[/] ",
        " [#fab387]●[/]  [#cdd6f4]▒█[/] [#f9e2af]▲▲[/] [#cdd6f4]▒█[/] [#cdd6f4]●█[/] ",
        "[#cdd6f4]████[/]  [#fab387]▼▶◀○[/] [#f9e2af]▼[/][#cdd6f4]██[/]  ",
        "[#cdd6f4]███[/]      [#f9e2af]▼▼[/][#cdd6f4]██[/]   ",
        " [#cdd6f4]▀▀▀[/]       [#cdd6f4]▀▀[/]   ",
        "                ",
    ),
    (
        "                ",
        "    [#cdd6f4]▄█[/]  [#cdd6f4]█▄[/]   [#cdd6f4]▄▄[/] ",
        " [#fab387]●[/]   [#cdd6f4]▒[/]  [#f9e2af]▲[/]  [#cdd6f4]▒[/] [#cdd6f4]▓█[/] ",
        "[#cdd6f4]████[/]  [#fab387]▼▶◀○[/] [#f9e2af]▼[/][#cdd6f4]██[/]  ",
        "[#cdd6f4]███[/]       [#f9e2af]▼[/][#cdd6f4]██[/]   ",
        " [#cdd6f4]▀▀▀[/]       [#cdd6f4]▀▀[/]   ",
        "  [#cdd6f4]░░[/]         [#cdd6f4]░░[/] ",
    ),
    (
        "   [#cdd6f4]▒█[/]    [#cdd6f4]▒█[/]     ",
        "   [#cdd6f4]▒█[/]    [#cdd6f4]▒█[/]  [#cdd6f4]▄▄[/] ",
        " [#fab387]●[/] [#cdd6f4]▒█[/]  [#f9e2af]▲[/]  [#cdd6f4]▒█[/] [#cdd6f4]▄█[/] ",
        "[#cdd6f4]████[/]  [#fab387]▼▶◀○[/] [#f9e2af]▼[/][#cdd6f4]██[/]  ",
        "[#cdd6f4]███[/]      [#f9e2af]▼▼[/][#cdd6f4]██[/]   ",
        " [#cdd6f4]▀▀▀[/]       [#cdd6f4]▀▀[/]   ",
        "                ",
    ),
    (
        "  [#cdd6f4]▄█[/]      [#cdd6f4]▄█[/] [#cdd6f4]▄▄[/] ",
        "  [#cdd6f4]▒█[/]    [#cdd6f4]▒█[/]   [#cdd6f4]▓█[/] ",
        " [#fab387]●[/] [#cdd6f4]▒█[/] [#f9e2af]▲▲[/]  [#cdd6f4]▒█[/] [#cdd6f4]▓█[/] ",
        "[#cdd6f4]████[/]  [#fab387]▼▶◀○[/] [#f9e2af]▼[/][#cdd6f4]██[/]  ",
        "[#cdd6f4]███[/]       [#f9e2af]▼[/][#cdd6f4]██[/]   ",
        " [#cdd6f4]▀▀▀[/]       [#cdd6f4]▀▀[/]   ",
        "    [#cdd6f4]░░[/]     [#cdd6f4]░░[/]   ",
    ),
)

OPUS_VISION: tuple[tuple[str, ...], ...] = (
    (
        "                ",
        " [#94e2d5]▄▀▀▀▄[/]  [#f9e2af]◇[/] [#94e2d5]▄▀▀▀▄[/] ",
        "  [#cdd6f4]▄▄▄[/]      [#cdd6f4]▄▄▄[/]  ",
        "  [#cdd6f4]█[/][#181825]█[/][#cdd6f4]█[/]      [#cdd6f4]█[/][#181825]█[/][#cdd6f4]█[/]  ",
        "  [#cdd6f4]▀▀▀[/]      [#cdd6f4]▀▀▀[/]  ",
        " [#a6e3a1]●[/] [#a6e3a1]▀▄▄▄▄▄▄▄▄▀[/] [#a6e3a1]●[/] ",
        "                ",
    ),
    (
        " [#94e2d5]▄▀▀▀▄[/]  [#f9e2af]◆[/] [#94e2d5]▄▀▀▀▄[/] ",
        "  [#cdd6f4]▄▄▄[/]      [#cdd6f4]▄▄▄[/]  ",
        "  [#cdd6f4]█[/][#181825]█[/][#cdd6f4]█[/]      [#cdd6f4]█[/][#181825]█[/][#cdd6f4]█[/]  ",
        "  [#cdd6f4]▀▀▀[/]      [#cdd6f4]▀▀▀[/]  ",
        " [#a6e3a1]●[/] [#a6e3a1]▀▄▄▄▄▄▄▄▄▀[/] [#a6e3a1]●[/] ",
        "                ",
        "                ",
    ),
    (
        " [#94e2d5]▄▀▀▀▄[/]  [#f9e2af]◆[/] [#94e2d5]▄▀▀▀▄[/] ",
        "                ",
        "  [#94e2d5]▄▄▄[/]      [#94e2d5]▄▄▄[/]  ",
        "                ",
        " [#a6e3a1]●▀▄▄▄▄▄▄▄▄▄▄▀●[/] ",
        "                ",
        "                ",
    ),
    (
        "                ",
        " [#94e2d5]▄▀▀▀▄[/]  [#f9e2af]◇[/] [#94e2d5]▄▀▀▀▄[/] ",
        "                ",
        "  [#cdd6f4]█[/][#181825]█[/][#cdd6f4]█[/]      [#cdd6f4]█[/][#181825]█[/][#cdd6f4]█[/]  ",
        "  [#cdd6f4]▀▀▀[/]      [#cdd6f4]▀▀▀[/]  ",
        " [#a6e3a1]●[/] [#a6e3a1]▀▄▄▄▄▄▄▄▄▀[/] [#a6e3a1]●[/] ",
        "                ",
    ),
    (
        "                ",
        "                ",
        " [#94e2d5]▄▀▀▀▄[/]  [#f9e2af]◇[/] [#94e2d5]▄▀▀▀▄[/] ",
        "  [#cdd6f4]▄▄▄[/]      [#cdd6f4]▄▄▄[/]  ",
        "  [#cdd6f4]█[/][#181825]█[/][#cdd6f4]█[/]      [#cdd6f4]█[/][#181825]█[/][#cdd6f4]█[/]  ",
        "  [#cdd6f4]▀▀▀[/]      [#cdd6f4]▀▀▀[/]  ",
        " [#a6e3a1]●[/] [#a6e3a1]▀▄▄▄▄▄▄▄▄▀[/] [#a6e3a1]●[/] ",
    ),
    (
        "                ",
        "                ",
        " [#94e2d5]▄▀▀▀▄[/]  [#f9e2af]◆[/] [#94e2d5]▄▀▀▀▄[/] ",
        "  [#cdd6f4]▄▄▄[/]      [#cdd6f4]▄▄▄[/]  ",
        "  [#cdd6f4]█[/][#181825]█[/][#cdd6f4]█[/]      [#cdd6f4]█[/][#181825]█[/][#cdd6f4]█[/]  ",
        "  [#cdd6f4]▀▀▀[/]      [#cdd6f4]▀▀▀[/]  ",
        " [#a6e3a1]●[/] [#a6e3a1]▀▄▄▄▄▄▄▄▄▀[/] [#a6e3a1]●[/] ",
    ),
)

OPUS_EXPLOITER: tuple[tuple[str, ...], ...] = (
    (
        "       [#cdd6f4]▄▄[/]       ",
        "    [#cdd6f4]▄▄[/] [#f38ba8]█[/][#eba0ac]▓[/]   [#fab387]▄▄▄[/] ",
        "    [#f38ba8]█[/][#eba0ac]▓[/] [#f38ba8]█[/][#eba0ac]▓[/]   [#fab387]█[/] [#fab387]█[/] ",
        "[#eba0ac]▄▄▄[/] [#f38ba8]█[/][#eba0ac]▓[/] [#f38ba8]█[/][#eba0ac]▓[/]   [#fab387]▀▀▀[/] ",
        "[#eba0ac]▀▀▀[/] [#f38ba8]█▄[/]   [#f38ba8]▄█[/]     ",
        "     [#f38ba8]▀█[/][#cdd6f4]▄[/][#f38ba8]█▀[/]      ",
        "                ",
    ),
    (
        "    [#cdd6f4]▄▄[/] [#cdd6f4]▄▄[/]       ",
        "    [#f38ba8]█[/][#eba0ac]▓[/] [#f38ba8]█[/][#eba0ac]▓[/]   [#fab387]▄▄▄[/] ",
        " [#eba0ac]▄▄[/] [#f38ba8]█[/][#eba0ac]▓[/] [#f38ba8]█[/][#eba0ac]▓[/]   [#fab387]█[/] [#fab387]█[/] ",
        "[#eba0ac]███[/] [#f38ba8]█[/][#eba0ac]▓[/] [#f38ba8]█[/][#eba0ac]▓[/]   [#fab387]▀▀▀[/] ",
        "[#eba0ac]▀▀▀[/] [#fab387]█▄[/]   [#fab387]▄█[/]     ",
        "     [#fab387]▀█[/][#cdd6f4]▼[/][#fab387]█▀[/]      ",
        "                ",
    ),
    (
        "                ",
        "       [#cdd6f4]▄▄[/]       ",
        "    [#cdd6f4]▄▄[/] [#f38ba8]█[/][#eba0ac]▓[/]   [#fab387]▄▄▄[/] ",
        "    [#f38ba8]█[/][#eba0ac]▓[/] [#f38ba8]█[/][#eba0ac]▓[/]   [#fab387]█[/] [#fab387]█[/] ",
        "[#eba0ac]▄▄▄[/] [#f38ba8]█[/][#eba0ac]▓[/] [#f38ba8]█[/][#eba0ac]▓[/]   [#fab387]▀▀▀[/] ",
        "[#eba0ac]▀▀▀[/] [#f38ba8]█▄[/]   [#f38ba8]▄█[/]     ",
        "     [#f38ba8]▀█[/][#cdd6f4]▼[/][#f38ba8]█▀[/]      ",
    ),
    (
        "                ",
        "                ",
        "       [#cdd6f4]▄▄[/]   [#fab387]▄▄▄[/] ",
        "    [#cdd6f4]▄▄[/] [#f38ba8]█[/][#eba0ac]▓[/]   [#fab387]█[/] [#fab387]█[/] ",
        "    [#f38ba8]█[/][#eba0ac]▓[/] [#f38ba8]█[/][#eba0ac]▓[/]   [#fab387]▀▀▀[/] ",
        "[#eba0ac]▄▄▄[/] [#fab387]█▄[/]   [#fab387]▄█[/]     ",
        "     [#fab387]▀█[/][#cdd6f4]▼[/][#fab387]█▀[/]      ",
    ),
    (
        "                ",
        "       [#cdd6f4]▄▄[/]       ",
        "    [#cdd6f4]▄▄[/] [#f38ba8]█[/][#eba0ac]▓[/]   [#fab387]▄▄▄[/] ",
        "    [#f38ba8]█[/][#eba0ac]▓[/] [#f38ba8]█[/][#eba0ac]▓[/]   [#fab387]█[/] [#fab387]█[/] ",
        "[#eba0ac]▄▄▄[/] [#f38ba8]█[/][#eba0ac]▓[/] [#f38ba8]█[/][#eba0ac]▓[/]   [#fab387]▀▀▀[/] ",
        "[#eba0ac]▀▀▀[/] [#f38ba8]█▄[/]   [#f38ba8]▄█[/]     ",
        "     [#f38ba8]▀█[/][#cdd6f4]▄[/][#f38ba8]█▀[/]      ",
    ),
    (
        "       [#cdd6f4]▄▄[/]       ",
        "    [#cdd6f4]▄▄[/] [#f38ba8]█[/][#eba0ac]▓[/]   [#fab387]▄▄▄[/] ",
        " [#eba0ac]▄▄[/] [#f38ba8]█[/][#eba0ac]▓[/] [#f38ba8]█[/][#eba0ac]▓[/]   [#fab387]█[/] [#fab387]█[/] ",
        "[#eba0ac]███[/] [#f38ba8]█[/][#eba0ac]▓[/] [#f38ba8]█[/][#eba0ac]▓[/]   [#fab387]▀▀▀[/] ",
        "[#eba0ac]▀▀▀[/] [#fab387]█▄[/]   [#fab387]▄█[/]     ",
        "     [#fab387]▀█[/][#cdd6f4]▼[/][#fab387]█▀[/]      ",
        "                ",
    ),
)

GROK_VISION: tuple[tuple[str, ...], ...] = (
    (
        "                ",
        "  [#a6e3a1]▄▀▀▀▄[/][#f9e2af]◆[/][#a6e3a1]▄▀▀▀▄[/]   ",
        "  [#89dceb]▄██████████▄[/]  ",
        "  [#89dceb]███[/][#cdd6f4]▄[/][#89dceb]████[/][#cdd6f4]▄[/][#89dceb]███[/]  ",
        "  [#89dceb]███[/][#cdd6f4]█[/][#89dceb]████[/][#cdd6f4]█[/][#89dceb]███[/]  ",
        "  [#89dceb]████[/][#cdd6f4]▀▄▄▀[/][#89dceb]████[/]  ",
        "  [#89dceb]▀██████████▀[/]  ",
    ),
    (
        "  [#a6e3a1]▄▀▀▀▄[/][#f9e2af]◆[/][#a6e3a1]▄▀▀▀▄[/]   ",
        "  [#89dceb]▄██████████▄[/]  ",
        "  [#89dceb]███[/][#cdd6f4]▄[/][#89dceb]████[/][#cdd6f4]▄[/][#89dceb]███[/]  ",
        "  [#89dceb]███[/][#cdd6f4]█[/][#89dceb]████[/][#cdd6f4]█[/][#89dceb]███[/]  ",
        "  [#89dceb]████[/][#cdd6f4]▀▄▄▀[/][#89dceb]████[/]  ",
        "  [#89dceb]▀██████████▀[/]  ",
        "    [#89dceb]▀▀████▀▀[/]    ",
    ),
    (
        "  [#a6e3a1]▄▀▀▀▄[/][#f9e2af]◇[/][#a6e3a1]▄▀▀▀▄[/]   ",
        "  [#89dceb]▄██████████▄[/]  ",
        "  [#89dceb]███[/][#cdd6f4]▀[/][#89dceb]████[/][#cdd6f4]▀[/][#89dceb]███[/]  ",
        "  [#89dceb]████████████[/]  ",
        "  [#89dceb]████[/][#cdd6f4]▀▄▄▀[/][#89dceb]████[/]  ",
        "  [#89dceb]▀██████████▀[/]  ",
        "    [#89dceb]▀▀████▀▀[/]    ",
    ),
    (
        "  [#a6e3a1]▄▀▀▀▄[/][#f9e2af]◆[/][#a6e3a1]▄▀▀▀▄[/]   ",
        "   [#89dceb]▄████████▄[/]   ",
        "   [#89dceb]██[/][#cdd6f4]▄[/][#89dceb]████[/][#cdd6f4]▄[/][#89dceb]██[/]   ",
        "   [#89dceb]██[/][#cdd6f4]█[/][#89dceb]████[/][#cdd6f4]█[/][#89dceb]██[/]   ",
        "   [#89dceb]███[/][#cdd6f4]▀▄▄▀[/][#89dceb]███[/]   ",
        "   [#89dceb]▀████████▀[/]   ",
        "     [#89dceb]▀▀▀▀▀▀[/]     ",
    ),
    (
        "  [#a6e3a1]▄▀▀▀▄[/][#f9e2af]◆[/][#a6e3a1]▄▀▀▀▄[/]   ",
        "  [#89dceb]▄██████████▄[/]  ",
        "  [#89dceb]███[/][#cdd6f4]▄[/][#89dceb]████[/][#cdd6f4]▄[/][#89dceb]███[/]  ",
        "  [#89dceb]███[/][#cdd6f4]█[/][#89dceb]████[/][#cdd6f4]█[/][#89dceb]███[/]  ",
        "  [#89dceb]████[/][#cdd6f4]▀▄▄▀[/][#89dceb]████[/]  ",
        "  [#89dceb]▀██████████▀[/]  ",
        "                ",
    ),
    (
        "                ",
        "  [#a6e3a1]▄▀▀▀▄[/][#f9e2af]◆[/][#a6e3a1]▄▀▀▀▄[/]   ",
        "[#89dceb]▄██████████████▄[/]",
        "[#89dceb]█████[/][#cdd6f4]▄[/][#89dceb]████[/][#cdd6f4]▄[/][#89dceb]█████[/]",
        "[#89dceb]█████[/][#cdd6f4]█[/][#89dceb]████[/][#cdd6f4]█[/][#89dceb]█████[/]",
        "[#89dceb]▀█████[/][#cdd6f4]▀▄▄▀[/][#89dceb]█████▀[/]",
        "  [#89dceb]▀▀▀▀▀▀▀▀▀▀▀▀[/]  ",
    ),
)

GROK_EXPLOITER: tuple[tuple[str, ...], ...] = (
    (
        "  [#cdd6f4]▒[/][#f38ba8]█[/]      [#cdd6f4]▒[/][#f38ba8]█[/] [#cdd6f4]▄▄[/] ",
        "  [#cdd6f4]▒[/][#f38ba8]███████[/][#cdd6f4]▒[/][#f38ba8]█[/] [#cdd6f4]▓█[/] ",
        " [#fab387]●[/][#cdd6f4]▒[/][#f38ba8]███[/][#f9e2af]▲▲[/][#f38ba8]██[/][#cdd6f4]▒[/][#f38ba8]█[/] [#cdd6f4]▓█[/] ",
        " [#f38ba8]████[/][#fab387]▼▶◀○[/][#f9e2af]▼[/][#f38ba8]██[/] [#cdd6f4]▀▀[/] ",
        " [#f38ba8]███[/]     [#f9e2af]▼[/][#f38ba8]██[/]    ",
        "  [#f38ba8]▀▀▀[/]     [#f38ba8]▀▀[/]    ",
        "                ",
    ),
    (
        "   [#cdd6f4]▒[/][#f38ba8]█[/]    [#cdd6f4]▒[/][#f38ba8]█[/]  [#cdd6f4]▄▄[/] ",
        "   [#cdd6f4]▒[/][#f38ba8]█████[/][#cdd6f4]▒[/][#f38ba8]█[/]  [#cdd6f4]▓█[/] ",
        " [#fab387]●[/] [#f38ba8]███[/][#f9e2af]▲▲[/][#f38ba8]███[/]  [#cdd6f4]●█[/] ",
        " [#f38ba8]████[/][#fab387]▼▶◀○[/][#f9e2af]▼[/][#f38ba8]██[/] [#cdd6f4]▀▀[/] ",
        " [#f38ba8]███[/]     [#f9e2af]▼[/][#f38ba8]██[/]    ",
        "  [#f38ba8]▀▀▀[/]     [#f38ba8]▀▀[/]    ",
        "                ",
    ),
    (
        "                ",
        "    [#f38ba8]▄█[/]   [#f38ba8]█▄[/]  [#cdd6f4]▄▄[/] ",
        " [#fab387]●[/]  [#cdd6f4]▒[/][#f38ba8]█[/][#f9e2af]▲▲▲[/][#cdd6f4]▒[/][#f38ba8]█[/]  [#cdd6f4]●█[/] ",
        " [#f38ba8]████[/][#fab387]▼▶◀○[/][#f9e2af]▼[/][#f38ba8]██[/] [#cdd6f4]▓█[/] ",
        " [#f38ba8]███[/]   [#f9e2af]▼▼[/] [#f38ba8]██[/] [#cdd6f4]▀▀[/] ",
        "  [#f38ba8]▀▀▀[/]     [#f38ba8]▀▀[/]    ",
        "    [#f38ba8]▀▀[/]          ",
    ),
    (
        "                ",
        "   [#f38ba8]▄█[/]     [#f38ba8]█▄[/] [#cdd6f4]▄▄[/] ",
        " [#fab387]●[/] [#cdd6f4]▒[/][#f38ba8]█[/]  [#f9e2af]▲▲[/] [#cdd6f4]▒[/][#f38ba8]█[/] [#cdd6f4]▓█[/] ",
        " [#f38ba8]████[/][#fab387]▼▶◀○[/][#f9e2af]▼[/][#f38ba8]██[/] [#cdd6f4]▓█[/] ",
        " [#f38ba8]███[/]     [#f9e2af]▼[/][#f38ba8]██[/] [#cdd6f4]▀▀[/] ",
        " [#f38ba8]██[/]      [#f9e2af]▼[/] [#f38ba8]█[/]    ",
        "  [#f38ba8]▀▀▀[/]     [#f38ba8]▀▀[/]    ",
    ),
    (
        "                ",
        "    [#f38ba8]▄█[/]   [#f38ba8]█▄[/]   [#cdd6f4]▄[/] ",
        " [#fab387]●[/]  [#cdd6f4]▒[/][#f38ba8]█[/] [#f9e2af]▲▲[/] [#cdd6f4]▒[/][#f38ba8]█[/] [#cdd6f4]▄█[/] ",
        " [#f38ba8]████[/][#fab387]▼▶◀○[/][#f9e2af]▼[/][#f38ba8]██[/] [#cdd6f4]●█[/] ",
        " [#f38ba8]███[/]   [#f9e2af]▼▼[/] [#f38ba8]██[/] [#cdd6f4]▓█[/] ",
        "  [#f38ba8]▀▀▀[/]     [#f38ba8]▀▀[/][#cdd6f4]▀▀[/]  ",
        "                ",
    ),
    (
        "  [#f38ba8]▄█[/]      [#f38ba8]▄█[/] [#cdd6f4]▄▄[/] ",
        "  [#cdd6f4]▒[/][#f38ba8]███████[/][#cdd6f4]▒[/][#f38ba8]█[/] [#cdd6f4]▓█[/] ",
        " [#fab387]●[/][#cdd6f4]▒[/][#f38ba8]███[/][#f9e2af]▲▲[/][#f38ba8]██[/][#cdd6f4]▒[/][#f38ba8]█[/] [#cdd6f4]▓█[/] ",
        " [#f38ba8]████[/][#fab387]▼▶◀○[/][#f9e2af]▼[/][#f38ba8]██[/] [#cdd6f4]▀▀[/] ",
        " [#f38ba8]███[/]     [#f9e2af]▼[/][#f38ba8]██[/]    ",
        "  [#f38ba8]▀▀▀[/]     [#f38ba8]▀▀[/]    ",
        "    [#f38ba8]░░[/]    [#f38ba8]░░[/]    ",
    ),
)

SKETCH_PACK = SpritePack(vision=SKETCH_VISION, exploiter=SKETCH_EXPLOITER)
OPUS_PACK = SpritePack(vision=OPUS_VISION, exploiter=OPUS_EXPLOITER)
GROK_PACK = SpritePack(vision=GROK_VISION, exploiter=GROK_EXPLOITER)
