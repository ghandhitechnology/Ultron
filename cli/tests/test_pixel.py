from ultron.cli.model import JobMeta, initial_snapshot
from ultron.cli.pixel import (
    FRAME_COUNT,
    SPRITE_HEIGHT,
    SPRITE_WIDTH,
    PixelStyle,
    SpriteId,
    advance_style_tick,
    frames_for,
    mascot_strip,
    sprite_block,
    style_at,
    visible_text,
)
from ultron.cli.render import footer_line, progress_block
from ultron.env.backend import IsolationBackend


def test_every_style_has_fixed_size_loop() -> None:
    for style in PixelStyle:
        for sprite in SpriteId:
            frames = frames_for(style, sprite)
            assert len(frames) == FRAME_COUNT
            for frame in frames:
                assert len(frame) == SPRITE_HEIGHT
                for row in frame:
                    assert len(visible_text(row)) == SPRITE_WIDTH


def test_styles_cycle_in_closed_order() -> None:
    assert [style_at(tick).value for tick in (0, 12, 24, 36)] == [
        "sketch",
        "opus",
        "grok",
        "sketch",
    ]
    assert style_at(3, pinned=PixelStyle.GROK) is PixelStyle.GROK
    assert style_at(advance_style_tick(0)) is PixelStyle.OPUS
    assert style_at(advance_style_tick(12)) is PixelStyle.GROK
    assert style_at(advance_style_tick(24)) is PixelStyle.SKETCH


def test_mascot_strip_names_both_sketch_characters() -> None:
    strip = mascot_strip(0)
    assert "exploiter" in strip
    assert "vision" in strip
    assert "pixel  sketch" in strip
    assert "◇" in strip or "◆" in strip
    opus = mascot_strip(12)
    assert "pixel  opus" in opus
    grok = mascot_strip(24)
    assert "pixel  grok" in grok


def test_sprite_block_changes_across_the_idle_loop() -> None:
    first = sprite_block(PixelStyle.OPUS, SpriteId.VISION, 0)
    blink = sprite_block(PixelStyle.OPUS, SpriteId.VISION, 2)
    assert first != blink


def test_gym_copy_mentions_pixel_cycle() -> None:
    snap = initial_snapshot(
        JobMeta(
            generation=0,
            profile_id="web",
            isolation=IsolationBackend.DOCKER,
            episodes_planned=1,
            turns_per_side=1,
        ),
        started_at_s=0.0,
    )
    assert "p pixel" in progress_block(snap)
    line = footer_line(snap, pixel_style=PixelStyle.OPUS)
    assert "pixel opus" in line
