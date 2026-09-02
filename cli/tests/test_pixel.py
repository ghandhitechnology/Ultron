from ultron.cli.pixel import (
    FRAME_COUNT,
    SPRITE_HEIGHT,
    SPRITE_WIDTH,
    SpriteId,
    frames_for,
    mascot_strip,
    sprite_block,
    visible_text,
)


def test_both_mascots_have_a_fixed_size_idle_loop() -> None:
    for sprite in SpriteId:
        frames = frames_for(sprite)
        assert len(frames) == FRAME_COUNT
        for frame in frames:
            assert len(frame) == SPRITE_HEIGHT
            for row in frame:
                assert len(visible_text(row)) == SPRITE_WIDTH


def test_mascot_strip_names_exploiter_and_vision() -> None:
    strip = mascot_strip(0)
    assert "exploiter" in strip
    assert "vision" in strip
    assert "sketch" not in strip
    assert "grok" not in strip
    assert "◆" in strip or "◇" in strip


def test_idle_loop_changes_the_drawn_frame() -> None:
    vision = [sprite_block(SpriteId.VISION, tick) for tick in range(FRAME_COUNT)]
    exploiter = [sprite_block(SpriteId.EXPLOITER, tick) for tick in range(FRAME_COUNT)]
    assert len(set(vision)) >= 4
    assert len(set(exploiter)) >= 4
    assert vision[0] != vision[3]
