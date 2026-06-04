from streamforge.clock import FrameBuffer, select_fill, RealtimeClock


def test_framebuffer_get_latest_does_not_consume_freshness():
    fb = FrameBuffer()
    assert fb.get_latest() is None
    fb.publish("ai-1")
    assert fb.get_latest() == "ai-1"           # peek, non-consuming
    # first freshness read after a publish is fresh, the next (no new publish) is a repeat
    assert fb.get_with_freshness().is_fresh is True
    r = fb.get_with_freshness()
    assert r.value == "ai-1" and r.is_fresh is False


def test_select_fill_priority_order():
    assert select_fill(ai="x", warped="w", held="h", raw="r").source == "ai"
    assert select_fill(ai=None, warped="w", held="h", raw="r").source == "warped"
    assert select_fill(ai=None, warped=None, held="h", raw="r").source == "held"
    assert select_fill(ai=None, warped=None, held=None, raw="r").source == "raw"


def test_clock_emits_at_target_count_and_counts_repeats():
    emitted = []
    fb = FrameBuffer()
    fb.publish(("frame", 0))
    clk = RealtimeClock(fps=50, frame_buffer=fb, emit=lambda f: emitted.append(f))
    clk.run_for_ticks(5)
    assert len(emitted) == 5
    assert clk.repeat_count == 4    # 1 fresh + 4 repeats (no new publishes)


def test_clock_resets_freshness_on_new_publish():
    emitted = []
    fb = FrameBuffer()
    fb.publish("a")
    clk = RealtimeClock(fps=30, frame_buffer=fb, emit=lambda f: emitted.append(f))
    clk.run_for_ticks(2)            # fresh, repeat
    fb.publish("b")                 # new frame -> next tick is fresh again
    clk.run_for_ticks(1)
    assert emitted == ["a", "a", "b"]
    assert clk.repeat_count == 1
