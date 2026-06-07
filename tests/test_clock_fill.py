import torch
from streamforge.clock import FrameBuffer, RealtimeClock
from streamforge.frame import GpuFrame


class _StubFiller:
    def __init__(self, out): self._out = out; self.calls = 0
    def fill(self, now): self.calls += 1; return self._out


def _frame(val=0.0):
    return GpuFrame(tensor=torch.full((1, 3, 2, 2), val), seq=1, pts=0.0, width=2, height=2)


def test_fresh_tick_emits_ai_not_warped():
    fb = FrameBuffer(); emitted = []
    f = _frame(0.0); fb.publish(f)
    clk = RealtimeClock(30, fb, emitted.append, filler=_StubFiller(torch.ones(1, 3, 2, 2)))
    clk.run_for_ticks(1)
    assert emitted[0] is f and clk.filled_count == 0


def test_stale_tick_emits_warped_with_held_metadata():
    fb = FrameBuffer(); emitted = []
    f = _frame(0.0); fb.publish(f)
    warped_t = torch.ones(1, 3, 2, 2)
    clk = RealtimeClock(30, fb, emitted.append, filler=_StubFiller(warped_t))
    clk.run_for_ticks(2)                       # tick1 fresh, tick2 stale
    assert emitted[0] is f
    assert torch.equal(emitted[1].tensor, warped_t)
    assert emitted[1].seq == f.seq             # warped frame carries held metadata
    assert clk.filled_count == 1


def test_stale_tick_holds_when_filler_returns_none():
    fb = FrameBuffer(); emitted = []
    f = _frame(0.0); fb.publish(f)
    clk = RealtimeClock(30, fb, emitted.append, filler=_StubFiller(None))
    clk.run_for_ticks(2)
    assert emitted[1] is f                      # held
    assert clk.filled_count == 0 and clk.repeat_count == 1


def test_no_filler_is_backcompat_repeat():
    fb = FrameBuffer(); emitted = []
    f = _frame(0.0); fb.publish(f)
    clk = RealtimeClock(30, fb, emitted.append)   # filler=None
    clk.run_for_ticks(2)
    assert emitted[0] is f and emitted[1] is f
    assert clk.repeat_count == 1 and clk.filled_count == 0
