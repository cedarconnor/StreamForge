import torch
from streamforge.fill.filler import FrameFiller


def test_none_before_anchor():
    f = FrameFiller(max_extrap_ms=120, fps=10)
    assert f.fill(now=1.0) is None


def test_warps_within_cap():
    f = FrameFiller(max_extrap_ms=200, fps=10)
    img = torch.zeros(1, 3, 4, 8); img[..., 2] = 1.0
    flow = torch.zeros(1, 2, 4, 8); flow[:, 0] = 2.0
    f.set_anchor(img, flow, t=0.0)
    out = f.fill(now=0.1)                 # disp = 2 * 10 * 0.1 = 2 px
    assert out is not None
    assert out[0, 0].sum(dim=0).argmax().item() == 4


def test_none_past_time_cap():
    f = FrameFiller(max_extrap_ms=120, fps=10)
    f.set_anchor(torch.zeros(1, 3, 4, 8), torch.zeros(1, 2, 4, 8), t=0.0)
    assert f.fill(now=0.2) is None        # 200ms > 120ms cap


def test_ema_refines_fps_from_anchor_times():
    f = FrameFiller(max_extrap_ms=500, fps=100.0)   # bad nominal
    img = torch.zeros(1, 3, 4, 8); flow = torch.zeros(1, 2, 4, 8)
    f.set_anchor(img, flow, t=0.0)
    f.set_anchor(img, flow, t=0.1)        # 0.1s interval -> 10 fps observed
    assert 9.0 < f.fps < 100.0            # pulled toward 10 from 100 by the EMA


def test_fixed_fps_ignores_anchor_interval_ema():
    # SANA: anchors arrive per CHUNK (~1/s) but the flow passed is per-INPUT-frame displacement,
    # so warp scale must use the display rate, not the chunk arrival rate. fixed_fps pins it.
    f = FrameFiller(max_extrap_ms=500, fps=99.0, fixed_fps=30.0)
    img = torch.zeros(1, 3, 4, 8); flow = torch.zeros(1, 2, 4, 8)
    f.set_anchor(img, flow, t=0.0)
    f.set_anchor(img, flow, t=1.0)        # 1s "chunk" interval would EMA fps toward 1.0...
    assert f.fps == 30.0                   # ...but fixed_fps holds the warp scale at the display rate
