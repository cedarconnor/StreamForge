import torch
from streamforge.fill.warp import warp_forward


def test_identity_when_flow_zero():
    img = torch.rand(1, 3, 8, 10)
    flow = torch.zeros(1, 2, 8, 10)
    out = warp_forward(img, flow, dt=1.0, fps=1.0)
    assert torch.allclose(out, img, atol=1e-5)


def test_positive_x_flow_shifts_content_right():
    # a bright vertical line at column 2; +x flow moves content to the right
    img = torch.zeros(1, 3, 4, 8); img[..., 2] = 1.0
    flow = torch.zeros(1, 2, 4, 8); flow[:, 0] = 2.0   # +2 px/frame in x
    out = warp_forward(img, flow, dt=1.0, fps=1.0)      # disp = 2*1*1 = 2 px
    col = out[0, 0].sum(dim=0).argmax().item()
    assert col == 4                                     # 2 + 2


def test_max_disp_clamps():
    img = torch.zeros(1, 3, 4, 16); img[..., 2] = 1.0
    flow = torch.zeros(1, 2, 4, 16); flow[:, 0] = 10.0
    out = warp_forward(img, flow, dt=1.0, fps=1.0, max_disp=3.0)
    col = out[0, 0].sum(dim=0).argmax().item()
    assert col == 5                                     # 2 + 3 (clamped)


def test_flow_resized_to_image_res():
    # flow at half the image resolution must be upscaled AND magnitude-scaled
    img = torch.zeros(1, 3, 4, 8); img[..., 1] = 1.0
    flow = torch.zeros(1, 2, 2, 4); flow[:, 0] = 1.0    # 1 px at flow-res (width 4)
    out = warp_forward(img, flow, dt=1.0, fps=1.0)      # scaled by 8/4=2 -> disp 2 px
    col = out[0, 0].sum(dim=0).argmax().item()
    assert col == 3                                     # 1 + 2
