import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="RAFT needs CUDA")
def test_raft_detects_rightward_shift():
    from streamforge.fill.flow import RaftFlow
    base = torch.rand(1, 3, 128, 128)
    shifted = torch.zeros_like(base); shifted[..., 8:] = base[..., :-8]   # content moved +8 in x
    rf = RaftFlow(device="cuda", max_side=128)
    flow = rf.estimate(base, shifted)                  # prev=base, cur=shifted
    assert flow.shape[1] == 2
    assert flow[:, 0, 32:96, 32:96].mean().item() > 0  # central x-flow is positive
