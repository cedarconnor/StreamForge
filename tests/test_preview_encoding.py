import io

import torch
from PIL import Image

from streamforge.frame import GpuFrame
from streamforge.runner import frame_to_jpeg


def test_frame_to_jpeg_returns_valid_image_bytes():
    frame = GpuFrame(tensor=torch.zeros(1, 3, 4, 6), seq=0, pts=0.0, width=6, height=4)
    data = frame_to_jpeg(frame)
    assert data.startswith(b"\xff\xd8")
    img = Image.open(io.BytesIO(data))
    assert img.size == (6, 4)
