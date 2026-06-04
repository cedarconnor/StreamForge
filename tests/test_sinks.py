import pathlib

import torch

from streamforge.frame import GpuFrame
from streamforge.sinks.null_sink import NullSink
from streamforge.sinks.file_sink import FileSink


def _frame():
    return GpuFrame(tensor=torch.zeros(1, 3, 16, 16), seq=0, pts=0.0, width=16, height=16)


def test_null_sink_counts():
    s = NullSink()
    s.open()
    s.send(_frame())
    s.send(_frame())
    s.close()
    assert s.count == 2


def test_file_sink_writes(tmp_path):
    s = FileSink(str(tmp_path))
    s.open()
    s.send(_frame())
    s.close()
    assert len(list(pathlib.Path(tmp_path).glob("*.png"))) == 1
