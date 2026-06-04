from streamforge.bench.harness import BenchHarness
from streamforge.bench.report import to_table, to_json
from streamforge.sources.synthetic import SyntheticSource


class _FakeRuntime:
    def set_prompt(self, p):
        pass

    def restyle(self, img, params):
        return img   # passthrough


def test_harness_collects_perstage_stats():
    src = SyntheticSource(8, 8, 30)
    h = BenchHarness(source=src, runtime=_FakeRuntime(), frames=20, fps=30)
    report = h.run()
    assert "infer" in report.stages
    assert report.stages["infer"].n == 20
    assert report.missed_deadlines >= 0
    assert report.frame_repeats == 0     # every frame is fresh in this fake
    assert report.frames == 20


def test_report_renders():
    src = SyntheticSource(8, 8, 30)
    report = BenchHarness(source=src, runtime=_FakeRuntime(), frames=5, fps=30).run()
    table = to_table(report)
    assert "stage" in table and "infer" in table
    assert '"frames": 5' in to_json(report)
