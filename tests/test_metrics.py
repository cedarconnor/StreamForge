from streamforge.metrics import percentile, jitter_ms, LatencyStats


def test_percentile_basic():
    data = list(range(1, 101))  # 1..100
    assert 50.0 <= percentile(data, 50) <= 51.0
    assert percentile(data, 99) >= 99.0
    assert percentile(data, 0) == 1
    assert percentile(data, 100) == 100


def test_jitter_is_spread_of_intervals():
    even = [i * (1 / 30) for i in range(10)]   # perfectly even ~33.3ms intervals
    assert jitter_ms(even) < 0.5
    uneven = [0, 0.033, 0.10, 0.12, 0.30]
    assert jitter_ms(uneven) > 5.0


def test_latencystats_rollup():
    s = LatencyStats.from_samples_ms([10, 20, 30, 40, 1000])
    assert s.p50 <= s.p95 <= s.p99 <= s.worst
    assert s.worst == 1000
    assert s.n == 5
