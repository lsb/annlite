import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from analyze.netcost import Access, crossover, preload_seconds, query_seconds, session_seconds  # noqa: E402


def test_ideal_profile_is_free():
    c = query_seconds(Access(hops=10, requests=100, bytes_fetched=10**6), "ideal")
    assert c["total_s"] == 0.0


def test_serial_hops_dominate_on_high_latency_links():
    # One request per hop: the cost is purely the round-trips.
    c = query_seconds(Access(hops=40, requests=40, bytes_fetched=0), "satellite")
    assert c["latency_s"] == 40 * 0.6
    assert c["transfer_s"] == 0.0


def test_parallel_requests_within_a_hop_are_not_charged_serially():
    # 6 requests in one hop fit one wave at the default concurrency of 6.
    one = query_seconds(Access(hops=1, requests=6, bytes_fetched=0), "lte")
    assert one["latency_s"] == 0.07
    # 12 requests need two waves.
    two = query_seconds(Access(hops=1, requests=12, bytes_fetched=0), "lte")
    assert abs(two["latency_s"] - 0.14) < 1e-9


def test_fewer_hops_beats_fewer_requests_on_a_slow_link():
    narrow = Access(hops=40, requests=40, bytes_fetched=40 * 4096)
    wide = Access(hops=8, requests=128, bytes_fetched=128 * 4096)
    assert (
        query_seconds(wide, "satellite")["total_s"]
        < query_seconds(narrow, "satellite")["total_s"]
    ), "widening the beam should win when latency dominates"


def test_preload_is_one_round_trip_plus_transfer():
    # 15 Mbit/s, 15 MB payload -> 8 s transfer plus one 70 ms round-trip.
    s = preload_seconds(15_000_000, "lte")
    assert abs(s - (0.07 + 15_000_000 * 8 / 15e6)) < 1e-9


def test_resident_codes_win_only_after_enough_queries():
    resident = Access(hops=8, requests=16, bytes_fetched=16 * 4096, preload_bytes=64_000_000)
    on_disk = Access(hops=38, requests=400, bytes_fetched=400 * 4096)
    assert session_seconds(resident, "lte", 1) > session_seconds(on_disk, "lte", 1)
    assert session_seconds(resident, "lte", 500) < session_seconds(on_disk, "lte", 500)
    n = crossover(resident, on_disk, "lte")
    assert n is not None and 1 < n < 500


def test_crossover_returns_none_when_never_better():
    worse = Access(hops=100, requests=1000, bytes_fetched=10**7, preload_bytes=10**8)
    better = Access(hops=1, requests=1, bytes_fetched=4096)
    assert crossover(worse, better, "lte", max_queries=100) is None
