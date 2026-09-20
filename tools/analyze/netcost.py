"""Convert measured access patterns into simulated query time.

The benchmarks report counts -- nodes read, distinct pages, contiguous runs, and
dependent hops. This module turns those into seconds under the `annlite-netsim`
profiles, which is the only unit in which the systems are actually comparable.

The model deliberately separates two things the raw counts conflate:

* **Dependent hops** are serial. The next hop's addresses are not known until the
  current hop's records come back, so each one costs a full round-trip that nothing
  can hide.
* **Fetches within a hop** are parallel, up to the client's connection limit. A
  browser opens ~6 connections per origin, so a frontier of 16 pages costs about
  three round-trips' worth of latency, not sixteen.

Ignoring that distinction is the most common way to make an index look worse (or
better) than it is, which is why hops are counted separately from pages throughout.
"""

from __future__ import annotations

from dataclasses import dataclass

# Matches the profile table in crates/annlite-netsim/src/profile.rs.
PROFILES: dict[str, tuple[float, float]] = {
    # name: (RTT milliseconds, throughput Mbit/s)
    "ideal": (0.0, float("inf")),
    "wifi": (15.0, 50.0),
    "5g": (35.0, 100.0),
    "lte": (70.0, 15.0),
    "leo": (45.0, 80.0),
    "3g": (200.0, 1.6),
    "slow-3g": (2000.0, 0.4),
    "satellite": (600.0, 20.0),
}

PAGE_BYTES = 4096

#: HTTP/1.1 browsers cap connections per origin at six. HTTP/2 and HTTP/3 multiplex
#: over one connection and allow ~100 concurrent streams, which is what a CDN
#: actually serves today, so six is a floor rather than the expected case. The
#: comparison sweeps this rather than assuming it.
DEFAULT_CONCURRENCY = 6
CONCURRENCY_LEVELS = (1, 6, 32, 128)


@dataclass
class Access:
    """One query's measured access pattern."""

    hops: int
    #: Requests issued, after any client-side coalescing of adjacent pages.
    requests: int
    bytes_fetched: int
    #: Bytes that can be fetched once per session rather than once per query,
    #: such as a resident PQ codebook.
    preload_bytes: int = 0
    #: Whether the client can have more than one request of a hop in flight.
    #:
    #: This is a property of the *client*, not of the network, and it is the most
    #: consequential asymmetry in the comparison. A purpose-built client -- the
    #: `SearchSession` in `annlite-wasm`, say -- hands out a whole frontier of node
    #: ids per round and can issue them together. SQLite reached through a VFS
    #: cannot: `xRead` is a synchronous, one-page-at-a-time interface, so FTS5 over
    #: `sql.js-httpvfs` discovers its next page only after the current one returns,
    #: whatever the link would allow. Setting this False pins effective concurrency
    #: at 1 no matter what is passed in.
    batchable: bool = True


def query_seconds(
    access: Access, profile: str, concurrency: int = DEFAULT_CONCURRENCY
) -> dict[str, float]:
    """Simulated wall-clock seconds for one query, excluding any preload.

    Reports `floor_s` alongside the total: `hops * rtt`, the cost that survives
    unlimited parallelism. Requests within a hop can be overlapped, so concurrency
    divides them; hops cannot, because the next hop's addresses are not known until
    the current one returns. Any system's latency is therefore bounded below by its
    hop count, and that bound is where the architectures actually differ.
    """
    rtt_ms, mbps = PROFILES[profile]
    rtt = rtt_ms / 1000.0

    effective = 1 if not access.batchable else max(1, concurrency)
    hops = max(access.hops, 1)
    per_hop = max(1, -(-access.requests // hops))
    waves = -(-per_hop // effective)
    latency = hops * waves * rtt

    transfer = 0.0 if mbps == float("inf") else access.bytes_fetched * 8 / (mbps * 1e6)
    return {
        "latency_s": latency,
        "transfer_s": transfer,
        "total_s": latency + transfer,
        "floor_s": hops * rtt + transfer,
    }


def concurrency_sweep(
    access: Access, profile: str, levels: tuple[int, ...] = CONCURRENCY_LEVELS
) -> dict[int, float]:
    """Total seconds at each concurrency level, for one access pattern."""
    return {c: query_seconds(access, profile, c)["total_s"] for c in levels}


def preload_seconds(preload_bytes: int, profile: str) -> float:
    """One sequential bulk download: a single round-trip plus transfer."""
    if preload_bytes <= 0:
        return 0.0
    rtt_ms, mbps = PROFILES[profile]
    transfer = 0.0 if mbps == float("inf") else preload_bytes * 8 / (mbps * 1e6)
    return rtt_ms / 1000.0 + transfer


def session_seconds(
    access: Access, profile: str, queries: int, concurrency: int = DEFAULT_CONCURRENCY
) -> float:
    """Total seconds for a session of `queries` queries, amortising the preload.

    This is the function that decides between architectures. An index that downloads
    all its PQ codes up front pays a large fixed cost and a small per-query one; an
    index that reads codes from disk pays nothing up front and much more per query.
    Which wins is not a property of either index but of how many queries a session
    asks, so it must be reported as a crossover rather than a winner.
    """
    per_query = query_seconds(access, profile, concurrency)["total_s"]
    return preload_seconds(access.preload_bytes, profile) + queries * per_query


def crossover(a: Access, b: Access, profile: str, max_queries: int = 10_000) -> int | None:
    """Smallest session length at which `a` becomes no slower than `b`, if any."""
    for n in range(1, max_queries + 1):
        if session_seconds(a, profile, n) <= session_seconds(b, profile, n):
            return n
    return None
