"""How each system's query cost responds to having several requests in flight.

Round-trips are the cost that matters over a network, but they are not all equal.
Requests *within* a hop can be overlapped, because their addresses are all known at
once. Hops cannot, because the next hop's addresses are not known until the current
one returns. So parallelism divides the first and leaves the second untouched, and
every system's latency is bounded below by `hops * RTT` no matter how wide the pipe.

That bound is where the three architectures genuinely differ, and it is invisible in
a page count:

* **FTS5 through a VFS cannot batch at all.** `xRead` is synchronous and
  one page at a time, so SQLite asks for its next page only after the current one
  returns. Its hop count equals its page count, and extra parallelism cannot reach
  it. This is a property of the client, not of the index.
* **The graph index batches within a hop but has many hops**, one per beam round,
  because each round's frontier depends on the last.
* **Late interaction has two or three hops regardless of corpus size**, because its
  stages are fixed: postings, then centroid scoring, then an optional rerank. It
  issues many requests but almost all of them are independent.

Hop counts are read from the measurement files and never inferred. An earlier
version of this analysis fell back to `hops = requests` when the field was absent,
which silently made every batchable system look serial.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from analyze.netcost import (  # noqa: E402
    CONCURRENCY_LEVELS,
    PAGE_BYTES,
    PROFILES,
    Access,
    query_seconds,
)


def load(pattern: str) -> list[dict]:
    out = []
    for f in glob.glob(str(REPO / "bench" / "results" / pattern)):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out


def dense_hops(config: str) -> float:
    """Beam rounds for a dense configuration, from the sweep that measured it.

    `tri-code.jsonl` records pages and requests but not hops, so the figure comes
    from the per-configuration sweep. Reranking adds one dependent round: the pool
    is not known until the traversal finishes.
    """
    parts = dict(p.split("=") for p in config.split("/") if "=" in p)
    ordering = config.split("/")[0].capitalize()
    mode = config.split("/")[1]
    suffix = "-m64" if parts.get("m") == "64" else ""
    rows = load(f"ann-code{suffix}.jsonl")
    for r in rows:
        if (
            r.get("record") == "query_set"
            and r["ordering"] == ordering
            and r["mode"] == mode
            and r["l"] == int(parts["L"])
            and r["beam"] == int(parts["beam"])
            and r["rerank"] == int(parts["rerank"])
        ):
            return r["mean_hops"] + (1 if r["rerank"] else 0)
    raise KeyError(f"no measured hop count for dense configuration {config!r}")


def rows_for_comparison() -> list[tuple[str, Access]]:
    tri = load("tri-*.jsonl")

    def pick(system: str, config: str) -> dict:
        # Last match, not first. These files are written by whole-run truncation now,
        # but a file that did accumulate duplicates should publish its most recent
        # measurement rather than its oldest -- the reverse silently kept a stale
        # contended timing in the tables after section 18.2 re-measured it.
        matches = [r for r in tri if r["system"] == system and r["config"] == config]
        if not matches:
            raise KeyError(f"no measurement for {system} {config!r}")
        return matches[-1]

    out: list[tuple[str, Access]] = []

    f = pick("fts5", "post-vacuum")
    # Hops equal *requests*, not pages. The VFS is synchronous, so every request is
    # serial and the round-trip count is however many requests the client issues --
    # which is the coalesced figure, since sql.js-httpvfs merges adjacent pages into
    # one range. Charging pages here while crediting coalesced requests would take
    # the pessimistic reading of one and the optimistic reading of the other.
    out.append((
        "FTS5",
        Access(hops=round(f["requests"]["mean"]), requests=round(f["requests"]["mean"]),
               bytes_fetched=round(f["pages"]["mean"] * PAGE_BYTES), batchable=False),
    ))

    for label, cfg in [
        ("dense m=64, no rerank", "bfs/resident/L=128/beam=4/rerank=0/m=64"),
        ("dense m=32, rerank 100", "bfs/resident/L=128/beam=4/rerank=100/m=32"),
    ]:
        d = pick("dense", cfg)
        out.append((label, Access(
            hops=round(dense_hops(cfg)), requests=round(d["requests"]["mean"]),
            bytes_fetched=round(d["pages"]["mean"] * PAGE_BYTES), batchable=True)))

    for label, cfg in [
        ("late k=1024, no rerank", "k=1024/probe=8/rerank=0"),
        ("late k=1024, rerank 100", "k=1024/probe=8/rerank=100"),
    ]:
        l = pick("late", cfg)
        if "hops" not in l:
            raise KeyError(f"late row {cfg!r} carries no hop count")
        out.append((label, Access(
            hops=round(l["hops"]), requests=round(l["requests"]["mean"]),
            bytes_fetched=round(l["pages"]["mean"] * PAGE_BYTES), batchable=True)))

    return out


def main() -> int:
    profile = sys.argv[1] if len(sys.argv) > 1 else "lte"
    rtt, mbps = PROFILES[profile]
    rows = rows_for_comparison()

    fmt = lambda v: f"{v:.2f} s" if v < 100 else f"{v:.0f} s"  # noqa: E731
    print(f"Code corpus, `{profile}` ({rtt:g} ms RTT, {mbps:g} Mbit/s). "
          "Seconds per query by requests in flight.\n")
    head = "".join(f"{'c=' + str(c):>10}" for c in CONCURRENCY_LEVELS)
    print(f"{'system':<26}{'hops':>6}{'reqs':>6}{head}{'floor':>10}")
    print("-" * (26 + 12 + 10 * (len(CONCURRENCY_LEVELS) + 1)))
    for label, acc in rows:
        cells = "".join(
            f"{fmt(query_seconds(acc, profile, c)['total_s']):>10}"
            for c in CONCURRENCY_LEVELS
        )
        floor = fmt(query_seconds(acc, profile, 1)["floor_s"])
        print(f"{label:<26}{acc.hops:>6}{acc.requests:>6}{cells}{floor:>10}")
    print("\nfloor = hops x RTT + transfer: the cost unlimited parallelism cannot remove.")
    print("FTS5 is unbatchable -- a synchronous VFS cannot have two reads outstanding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
