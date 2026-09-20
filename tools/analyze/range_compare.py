"""What a real SQLite client costs over HTTP when it does not speculate.

RESEARCH_LOG.md section 13.2 measured sql.js-httpvfs escalating its read-ahead until
it had fetched the whole database: about 5.2 MB of a 5.9 MB file for one query. That
made the browser path useless as an instrument -- any page-locality work in the index
is invisible to a client that has already downloaded everything -- and left section
13.4's crossover question unanswerable through SQLite itself.

`web/sqlite-wasm` is the other instrument: the same SQLite the native benchmarks link
(3.46.0, from the amalgamation libsqlite3-sys vendors), compiled with emscripten,
with a VFS whose `xRead` issues one HTTP Range request for exactly the bytes asked
for. This script drives it over the network simulator and records what crossed.

Every run is counted twice and the two must agree: the VFS counts what SQLite asked
for, the simulator's request log counts what arrived. They are different processes,
and a disagreement would mean something in between is caching or coalescing, which
would make every page number here meaningless. That is the same discipline section
9.3 used when it checked a pass-through VFS against SQLITE_DBSTATUS_CACHE_MISS.

    make range-demo
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WASM = REPO / "web/sqlite-wasm/annlite-sqlite.js"
MEASURE = REPO / "web/sqlite-wasm/measure.js"
NETSIM = REPO / "target/release/annlite-netsim"
OUT = REPO / "bench/results/range-vfs.jsonl"

# The unit operations a client actually performs, named by what they are rather than
# by their SQL. The two four-node rows are the point of the exercise: the same number
# of records, once with ids scattered across the table and once with them adjacent.
# Adjacency is what breadth-first ordering (section 11.4) buys a graph frontier, and
# the difference between those two rows is that ordering's value measured through
# SQLite rather than through a flat file.
QUERIES = [
    ("open only", None),
    ("point lookup (1 node)", "SELECT id, length(rec) FROM annlite_nodes WHERE id = 1337"),
    (
        "4 nodes, scattered ids",
        "SELECT id, length(rec) FROM annlite_nodes WHERE id IN (17, 533, 1337, 1904)",
    ),
    (
        "4 nodes, adjacent ids",
        "SELECT id, length(rec) FROM annlite_nodes WHERE id IN (1337, 1338, 1339, 1340)",
    ),
    ("document text (1 doc)", "SELECT length(body) FROM annlite_docs WHERE id = 1337"),
    ("resident code blob", "SELECT length(codes) FROM annlite_codeblob WHERE id = 1"),
    ("full table scan", "SELECT count(*) FROM annlite_nodes"),
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_one(url: str, sql: str | None) -> dict:
    cmd = ["node", str(MEASURE), "--url", url, "--json"]
    if sql:
        cmd += ["--sql", sql]
    else:
        # There is no such thing as opening without a statement, so the "open only"
        # row runs the cheapest statement there is and reports only its open cost.
        cmd += ["--sql", "SELECT 1"]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO / "web/sqlite-wasm")
    if p.returncode != 0:
        raise SystemExit(f"measure.js failed:\n{p.stdout}\n{p.stderr}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main() -> int:
    if not WASM.exists():
        raise SystemExit(f"error: {WASM} not built.\n  make sqlite-wasm")
    if not NETSIM.exists():
        raise SystemExit(f"error: {NETSIM} not built.\n  cargo build --release -p annlite-netsim")
    if not shutil.which("node"):
        raise SystemExit("error: node not found; the WASM client runs under node.")

    dbs = sorted(
        (p for p in (REPO / "web/demo").glob("annlite-demo*.db")),
        key=lambda p: p.stat().st_size,
    )
    if not dbs:
        raise SystemExit(
            "error: no demo database.\n"
            "  cargo run --release -p annlite-sqlite --example build_demo -- 2000 "
            "web/demo/annlite-demo.db"
        )

    port = free_port()
    log = REPO / "data/netsim-range.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        log.unlink()

    # `ideal` because this measurement is about counts, not seconds: latency would
    # only make it slow. The seconds-per-link conversion is netcost.py's job, from
    # exactly these counts.
    server = subprocess.Popen(
        [str(NETSIM), "--root", str(REPO / "web/demo"), "--addr", f"127.0.0.1:{port}",
         "--profile", "ideal", "--log", str(log)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise SystemExit("error: the simulator never came up")

        rows = []
        print(f"{'database':>12} {'query':<24} {'pages':>6} {'requests':>9} {'bytes':>10} "
              f"{'of file':>8}")
        print("-" * 74)
        for db in dbs:
            size = db.stat().st_size
            url = f"http://127.0.0.1:{port}/{db.name}"
            for label, sql in QUERIES:
                before = sum(1 for _ in open(log)) if log.exists() else 0
                r = run_one(url, sql)
                stage = r["open"] if sql is None else r["query"]
                after = sum(1 for _ in open(log)) if log.exists() else 0
                # The server sees this process's open requests too; the comparison
                # that matters is per-stage and comes from the client, so the log is
                # used as a total-request cross-check rather than a per-stage one.
                served = after - before
                if not r["vfs_and_transport_agree"]:
                    raise SystemExit(f"VFS and transport disagree on {label!r} -- not recording")
                frac = stage["bytes"] / size
                print(f"{size / 1e6:>10.1f}MB {label:<24} {stage['pages']:>6} "
                      f"{stage['requests']:>9} {stage['bytes']:>10,} {frac:>7.2%}")
                rows.append({
                    "record": "system",
                    "system": "sqlite-wasm/bounded-range",
                    "client": "annlite httprange VFS (emscripten, SQLite 3.46.0)",
                    "database": db.name,
                    "database_bytes": size,
                    "documents": None,
                    "query": label,
                    "sql": sql,
                    "pages": stage["pages"],
                    "requests": stage["requests"],
                    "bytes": stage["bytes"],
                    "fraction_of_file": frac,
                    "open_pages": r["open"]["pages"],
                    "open_bytes": r["open"]["bytes"],
                    "server_requests_observed": served,
                    "vfs_and_transport_agree": True,
                    "notes":
                        "One HTTP Range request per xRead, for exactly the bytes SQLite "
                        "asked for: no read-ahead, no speculation, no caching. Counted "
                        "independently by the VFS (what SQLite asked for) and by the "
                        "simulator's request log (what arrived); a run is discarded "
                        "unless they agree. 'open' is the header and schema read, paid "
                        "once per session rather than per query, so it is reported "
                        "separately. Compare with RESEARCH_LOG 13.2, where sql.js-httpvfs "
                        "fetched about 5.2 MB of this same 5.9 MB database for one query.",
                })
            print("-" * 74)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"-> {OUT} ({len(rows)} rows)")
    finally:
        server.terminate()
        server.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
