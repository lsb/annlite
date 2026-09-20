"""Join the code-corpus runs into one record per (system, configuration).

The three retrieval systems are measured by three different programs writing three
different result formats, which is fine for each on its own and useless for comparing
them. This flattens FTS5 (`fts5-code.jsonl`) and the dense Vamana+PQ index
(`ann-code*.jsonl`) into the shared schema in `bench/results/tri-code.jsonl`, so a
row is a configuration and every row carries the same columns.

Nothing here computes a metric. Every number is copied from a run that happened;
where a field has no measured value the row says so rather than carrying a guess.
Rows written by other systems -- late interaction is measured separately -- are left
untouched, so the three can be appended independently and in any order.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "bench" / "results"
OUT = RESULTS / "tri-code.jsonl"

#: Systems this script owns. Any other row already in the output file is preserved.
OWNED = {"fts5", "dense"}

DOCS = 3366
QUERIES = 500

#: Every timing in this file was taken on a machine running other benchmark work, so
#: it is an upper bound rather than a measurement. Page counts, byte counts and
#: quality figures are unaffected by that and are the numbers to trust.
def _observed_contention() -> bool:
    """Whether anything else was competing for CPU, read rather than assumed.

    Earlier runs hardcoded this True, which was correct while three benchmarks shared
    the machine and became silently wrong once it went quiet. `/proc/loadavg`'s fourth
    field is `running/total`; a lone benchmark plus this reader is two.
    """
    try:
        with open("/proc/loadavg") as f:
            fields = f.read().split()
        return int(fields[3].split("/")[0]) > 2 or float(fields[0]) > 1.5
    except (OSError, ValueError, IndexError):
        return True


CONTENDED = _observed_contention()


def records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def quality_from(known: dict) -> dict:
    """The four published cutoffs out of `annlite_fts5::metrics::KnownItemQuality`.

    Indexing by cutoff rather than by position, because the two sides agreeing on
    *which* k is being reported is the entire point of sharing the arithmetic.
    """
    s = dict(known["success_at"])
    m = dict(known["mrr_at"])
    return {
        "success@1": s[1],
        "success@10": s[10],
        "success@100": s[100],
        "mrr@10": m[10],
    }


def row(system: str, config: str, **kw) -> dict:
    out = {
        "record": "system",
        "corpus": "code",
        "docs": DOCS,
        "queries": QUERIES,
        "system": system,
        "config": config,
    }
    out.update(kw)
    out["contended"] = CONTENDED
    return out


# --- FTS5 ---------------------------------------------------------------------------

#: `phase` in the FTS5 output is the state of the file the phase measured, which is
#: what a deployment would actually serve.
FTS5_PHASES = {
    "pre_optimize": "as-built",
    "post_optimize": "post-optimize",
    "post_vacuum": "post-vacuum",
}


def fts5_rows() -> list[dict]:
    recs = records(RESULTS / "fts5-code.jsonl")
    if not recs:
        return []
    by = lambda kind: [r for r in recs if r["record"] == kind]  # noqa: E731
    build = by("build")[0]["build"]
    # File size is a property of the phase, not of the run: `optimize` and `VACUUM`
    # each rewrite the file, and a CDN ships whichever one was published.
    size = {"pre_optimize": build["stats"]}
    for r in by("optimize"):
        size["post_optimize"] = r["optimize"]["stats"]
    for r in by("vacuum"):
        size["post_vacuum"] = r["vacuum"]["stats"]

    out = []
    for phase, label in FTS5_PHASES.items():
        pages = [r for r in by("pages") if r["phase"] == phase and r["group"] == "all"]
        qual = [r for r in by("quality") if r["phase"] == phase]
        cpu = [r for r in by("cpu") if r["phase"] == phase]
        if not pages or not qual or phase not in size:
            continue
        p = pages[0]
        stats = size[phase]
        content = next(
            (t["pages"] for t in stats.get("per_table", []) if t["name"] == "docs_content"), 0
        )
        out.append(row(
            "fts5", label,
            build_seconds=round(build["build_secs"], 3),
            index_bytes=stats["file_bytes"],
            bytes_per_doc=round(stats["file_bytes"] / DOCS, 1),
            quality=quality_from(qual[0]["known_item"]["overall"]),
            pages={"mean": p["distinct_pages"]["mean"],
                   "median": p["distinct_pages"]["median"],
                   "p95": p["distinct_pages"]["p95"]},
            requests={"mean": p["contiguous_runs"]["mean"],
                      "median": p["contiguous_runs"]["median"]},
            cpu_ms_per_query=round(cpu[0]["cpu_ms_per_query"], 3) if cpu else None,
            notes=(
                f"SQLite {[r for r in by('meta')][0]['sqlite_version']}, 4 KiB pages, "
                "ORDER BY bm25() LIMIT 100. Pages are distinct 4 KiB file pages observed "
                "by a pass-through VFS intercepting xRead, and agreed with "
                "SQLITE_DBSTATUS_CACHE_MISS on 500/500 queries; requests are maximal runs "
                "of consecutive page numbers. Cold pager cache per query via PRAGMA "
                "shrink_memory, so the ~4 pages of connection setup are excluded. "
                f"index_bytes is the whole file; {content} of its {stats['page_count']} "
                "pages are docs_content, which these queries never read because they "
                "select rowid and score only. Timing is process CPU time on a shared "
                "machine, so it is an upper bound."
            ),
        ))
    return out


# --- Dense (MiniLM + Vamana/PQ) -----------------------------------------------------

#: Wall-clock seconds the MiniLM pass over the corpus took, read from the sidecar the
#: encoder wrote. Charged into build_seconds because a dense index is not queryable
#: without it, and leaving it out would make the build look 5x cheaper than it is.
def embed_seconds() -> float:
    meta = json.loads((REPO / "data/embeddings/code-dense.f32.json").read_text())
    return meta["seconds"]


def dense_rows() -> list[dict]:
    out = []
    embed_s = embed_seconds()
    for path in sorted(RESULTS.glob("ann-code*.jsonl")):
        recs = records(path)
        if not recs:
            continue
        meta = recs[0]
        if "exact_known_item" not in meta:
            continue
        index_by_ordering = {
            r["ordering"]: r for r in recs if r["record"] == "index"
        }
        m = meta["m"]

        # The ceiling: the same vectors, nothing approximated. Emitted once, from the
        # first file that carries it, because it does not depend on the index.
        if not any(r["config"].startswith("exact") for r in out):
            matrix_bytes = DOCS * meta["dim"] * 4
            out.append(row(
                "dense", "exact brute force (no index)",
                build_seconds=round(embed_s, 3),
                index_bytes=matrix_bytes,
                bytes_per_doc=round(matrix_bytes / DOCS, 1),
                quality=quality_from(meta["exact_known_item"]),
                # A full scan reads the whole matrix, so these are arithmetic on its
                # size rather than an instrumented traversal: every query touches every
                # page, once, in one sequential range.
                pages={"mean": float(-(-matrix_bytes // 4096)),
                       "median": float(-(-matrix_bytes // 4096)),
                       "p95": float(-(-matrix_bytes // 4096))},
                requests={"mean": 1.0, "median": 1.0},
                cpu_ms_per_query=round(meta["exact_cpu_ms_per_query"], 3),
                notes=(
                    "MiniLM mean-pooled, L2-normalised, 256-token cap; exact inner "
                    "product over all 3,366 documents. This is the quality ceiling for "
                    "every dense row below -- any shortfall there is the quantizer or "
                    "the graph, not the embeddings. pages/requests are arithmetic on the "
                    "5,170,176-byte matrix (a full scan), not an instrumented run. "
                    f"build_seconds is the {embed_s:.1f}s encoder pass on 4 worker "
                    "processes, wall clock on a shared machine."
                ),
            ))

        for r in recs:
            if r["record"] != "query_set":
                continue
            idx = index_by_ordering[r["ordering"]]
            build_s = (embed_s + meta["pq_train_seconds"]
                       + meta["vamana_build_seconds"] + idx["write_seconds"])
            config = (f"{r['ordering'].lower()}/{r['mode']}/L={r['l_effective']}"
                      f"/beam={r['beam']}/rerank={r['rerank']}/m={m}")
            traversal = sum(t["bytes"] for t in idx["bytes_by_table"]
                            if t["name"] in ("annlite_nodes", "annlite_codeblob"))
            out.append(row(
                "dense", config,
                build_seconds=round(build_s, 3),
                index_bytes=idx["db_bytes"],
                bytes_per_doc=round(idx["db_bytes"] / DOCS, 1),
                quality=quality_from(r["known_item"]),
                # The VFS figures, not the layout-derived ones, because only these
                # are the same kind of measurement as the FTS5 rows: real file pages,
                # so real file adjacency.
                pages={"mean": r["vfs_pages_per_query"]["mean"],
                       "median": r["vfs_pages_per_query"]["median"],
                       "p95": r["vfs_pages_per_query"]["p95"]},
                requests={"mean": r["vfs_runs_per_query"]["mean"],
                          "median": r["vfs_runs_per_query"]["median"]},
                cpu_ms_per_query=round(r["cpu_ms_per_query"], 3),
                notes=(
                    f"Vamana R={meta['r']} alpha={meta['alpha']:.1f} L_build="
                    f"{meta['l_build']}, PQ m={m} trained on all "
                    f"{meta['pq_train_vectors']} vectors, {idx['record_bytes']}-byte "
                    "records. Pages and requests are 4 KiB file pages seen by the same "
                    "pass-through VFS the FTS5 rows use, with a cold pager cache per "
                    "query, so the two systems' figures are the same kind of thing. "
                    "They include reranking's reads of annlite_vectors when rerank>0. "
                    "The layout-derived counts RESEARCH_LOG.md sections 11 and 16 report "
                    f"are lower -- {r['pages_per_query']['mean']:.1f} pages in "
                    f"{r['runs_per_query']['mean']:.1f} runs for this row -- and the gap "
                    "in runs is the important one: write_index fills annlite_nodes, "
                    "annlite_vectors and annlite_docs in one interleaved row-by-row pass, "
                    "so SQLite allocates their leaf pages round-robin and consecutive "
                    "node records land ~15 file pages apart. The id-to-page arithmetic "
                    "still holds within the node table; file adjacency does not. "
                    f"index_bytes is the whole file; only {traversal:,} bytes of it "
                    "(annlite_nodes + annlite_codeblob) are read by traversal, the rest "
                    "being the float32 vectors reranking needs and the PQ codebook. "
                    "'resident' excludes the one-off "
                    f"{idx['code_blob_bytes']:,}-byte code-blob download. build_seconds "
                    f"is {embed_s:.1f}s encoding + {meta['pq_train_seconds']:.1f}s PQ + "
                    f"{meta['vamana_build_seconds']:.1f}s graph + write, all wall clock "
                    "on a shared machine; cpu_ms_per_query is process CPU time."
                ),
            ))
    return out


def main() -> int:
    kept = [r for r in records(OUT) if r.get("system") not in OWNED]
    rows = fts5_rows() + dense_rows()
    if not rows:
        print("no code-corpus results found; run the FTS5 and dense measurements first",
              file=sys.stderr)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in kept + rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows written, {len(kept)} rows from other systems kept -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
