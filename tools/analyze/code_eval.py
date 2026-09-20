"""Head-to-head on the code-search corpus: BM25 against dense against late interaction.

This is the comparison the late-interaction model was built for -- a natural-language
query against a code document -- and the one the random-word corpus cannot support.
Ground truth comes from the corpus construction (a docstring's own function), so no
judgments are involved and success@k is exact.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from embed.dense import DenseEncoder  # noqa: E402
from embed.late import LateEncoder, maxsim  # noqa: E402


def unescape(line: str) -> str:
    return line.rstrip("\n").replace("\\n", "\n").replace("\\\\", "\\")


def attainable_ceiling(queries: list[dict], all_queries: list[dict]) -> tuple[float, int]:
    """Highest success@1 any system could reach on this query set.

    The benchmark is known-item retrieval: each query has exactly one correct
    document. But a docstring that appears verbatim on several functions makes those
    queries mutually indistinguishable -- the query text carries nothing that could
    separate them -- so for a docstring shared by `m` functions, any fixed ranking
    answers exactly one of those `m` queries correctly.

    The ceiling is therefore the mean of `1/m`, and quoting success@1 against 1.0
    rather than against this number overstates how much headroom is left.
    """
    import collections

    multiplicity = collections.Counter(q["text"] for q in all_queries)
    ambiguous = sum(1 for q in queries if multiplicity[q["text"]] > 1)
    ceiling = sum(1.0 / multiplicity[q["text"]] for q in queries) / len(queries)
    return ceiling, ambiguous


def metrics(ranks: list[int]) -> dict:
    r = np.array(ranks, dtype=float)
    return {
        "success@1": float((r <= 1).mean()),
        "success@10": float((r <= 10).mean()),
        "success@100": float((r <= 100).mean()),
        "mrr@10": float(np.mean(np.where(r <= 10, 1.0 / r, 0.0))),
    }


def main() -> int:
    n_q = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    docs = [unescape(l) for l in open(REPO / "data/corpus/code-docs.txt", encoding="utf-8")]
    qs = [json.loads(l) for l in open(REPO / "data/corpus/code-queries.jsonl", encoding="utf-8")]
    all_qs = qs
    qs = qs[:n_q]
    ceiling, ambiguous = attainable_ceiling(qs, all_qs)
    print(f"{len(docs)} documents, {len(qs)} queries", flush=True)
    print(f"{ambiguous} queries ({ambiguous / len(qs) * 100:.1f}%) share a docstring with "
          f"another function and cannot be answered as known-item retrieval; "
          f"maximum attainable success@1 is {ceiling:.3f}", flush=True)
    results = {}

    # --- BM25 via FTS5 -----------------------------------------------------
    t0 = time.time()
    con = sqlite3.connect(":memory:")
    con.execute("CREATE VIRTUAL TABLE d USING fts5(body)")
    con.executemany("INSERT INTO d(rowid, body) VALUES (?,?)", enumerate(docs))
    con.commit()
    build_bm25 = time.time() - t0
    ranks = []
    t0 = time.time()
    for q in qs:
        # OR the terms so partial matches rank rather than requiring all of them.
        terms = [w for w in "".join(c if c.isalnum() else " " for c in q["text"]).split() if w]
        if not terms:
            ranks.append(10**6)
            continue
        expr = " OR ".join(f'"{w}"' for w in terms)
        rows = con.execute(
            "SELECT rowid FROM d WHERE d MATCH ? ORDER BY bm25(d) LIMIT 100", (expr,)
        ).fetchall()
        ids = [r[0] for r in rows]
        ranks.append(ids.index(q["source_doc"]) + 1 if q["source_doc"] in ids else 10**6)
    results["BM25 (FTS5)"] = {**metrics(ranks), "build_s": round(build_bm25, 1),
                              "ms_per_query": round((time.time() - t0) * 1000 / len(qs), 2)}
    print("BM25 done", flush=True)

    # --- Dense: MiniLM, mean-pooled ---------------------------------------
    enc = DenseEncoder(REPO / "model_qint8_arm64.onnx",
                       REPO / "models/tokenizers/bert-base-uncased-vocab.txt", threads=2)
    t0 = time.time()
    D = np.vstack(list(enc.encode_stream(iter(docs), batch_size=16)))
    build_dense = time.time() - t0
    Q = np.vstack(list(enc.encode_stream(iter([q["text"] for q in qs]), batch_size=16)))
    t0 = time.time()
    order = np.argsort(-(Q @ D.T), axis=1)
    dense_ms = (time.time() - t0) * 1000 / len(qs)
    ranks = [int(np.where(order[i] == q["source_doc"])[0][0]) + 1 for i, q in enumerate(qs)]
    results["Dense (MiniLM)"] = {**metrics(ranks), "build_s": round(build_dense, 1),
                                 "ms_per_query": round(dense_ms, 2),
                                 "bytes_per_doc": 384 * 4}
    print("dense done", flush=True)

    # --- Late interaction: LateOn, exact MaxSim ---------------------------
    late = LateEncoder(REPO / "model_int8.onnx", REPO / "tokenizer.json", threads=2)
    t0 = time.time()
    DV = []
    for i in range(0, len(docs), 8):
        DV.extend(late.encode(docs[i:i + 8]))
    build_late = time.time() - t0
    QV = []
    for i in range(0, len(qs), 8):
        QV.extend(late.encode([q["text"] for q in qs[i:i + 8]]))
    tokens = sum(d.shape[0] for d in DV)
    t0 = time.time()
    ranks = []
    for i, q in enumerate(qs):
        scores = np.array([maxsim(QV[i], d) for d in DV])
        order = np.argsort(-scores)
        ranks.append(int(np.where(order == q["source_doc"])[0][0]) + 1)
    late_ms = (time.time() - t0) * 1000 / len(qs)
    results["Late interaction (LateOn)"] = {
        **metrics(ranks), "build_s": round(build_late, 1),
        "ms_per_query": round(late_ms, 2),
        "bytes_per_doc": round(tokens * 48 * 4 / len(docs)),
        "mean_tokens_per_doc": round(tokens / len(docs), 1),
    }

    print(f"\n{'system':<28} {'succ@1':>7} {'of max':>7} {'succ@10':>8} {'succ@100':>9} "
          f"{'MRR@10':>7} {'build_s':>8} {'B/doc':>8}")
    print("-" * 88)
    for name, m in results.items():
        print(f"{name:<28} {m['success@1']:>7.3f} {m['success@1'] / ceiling:>7.3f} "
              f"{m['success@10']:>8.3f} {m['success@100']:>9.3f} {m['mrr@10']:>7.3f} "
              f"{m['build_s']:>8.1f} {str(m.get('bytes_per_doc','-')):>8}")
    print(f"\nsuccess@1 ceiling {ceiling:.3f}; 'of max' is success@1 divided by it.")
    print("Per-query latency is not reported: these runs share the machine with index")
    print("builds, and the timings move by more than 2x with background load.")

    out = REPO / "bench/results/code-eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "documents": len(docs), "queries": len(qs),
        "success_at_1_ceiling": round(ceiling, 4),
        "ambiguous_queries": ambiguous,
        "results": results,
    }, indent=2) + "\n")
    print(f"\n-> {out}")
    # Hand the multi-vector embeddings to the PLAID measurement.
    #
    # Lengths are written as raw int32 in the same pass as the vectors they describe,
    # not as .npy converted later. An earlier version left that conversion as a
    # manual step, and re-running this script updated the vectors while the lengths
    # stayed behind -- so the Rust side would have been slicing 495,075 vectors with
    # a table describing 487,313. Writing both here makes the pair inseparable.
    emb = REPO / "data/embeddings"
    emb.mkdir(parents=True, exist_ok=True)
    np.array([d.shape[0] for d in DV], dtype=np.int32).tofile(emb / "code-late-lengths.i32")
    np.vstack(DV).astype(np.float32).tofile(emb / "code-late.f32")
    np.array([q.shape[0] for q in QV], dtype=np.int32).tofile(emb / "code-late-qlengths.i32")
    np.vstack(QV).astype(np.float32).tofile(emb / "code-late-q.f32")
    print(f"multi-vector embeddings: {tokens:,} document tokens -> {emb}/code-late.f32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
