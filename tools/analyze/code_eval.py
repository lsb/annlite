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
    qs = qs[:n_q]
    print(f"{len(docs)} documents, {len(qs)} queries", flush=True)
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

    print(f"\n{'system':<28} {'succ@1':>7} {'succ@10':>8} {'succ@100':>9} {'MRR@10':>7} "
          f"{'build_s':>8} {'ms/query':>9} {'B/doc':>8}")
    print("-" * 92)
    for name, m in results.items():
        print(f"{name:<28} {m['success@1']:>7.3f} {m['success@10']:>8.3f} "
              f"{m['success@100']:>9.3f} {m['mrr@10']:>7.3f} {m['build_s']:>8.1f} "
              f"{m['ms_per_query']:>9.2f} {str(m.get('bytes_per_doc','-')):>8}")

    out = REPO / "bench/results/code-eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"documents": len(docs), "queries": len(qs),
                               "results": results}, indent=2) + "\n")
    print(f"\n-> {out}")
    # Keep the multi-vector embeddings for the PLAID index measurement.
    np.save(REPO / "data/embeddings/code-late-lengths.npy",
            np.array([d.shape[0] for d in DV], dtype=np.int32))
    np.vstack(DV).astype(np.float32).tofile(REPO / "data/embeddings/code-late.f32")
    np.save(REPO / "data/embeddings/code-late-qlengths.npy",
            np.array([q.shape[0] for q in QV], dtype=np.int32))
    np.vstack(QV).astype(np.float32).tofile(REPO / "data/embeddings/code-late-q.f32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
