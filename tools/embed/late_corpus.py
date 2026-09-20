"""Embed a corpus or query set with the late-interaction encoder.

The dense encoder writes one fixed-width row per document, so its workers can share
one memory-mapped output and write straight into their own slice (see `__main__.py`).
A late-interaction encoder emits *one vector per token*, and token counts vary per
document, so no worker knows where its slice begins until every earlier worker has
finished. Each worker therefore writes its own shard pair and the parent concatenates
them in order.

Vectors and lengths are written together, never in separate passes. `code_eval.py`
records why: an earlier version left the length table as a manual conversion step, so
re-running the encoder updated the vectors while the lengths stayed behind, and the
Rust side would have sliced 495,075 vectors with a table describing 487,313. The two
files are only meaningful as a pair, so nothing here can produce one without the other.

    python -m tools.embed.late_corpus --input data/corpus/docs-10k.txt \
        --out data/embeddings/words-10k-late.f32
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embed.late import LateEncoder  # noqa: E402
from corpus.code import unescape  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO / "model_int8.onnx"
DEFAULT_TOKENIZER = REPO / "tokenizer.json"
LATE_DIM = 48


def _read_texts(path: Path, start: int, end: int, escaped: bool) -> list[str]:
    """Lines [start, end) of a corpus, or the `text` fields of a query JSONL."""
    out = []
    jsonl = path.suffix == ".jsonl"
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= end:
                break
            if i >= start:
                if jsonl:
                    out.append(json.loads(line)["text"])
                else:
                    out.append(unescape(line) if escaped else line.rstrip("\n"))
    return out


def _lengths_path(out: Path) -> Path:
    """`x-late.f32` -> `x-late-lengths.i32`, so the pair is named by one argument."""
    return out.with_suffix("").with_name(out.with_suffix("").name + "-lengths").with_suffix(".i32")


def _shard_paths(out: Path, i: int) -> tuple[Path, Path]:
    return (out.with_suffix(f".shard{i}.f32"), out.with_suffix(f".shard{i}.i32"))


def _worker(args) -> tuple[int, int, float]:
    inp, out, shard, start, end, batch, threads, model, tok, escaped = args
    enc = LateEncoder(model, tok, threads=threads)
    texts = _read_texts(Path(inp), start, end, escaped)
    vec_path, len_path = _shard_paths(Path(out), shard)
    t0 = time.time()
    tokens = 0
    with open(vec_path, "wb") as vf, open(len_path, "wb") as lf:
        for i in range(0, len(texts), batch):
            mats = enc.encode(texts[i : i + batch])
            for m in mats:
                vf.write(np.ascontiguousarray(m, dtype=np.float32).tobytes())
                tokens += m.shape[0]
            lf.write(np.array([m.shape[0] for m in mats], dtype=np.int32).tobytes())
    return len(texts), tokens, time.time() - t0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="corpus .txt or query .jsonl")
    p.add_argument("--out", required=True, type=Path, help="output .f32 matrix")
    p.add_argument(
        "--lengths",
        type=Path,
        default=None,
        help="token-count sidecar (default: <out> with '-lengths.i32'). The readers "
        "expect '<prefix>-lengths.i32' for documents and '<prefix>-qlengths.i32' for "
        "queries, so a query set needs this passed explicitly.",
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="embed only the first N rows")
    p.add_argument(
        "--unescape",
        action="store_true",
        help="undo the code corpus's line encoding (never for word corpora)",
    )
    a = p.parse_args()

    n = sum(1 for _ in open(a.input, "rb"))
    if a.limit:
        n = min(n, a.limit)
    a.out.parent.mkdir(parents=True, exist_ok=True)

    workers = max(1, min(a.workers, n))
    bounds = [(n * i // workers, n * (i + 1) // workers) for i in range(workers)]
    jobs = [
        (
            str(a.input), str(a.out), i, s, e, a.batch_size, a.threads_per_worker,
            str(a.model), str(a.tokenizer), a.unescape,
        )
        for i, (s, e) in enumerate(bounds)
        if e > s
    ]

    print(f"{n:,} rows -> {a.out} ({len(jobs)} workers)", flush=True)
    t0 = time.time()
    if len(jobs) == 1:
        results = [_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            results = pool.map(_worker, jobs)
    encode_s = time.time() - t0

    # Concatenate shards in corpus order. Document `i` of the output is document `i`
    # of the input, which everything downstream (gold ids, query alignment) assumes.
    lengths_path = a.lengths or _lengths_path(a.out)
    lengths_path.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    with open(a.out, "wb") as vf, open(lengths_path, "wb") as lf:
        for i in range(len(jobs)):
            vec_path, len_path = _shard_paths(a.out, i)
            with open(vec_path, "rb") as s:
                while chunk := s.read(1 << 22):
                    vf.write(chunk)
            lens = np.fromfile(len_path, dtype=np.int32)
            total_tokens += int(lens.sum())
            lf.write(lens.tobytes())
            vec_path.unlink()
            len_path.unlink()

    written = a.out.stat().st_size
    expected = total_tokens * LATE_DIM * 4
    if written != expected:
        raise SystemExit(
            f"error: {a.out} is {written} bytes but its length table describes "
            f"{expected} ({total_tokens} tokens x {LATE_DIM} dims x 4). "
            "The pair is inconsistent; not leaving it on disk to be read as valid."
        )

    rate = n / encode_s if encode_s else float("inf")
    print(
        f"{total_tokens:,} tokens ({total_tokens / n:.1f}/doc), "
        f"{written / 1e6:.1f} MB in {encode_s:.1f}s ({rate:.0f} docs/s)"
    )
    print(f"-> {a.out}\n-> {lengths_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
