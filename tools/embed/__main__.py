"""Embed a corpus or query set with the dense encoder.

Work is split across processes rather than across onnxruntime's intra-op threads.
A 6-layer model at batch 16 does not scale well to four threads -- the per-operator
work is too small to amortise the synchronisation -- whereas independent processes
on disjoint shards scale nearly linearly until memory bandwidth saturates.

Each worker writes directly into its own slice of a shared memory-mapped output
file, so there is no concatenation pass and peak memory stays at one batch per
worker regardless of corpus size.
"""

from __future__ import annotations

import argparse
import inspect
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embed.dense import EMBED_DIM, DenseEncoder  # noqa: E402
from embed import store  # noqa: E402
from corpus.code import unescape  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO / "model_qint8_arm64.onnx"
DEFAULT_VOCAB = REPO / "models" / "tokenizers" / "bert-base-uncased-vocab.txt"


def _count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def _read_texts(path: Path, start: int, end: int, escaped: bool = False) -> list[str]:
    """Lines [start, end) of a corpus, or the `text` fields of a query JSONL.

    `escaped` undoes the code corpus's line encoding. Skipping it does not fail, it
    quietly embeds something else: the `\\n` markers survive WordPiece as a backslash
    and an `n`, so every line break becomes two tokens of noise in a 256-token window.
    """
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


def _worker(args):
    inp, outp, start, end, rows, batch, threads, model, vocab, escaped = args
    enc = DenseEncoder(model, vocab, threads=threads)
    mm = store.open_memmap(outp, rows, EMBED_DIM, mode="r+")
    texts = _read_texts(Path(inp), start, end, escaped)
    pos = start
    t0 = time.time()
    for block in enc.encode_stream(iter(texts), batch_size=batch):
        mm[pos : pos + len(block)] = block
        pos += len(block)
    mm.flush()
    del mm
    return end - start, time.time() - t0


def main() -> int:
    p = argparse.ArgumentParser(prog="embed", description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="corpus .txt or query .jsonl")
    p.add_argument("--out", required=True, type=Path, help="output .f32 matrix")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="embed only the first N rows")
    p.add_argument("--unescape", action="store_true",
                   help="corpus lines are backslash-escaped (the code corpus); decode "
                        "them before tokenizing")
    args = p.parse_args()

    rows = _count_lines(args.input)
    if args.limit:
        rows = min(rows, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Preallocate so every worker can write its slice independently.
    store.open_memmap(args.out, rows, EMBED_DIM, mode="w+").flush()

    bounds = np.linspace(0, rows, args.workers + 1).astype(int)
    jobs = [
        (str(args.input), str(args.out), int(a), int(b), rows,
         args.batch_size, args.threads_per_worker, str(args.model), str(args.vocab),
         args.unescape)
        for a, b in zip(bounds[:-1], bounds[1:]) if b > a
    ]

    print(f"embedding {rows} rows from {args.input} with {len(jobs)} workers", flush=True)
    t0 = time.time()
    if len(jobs) == 1:
        results = [_worker(jobs[0])]
    else:
        with mp.Pool(len(jobs)) as pool:
            results = pool.map(_worker, jobs)
    elapsed = time.time() - t0

    done = sum(r[0] for r in results)
    store.write_manifest(
        args.out, rows, EMBED_DIM,
        source=str(args.input), model=args.model.name,
        pooling="masked-mean", normalized="l2", unescaped=args.unescape,
        # Recorded because it is a real confound, not a detail: RESEARCH_LOG.md 15.1
        # found an ungrounded sequence cap silently truncating documents.
        max_length=inspect.signature(DenseEncoder.__init__).parameters["max_length"].default,
        seconds=round(elapsed, 3), docs_per_second=round(done / elapsed, 1),
    )
    print(f"done: {done} rows in {elapsed:.1f}s ({done/elapsed:.1f} rows/s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
