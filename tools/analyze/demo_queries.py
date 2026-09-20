"""Precompute query embeddings for the browser demo.

Live embedding needs onnxruntime-web plus the 23 MB encoder. Shipping vectors for a
fixed query list exercises the identical retrieval path -- only the source of the
query vector differs -- and keeps the demo to a few hundred kilobytes.

Paths are derived from this file's location rather than written out absolutely, so
the script runs from any checkout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from embed.dense import DenseEncoder  # noqa: E402

DEMO_DOCS = 2000

enc = DenseEncoder(
    REPO / "model_qint8_arm64.onnx",
    REPO / "models" / "tokenizers" / "bert-base-uncased-vocab.txt",
    threads=2,
)

qs = [
    "guitar music concert stage",
    "recipe flour sugar baking",
    "quantum physics particles",
    "government election policy",
    "computer software programming",
]

# Plus some known-item queries drawn from real documents, which have a correct answer.
with open(REPO / "data/corpus/queries-10k.jsonl", encoding="utf-8") as f:
    for line in f:
        q = json.loads(line)
        if (
            q["kind"] == "known_item"
            and q["k"] == 10
            and q["source_doc"] < DEMO_DOCS
            and len(qs) < 12
        ):
            qs.append(q["text"])

E = enc.encode(qs)
out = {q: [round(float(x), 6) for x in E[i]] for i, q in enumerate(qs)}

dest = REPO / "web/demo/queries.js"
dest.parent.mkdir(parents=True, exist_ok=True)
with open(dest, "w", encoding="utf-8") as f:
    f.write(
        "// Precomputed query embeddings. Live embedding needs onnxruntime-web plus the\n"
        "// 23 MB encoder; the retrieval path exercised below is identical either way.\n"
    )
    f.write("window.__ANNLITE_QUERIES__ = " + json.dumps(out) + ";\n")
print("queries:", list(out)[:6], "...", len(out), "total")
print(f"-> {dest}")
