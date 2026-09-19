"""Precompute query embeddings for the browser demo.

Live embedding needs onnxruntime-web plus the 23 MB encoder. Shipping vectors for a
fixed query list exercises the identical retrieval path -- only the source of the
query vector differs -- and keeps the demo to a few hundred kilobytes.
"""
sys.path.insert(0,"/home/user/annlite/tools")
from embed.dense import DenseEncoder
enc = DenseEncoder("/home/user/annlite/model_qint8_arm64.onnx",
                   "/home/user/annlite/models/tokenizers/bert-base-uncased-vocab.txt", threads=2)
docs=[l.rstrip("\n") for l in open("/home/user/annlite/data/corpus/docs-10k.txt")][:2000]
qs=["guitar music concert stage","recipe flour sugar baking","quantum physics particles",
    "government election policy","computer software programming"]
# Plus some known-item queries drawn from real documents, which have a correct answer.
import json as j
for q in [j.loads(l) for l in open("/home/user/annlite/data/corpus/queries-10k.jsonl")]:
    if q["kind"]=="known_item" and q["k"]==10 and q["source_doc"]<2000 and len(qs)<12:
        qs.append(q["text"])
E = enc.encode(qs)
out={q: [round(float(x),6) for x in E[i]] for i,q in enumerate(qs)}
with open("/home/user/annlite/web/demo/queries.js","w") as f:
    f.write("// Precomputed query embeddings. Live embedding needs onnxruntime-web plus the\n"
            "// 23 MB encoder; the retrieval path exercised below is identical either way.\n")
    f.write("window.__ANNLITE_QUERIES__ = " + json.dumps(out) + ";\n")
print("queries:", list(out)[:6], "...", len(out), "total")
