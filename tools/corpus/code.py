"""A code-search corpus with ground truth, built from the local Python standard library.

`LateOn-Code-edge` is a code model, so evaluating it on bags of random dictionary
words measures nothing about what it was trained to do. This builds the standard
docstring-to-code benchmark instead, the same construction CodeSearchNet uses:

* A **document** is a function's source with its docstring removed.
* A **query** is the first sentence of that docstring.
* The **gold answer** is the function the docstring was taken from.

Relevance is therefore established by the corpus itself rather than by judgment, and
the query is natural language while the document is code -- which is exactly the
cross-modal retrieval the model exists for. Stripping the docstring matters: leaving
it in makes the query a literal substring of its answer and the task degenerates into
exact matching, which would flatter every lexical baseline and tell us nothing.

Determinism comes from sorting by a content hash rather than by filesystem order, so
the corpus does not depend on how the standard library happens to be laid out.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import sysconfig
from pathlib import Path

#: Functions shorter than this are usually one-line wrappers that carry no signal.
MIN_BODY_LINES = 3
#: Very long functions blow past the encoder's window and dominate MaxSim by length.
MAX_BODY_LINES = 60
MIN_DOC_CHARS = 30
MAX_DOC_CHARS = 300


def escape(code: str) -> str:
    """A function's source as one corpus line.

    Line number is the document id, exactly as in the word corpora, so a document has
    to fit on one line; newlines and backslashes are escaped rather than dropped.
    """
    return code.replace("\\", "\\\\").replace("\n", "\\n")


def unescape(line: str) -> str:
    """One corpus line back to the source it stands for.

    Every reader must apply this before indexing or embedding: left alone, the `\\n`
    markers reach a tokenizer as a stray backslash and `n`, which changes both the
    document length BM25 divides by and the tokens the encoder sees.

    Note that the replacements are *not* a strict inverse of `escape`: source that
    literally contains a backslash followed by `n` comes back as a backslash followed
    by a newline. The order is kept anyway, because every measurement published
    against this corpus -- Python and Rust alike -- decodes it this way, and changing
    it would silently move results on the documents it affects rather than fixing
    anything already reported.
    """
    return line.rstrip("\n").replace("\\n", "\n").replace("\\\\", "\\")


def _first_sentence(doc: str) -> str:
    text = " ".join(doc.strip().split())
    for stop in (". ", "! ", "? "):
        i = text.find(stop)
        if i > 0:
            return text[: i + 1].strip()
    return text[:MAX_DOC_CHARS].strip()


def _strip_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef, src: str) -> str | None:
    """Source of `node` with its docstring expression removed."""
    lines = src.splitlines()
    body = node.body
    if not body:
        return None
    first = body[0]
    is_doc = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    start = node.lineno - 1
    end = max(getattr(n, "end_lineno", node.lineno) for n in ast.walk(node))
    kept = lines[start:end]
    if is_doc:
        d_start = first.lineno - 1 - start
        d_end = (first.end_lineno or first.lineno) - start
        kept = kept[:d_start] + kept[d_end:]
    out = "\n".join(kept).rstrip()
    return out or None


def extract(roots: list[Path], limit: int = 0) -> list[dict]:
    """Every qualifying (query, document) pair under `roots`, deterministically ordered."""
    found: list[dict] = []
    files = sorted({p for root in roots for p in root.rglob("*.py")})
    for path in files:
        # Test fixtures and vendored copies skew the corpus toward boilerplate.
        parts = set(path.parts)
        if parts & {"test", "tests", "idlelib", "lib2to3", "__pycache__"}:
            continue
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            query = _first_sentence(doc)
            if not (MIN_DOC_CHARS <= len(query) <= MAX_DOC_CHARS):
                continue
            code = _strip_docstring(node, src)
            if code is None:
                continue
            n_lines = len(code.splitlines())
            if not (MIN_BODY_LINES <= n_lines <= MAX_BODY_LINES):
                continue
            found.append({
                "name": node.name,
                "module": str(path.name),
                "query": query,
                "code": code,
                "lines": n_lines,
            })

    # Deduplicate identical bodies: the standard library repeats some verbatim, and a
    # duplicate makes the "gold" document ambiguous.
    seen: set[str] = set()
    unique = []
    for f in found:
        h = hashlib.sha256(f["code"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        f["hash"] = h
        unique.append(f)

    # Order by content hash so the corpus does not depend on filesystem layout.
    unique.sort(key=lambda f: f["hash"])
    for i, f in enumerate(unique):
        f["id"] = i
    return unique[:limit] if limit else unique


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("data/corpus"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--roots", nargs="*", type=Path)
    args = p.parse_args()

    roots = args.roots or [Path(sysconfig.get_paths()["stdlib"])]
    roots = [r for r in roots if r.exists()]
    if not roots:
        print("error: no source roots found", file=sys.stderr)
        return 1

    items = extract(roots, args.limit)
    if not items:
        print("error: no functions with docstrings found", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    docs_path = args.out_dir / "code-docs.txt"
    qs_path = args.out_dir / "code-queries.jsonl"

    # One document per line, so line number is the document id exactly as with the
    # word corpora. Newlines inside code are escaped rather than dropped.
    with open(docs_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(escape(it["code"]) + "\n")
    with open(qs_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({
                "qid": it["id"], "kind": "docstring", "text": it["query"],
                "source_doc": it["id"], "name": it["name"], "module": it["module"],
            }) + "\n")

    digest = hashlib.sha256(docs_path.read_bytes()).hexdigest()
    (args.out_dir / "code-docs.manifest.json").write_text(json.dumps({
        "documents": len(items),
        "sha256": digest,
        "bytes": docs_path.stat().st_size,
        "mean_lines": round(sum(i["lines"] for i in items) / len(items), 1),
        "roots": [str(r) for r in roots],
        "generator": "tools/corpus/code.py",
    }, indent=2) + "\n")
    print(f"code corpus: {len(items)} functions -> {docs_path}")
    print(f"  sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
