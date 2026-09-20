"""The multi-vector encoder's file naming, which the readers depend on exactly.

A late-interaction corpus is two files that are only meaningful together: the vectors
and the per-document token counts that say where each document starts. `code_eval.py`
records what happens when they drift apart — the Rust side sliced 495,075 vectors with
a table describing 487,313 — and the Rust readers find the sidecar by *name*, building
`<prefix>-lengths.i32` for documents and `<prefix>-qlengths.i32` for queries.

Those two conventions are not the same string, and deriving the query one from the
output path silently produces `-q-lengths.i32`, which no reader looks for. The
encoder therefore takes the query sidecar as an explicit argument, and this pins both
halves of the convention so a rename cannot quietly break the pair.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from embed.late_corpus import _lengths_path, _shard_paths  # noqa: E402


def test_document_sidecar_matches_what_the_readers_open():
    out = Path("data/embeddings/words-10k-late.f32")
    assert _lengths_path(out).name == "words-10k-late-lengths.i32"


def test_code_corpus_naming_is_unchanged():
    # late_pages.rs and residual_eval.rs both open exactly this name.
    assert _lengths_path(Path("data/embeddings/code-late.f32")).name == "code-late-lengths.i32"


def test_query_sidecar_is_not_derivable_from_the_output_path():
    # The derived name is wrong for a query set; this is why --lengths exists and is
    # passed explicitly by the Makefile. If this ever starts matching, the flag has
    # become redundant and the Makefile should be simplified rather than left to
    # disagree with the code.
    derived = _lengths_path(Path("data/embeddings/words-10k-late-q.f32")).name
    assert derived == "words-10k-late-q-lengths.i32"
    assert derived != "words-10k-late-qlengths.i32"


def test_shards_are_distinct_per_worker_and_carry_both_halves():
    out = Path("data/embeddings/words-10k-late.f32")
    a_vec, a_len = _shard_paths(out, 0)
    b_vec, b_len = _shard_paths(out, 1)
    assert a_vec != b_vec and a_len != b_len
    # Shards must not collide with the final output, or a worker would overwrite it.
    assert a_vec != out and b_vec != out
    assert a_vec.suffix == ".f32" and a_len.suffix == ".i32"


def test_shard_paths_stay_in_the_output_directory():
    out = Path("data/embeddings/words-10k-late.f32")
    for p in _shard_paths(out, 3):
        assert p.parent == out.parent
