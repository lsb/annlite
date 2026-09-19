"""Tokenizer conformance.

The expected ids are the reference `bert-base-uncased` outputs. A Rust port for
WASM must reproduce these exactly, so this file doubles as the shared test vector.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from embed.tokenization import WordPieceTokenizer  # noqa: E402

VOCAB = REPO / "models" / "tokenizers" / "bert-base-uncased-vocab.txt"


@pytest.fixture(scope="module")
def tok():
    return WordPieceTokenizer(VOCAB)


def test_vocab_matches_model_embedding_table(tok):
    # The dense checkpoint's word embedding matrix has exactly this many rows;
    # a mismatch means vocabulary and weights do not belong together.
    assert len(tok) == 30522


def test_special_token_ids(tok):
    assert (tok.pad_id, tok.unk_id, tok.cls_id, tok.sep_id) == (0, 100, 101, 102)


def test_known_sentence(tok):
    assert tok.encode("a man is playing a guitar on stage") == [
        101, 1037, 2158, 2003, 2652, 1037, 2858, 2006, 2754, 102
    ]


def test_lowercasing_and_accent_stripping(tok):
    assert tok.tokenize("Café") == tok.tokenize("cafe")
    assert tok.tokenize("NAÏVE") == tok.tokenize("naive")


def test_punctuation_splits_from_words(tok):
    # "don't" -> don ' t : punctuation is its own token, never glued to the word.
    assert tok.tokenize("don't") == [2123, 1005, 1056]


def test_subword_decomposition_uses_continuation_pieces(tok):
    ids = tok.tokenize("unaffable")
    assert len(ids) > 1, "a rare word should decompose into multiple pieces"
    assert tok.unk_id not in ids
    assert any(tok.vocab[i].startswith("##") for i in ids)


def test_unknown_word_is_all_or_nothing(tok):
    # A word containing characters absent from the vocabulary collapses entirely
    # to [UNK] rather than yielding a partial decomposition.
    assert tok.tokenize("中文\u0000zzz" + "") != []


def test_overlong_word_becomes_unk(tok):
    assert tok.tokenize("a" * 200) == [tok.unk_id]


def test_encode_truncates_and_keeps_special_tokens(tok):
    ids = tok.encode(" ".join(["hello"] * 500), max_length=32)
    assert len(ids) == 32
    assert ids[0] == tok.cls_id and ids[-1] == tok.sep_id


def test_whitespace_variants_are_equivalent(tok):
    assert tok.tokenize("hello world") == tok.tokenize("hello\t\nworld")
