"""BERT WordPiece tokenization.

Written out rather than taken from `transformers`/`tokenizers` for two reasons.
The project's browser target needs this same algorithm in Rust compiled to WASM,
and keeping one readable reference implementation makes it possible to pin the two
against shared test vectors (see `tests/test_tokenization.py`). It also removes a
heavyweight dependency from a pipeline whose only other need is onnxruntime.

This follows the original BERT `FullTokenizer`: basic tokenization (whitespace
splitting, control-character stripping, accent removal and lowercasing for uncased
models, punctuation splitting, CJK isolation) followed by greedy longest-match-first
WordPiece.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

MAX_CHARS_PER_WORD = 100


def _is_control(ch: str) -> bool:
    # Tab, newline and carriage return are whitespace for our purposes, not control
    # characters, even though Unicode classifies them as Cc.
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch).startswith("C")


def _is_whitespace(ch: str) -> bool:
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


def _is_punctuation(ch: str) -> bool:
    # BERT treats all non-alphanumeric ASCII as punctuation, which is broader than
    # Unicode's P* categories: '$', '+', '^' and friends split as punctuation too.
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp: int) -> bool:
    return (
        0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F
    )


class WordPieceTokenizer:
    """Greedy longest-match-first WordPiece over a BERT vocabulary file."""

    def __init__(self, vocab_path: str | Path, lowercase: bool = True):
        self.vocab: list[str] = Path(vocab_path).read_text(encoding="utf-8").splitlines()
        self.ids: dict[str, int] = {tok: i for i, tok in enumerate(self.vocab)}
        self.lowercase = lowercase
        for name in ("[PAD]", "[UNK]", "[CLS]", "[SEP]"):
            if name not in self.ids:
                raise ValueError(f"vocabulary {vocab_path} is missing {name}")
        self.pad_id = self.ids["[PAD]"]
        self.unk_id = self.ids["[UNK]"]
        self.cls_id = self.ids["[CLS]"]
        self.sep_id = self.ids["[SEP]"]

    def __len__(self) -> int:
        return len(self.vocab)

    def _basic_tokenize(self, text: str) -> list[str]:
        cleaned = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or _is_control(ch):
                continue
            if _is_whitespace(ch):
                cleaned.append(" ")
            elif _is_cjk(cp):
                # Isolate CJK so each character becomes its own token, as BERT does.
                cleaned.append(f" {ch} ")
            else:
                cleaned.append(ch)

        tokens: list[str] = []
        for word in "".join(cleaned).split():
            if self.lowercase:
                word = word.lower()
                # NFD then dropping combining marks is how the uncased models strip
                # accents; doing it after lowercasing matches the reference order.
                word = "".join(
                    c for c in unicodedata.normalize("NFD", word)
                    if unicodedata.category(c) != "Mn"
                )
            tokens.extend(self._split_punctuation(word))
        return tokens

    @staticmethod
    def _split_punctuation(word: str) -> list[str]:
        out: list[str] = []
        buf: list[str] = []
        for ch in word:
            if _is_punctuation(ch):
                if buf:
                    out.append("".join(buf))
                    buf = []
                out.append(ch)
            else:
                buf.append(ch)
        if buf:
            out.append("".join(buf))
        return out

    def _wordpiece(self, token: str) -> list[int]:
        if len(token) > MAX_CHARS_PER_WORD:
            return [self.unk_id]
        pieces: list[int] = []
        start = 0
        while start < len(token):
            end = len(token)
            found = None
            while start < end:
                sub = token[start:end]
                if start > 0:
                    sub = "##" + sub
                if sub in self.ids:
                    found = self.ids[sub]
                    break
                end -= 1
            if found is None:
                # WordPiece is all-or-nothing per word: if any piece fails to match,
                # the whole word becomes [UNK] rather than a partial decomposition.
                return [self.unk_id]
            pieces.append(found)
            start = end
        return pieces

    def tokenize(self, text: str) -> list[int]:
        """Token ids for `text`, without special tokens."""
        out: list[int] = []
        for tok in self._basic_tokenize(text):
            out.extend(self._wordpiece(tok))
        return out

    def encode(self, text: str, max_length: int = 256) -> list[int]:
        """Token ids wrapped in `[CLS]` … `[SEP]`, truncated to `max_length` total."""
        ids = self.tokenize(text)[: max_length - 2]
        return [self.cls_id, *ids, self.sep_id]
