"""Dense sentence embeddings from `all-MiniLM-L6-v2` (`model_qint8_arm64.onnx`).

The ONNX graph emits `last_hidden_state` only; the pooling and normalisation that
turn it into a sentence embedding live here, and both matter. Mean-pooling must be
masked — averaging over padding positions drags embeddings toward whatever the model
emits for `[PAD]` and the damage grows with how much padding a batch carries, so an
unmasked mean makes a vector depend on its batchmates. L2 normalisation makes the
inner product equal cosine similarity, which is what lets every downstream index use
plain dot products.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import onnxruntime as ort

from .tokenization import WordPieceTokenizer

EMBED_DIM = 384


class DenseEncoder:
    def __init__(
        self,
        model_path: str | Path,
        vocab_path: str | Path,
        max_length: int = 256,
        threads: int | None = None,
    ):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = WordPieceTokenizer(vocab_path)
        self.max_length = max_length
        expected = {"input_ids", "attention_mask", "token_type_ids"}
        actual = {i.name for i in self.session.get_inputs()}
        if actual != expected:
            raise ValueError(f"unexpected model inputs {sorted(actual)}; expected {sorted(expected)}")

    def _batch_inputs(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        encoded = [self.tokenizer.encode(t, self.max_length) for t in texts]
        width = max(len(e) for e in encoded)
        ids = np.full((len(encoded), width), self.tokenizer.pad_id, dtype=np.int64)
        mask = np.zeros((len(encoded), width), dtype=np.int64)
        for i, e in enumerate(encoded):
            ids[i, : len(e)] = e
            mask[i, : len(e)] = 1
        return {
            "input_ids": ids,
            "attention_mask": mask,
            # Single-segment input, so all token types are 0.
            "token_type_ids": np.zeros_like(ids),
        }

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch, returning `(len(texts), 384)` L2-normalised float32."""
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        feeds = self._batch_inputs(texts)
        hidden = self.session.run(None, feeds)[0]

        mask = feeds["attention_mask"][..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-12)).astype(np.float32)

    def encode_stream(
        self, texts: Iterator[str], batch_size: int = 64, sort_by_length: bool = True
    ) -> Iterator[np.ndarray]:
        """Embed an iterator of texts in batches, yielding one array per batch.

        With `sort_by_length`, texts within a window are grouped by token count before
        batching and the results are restored to input order. Padding is charged at the
        length of the longest member of a batch, so grouping similar lengths together
        cuts wasted compute substantially on mixed-length corpora. Output order is
        unchanged, so callers cannot tell the difference except in speed.
        """
        window = batch_size * 32
        buf: list[str] = []
        for text in texts:
            buf.append(text)
            if len(buf) >= window:
                yield from self._flush(buf, batch_size, sort_by_length)
                buf = []
        if buf:
            yield from self._flush(buf, batch_size, sort_by_length)

    def _flush(self, buf: list[str], batch_size: int, sort_by_length: bool):
        order = range(len(buf))
        if sort_by_length:
            order = sorted(order, key=lambda i: len(buf[i]))
        out = np.zeros((len(buf), EMBED_DIM), dtype=np.float32)
        for start in range(0, len(buf), batch_size):
            idx = list(order)[start : start + batch_size]
            out[idx] = self.encode([buf[i] for i in idx])
        yield out
