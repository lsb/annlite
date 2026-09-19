"""Multi-vector embeddings from `lightonai/LateOn-Code-edge` (`model_int8.onnx`).

A late-interaction encoder emits one vector per token rather than one per document,
and the vectors leave this graph already L2-normalised (`ReduceL2` then `Clip` at the
output, per RESEARCH_LOG.md section 2.2), so MaxSim reduces to plain dot products
with no normalisation step here.

Two conventions matter and both were settled by measurement rather than assumption
(RESEARCH_LOG.md section 14.1):

* **Padding must be masked out.** The tokenizer names `[MASK]` as its pad token,
  which looks like ColBERT's query-augmentation trick, where a query is padded with
  `[MASK]` and those positions are *attended to* as learned query expansion. Doing
  that here drops top-1 accuracy on a code-retrieval probe from 5/5 to 2/5: this
  model was not trained with query augmentation, and attending to padding adds
  meaningless vectors that every query token can max against. Padded positions are
  therefore excluded from both the attention mask and the returned vectors.
* **Special tokens are kept.** `[CLS]` and `[SEP]` come from the tokenizer's own
  template. Dropping them measured identically (5/5 either way), so they stay, which
  keeps the encoder faithful to the template rather than second-guessing it.

The default `max_length` is the tokenizer's own declared limit rather than a round
number. An earlier version capped at 512, which silently truncated 91 of the 3,366
code documents; the model is RoPE-based and has no position-table ceiling, and was
verified to accept a 740-token input unchanged. Truncating documents costs recall in
a way that looks like model weakness rather than like a configuration mistake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

LATE_DIM = 48
PAD_ID = 50284  # [MASK], per tokenizer.json's padding config


class LateEncoder:
    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        max_length: int = 2047,
        threads: int | None = None,
    ):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.max_length = max_length
        expected = {"input_ids", "attention_mask"}
        actual = {i.name for i in self.session.get_inputs()}
        if actual != expected:
            raise ValueError(f"unexpected model inputs {sorted(actual)}; expected {sorted(expected)}")
        vocab = self.tokenizer.get_vocab_size(with_added_tokens=True)
        if vocab != 50370:
            raise ValueError(f"tokenizer has {vocab} tokens; the checkpoint expects 50370")

    def encode(self, texts: Sequence[str]) -> list[np.ndarray]:
        """One `(tokens_i, 48)` float32 array per input, padding excluded."""
        if not texts:
            return []
        encoded = [self.tokenizer.encode(t).ids[: self.max_length] for t in texts]
        width = max(len(e) for e in encoded)
        ids = np.full((len(encoded), width), PAD_ID, dtype=np.int64)
        mask = np.zeros((len(encoded), width), dtype=np.int64)
        for i, e in enumerate(encoded):
            ids[i, : len(e)] = e
            mask[i, : len(e)] = 1
        out = self.session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        # Return only real token positions: a padded vector is not a token of the
        # document and must never be available for a query token to match against.
        return [out[i, : len(e)].astype(np.float32) for i, e in enumerate(encoded)]

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def maxsim(query: np.ndarray, doc: np.ndarray) -> float:
    """Sum over query tokens of the best matching document token.

    Note the asymmetry: the sum is over *query* tokens, so the score scales with
    query length but not with document length -- except that a longer document offers
    more tokens to take a maximum over, which biases MaxSim toward long documents.
    That bias is real and is reported rather than corrected for.
    """
    return float((query @ doc.T).max(axis=1).sum())
