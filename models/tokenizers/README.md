# Tokenizer assets

Committed rather than fetched at runtime. `huggingface.co` is unreachable from the
build environment (see RESEARCH_LOG.md §1.1), and a retrieval benchmark whose
tokenizer can silently change is not reproducible in any case.

## `bert-base-uncased-vocab.txt`

WordPiece vocabulary for `all-MiniLM-L6-v2` (`model_qint8_arm64.onnx`).

* 30,522 entries, matching the model's `embeddings.word_embeddings.weight` rows
  exactly — the check that confirms vocabulary and checkpoint belong together.
* SHA-256 `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`
  (recompute with `sha256sum`; the manifest in this directory is authoritative).
* `[PAD]`=0, `[UNK]`=100, `[CLS]`=101, `[SEP]`=102, `[MASK]`=103.

## Missing: `LateOn-Code-edge`

`model_int8.onnx` needs a ModernBERT-family BPE tokenizer with 50,370 entries.
Not yet available — see RESEARCH_LOG.md §3.2.
