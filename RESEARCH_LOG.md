# annlite research log

A running record of what was built, what was measured, and why each decision was
made. Newest entries at the bottom of each section. Numbers in this log are
reproducible with `make`; where a figure is quoted, the command that produced it is
quoted beside it.

---

## 0. Goal

Vector storage extensions for SQLite, usable from a browser over
[sql.js-httpvfs](https://github.com/phiresky/sql.js-httpvfs)-style HTTP range
requests:

* **Dense retrieval** over `all-MiniLM-L6-v2` embeddings, indexed with DiskANN.
* **Late-interaction retrieval** over multi-vector embeddings, indexed fast-plaid style.

The binding constraint throughout is not CPU but **network round-trips**. A browser
reading a remote SQLite file pays a full HTTP request per page range it has not
cached. An index that is fast on a local SSD can be unusable over a cell link if its
access pattern scatters across pages. Every measurement in this log therefore reports
round-trips and bytes fetched alongside wall-clock time.

---

## 1. Environment

Recorded 2026-09-19.

| Component | Version |
|---|---|
| Rust | 1.94.1 |
| Python | 3.11.15 |
| Node | 22.22.2 |
| onnxruntime | 1.30.0 |
| onnx | 1.23.0 |
| numpy | 2.4.6 |
| CPU / RAM | 4 cores / 15 GB |
| emscripten | not installed (needed for milestone 6) |

### 1.1 Egress policy

This environment reaches the network through a policy-enforcing proxy. Findings,
because they shaped several decisions below:

| Host | Result |
|---|---|
| `pypi.org`, `files.pythonhosted.org` | reachable |
| `index.crates.io` + crate downloads | reachable (`cargo fetch` works end to end) |
| `raw.githubusercontent.com`, `github.com`, `codeload.github.com` | reachable |
| `archive.ubuntu.com` | reachable |
| `storage.googleapis.com`, `www.googleapis.com` | reachable (Drive API still needs a key) |
| **`huggingface.co`, `cdn-lfs.huggingface.co`, `hf-mirror.com`** | **403 at the proxy** |
| **`drive.google.com`, `drive.usercontent.google.com`** | **403 at the proxy** |
| `download.pytorch.org`, `ollama.com` | 403 at the proxy |

Consequences:

* The Liquid AI 230M model could not be downloaded, so the LLM-written paragraph
  corpora are **out of scope** for this pass (confirmed with the project owner).
* Model weights were already committed to the repository, so inference is unaffected.
  Tokenizers, which normally come from Hugging Face, had to be sourced separately —
  see §3.

---

## 2. Identifying the committed models

The repository arrived with two ONNX files and no documentation of which was which.
Both were identified from their graph structure rather than their filenames, which
turn out to be uninformative.

### 2.1 `model_qint8_arm64.onnx` — dense encoder

```
inputs   input_ids, attention_mask, token_type_ids   [batch, seq]  int64
output   last_hidden_state                           [batch, seq, 384]  float32
params   22.6M    embeddings.word_embeddings.weight  [30522, 384]
layers   6        intermediate width 1536
```

Vocabulary 30522 and hidden width 384 over 6 layers is `all-MiniLM-L6-v2` exactly,
and 22.6M parameters matches the "23M parameter" description. The presence of
`token_type_ids` confirms an original-BERT architecture rather than a later variant.

**Confirmed empirically**, not just structurally. Mean-pooling `last_hidden_state`
over the attention mask and L2-normalising gives well-behaved cosine similarities:

| Pair | Cosine |
|---|---|
| "a man is playing a guitar on stage" / "someone performs music with a guitar at a concert" | **0.700** |
| "the recipe calls for two cups of flour…" / "bake the cake using flour, sugar and salt" | **0.593** |
| guitar / baking | 0.037 |
| guitar / quantum entanglement | −0.116 |
| baking / quantum entanglement | 0.073 |

Paraphrase pairs separate from unrelated pairs by roughly 0.6 cosine. The model,
the tokenizer and the pooling strategy are all correct.

### 2.2 `model_int8.onnx` — late-interaction encoder

```
inputs   input_ids, attention_mask                   [batch, seq]  int64
output   output                                      [batch, seq, 48]  float32
params   17.0M    bert.embeddings.tok_embeddings.weight  [50370, 256]
```

This is `lightonai/LateOn-Code-edge`. The diagnostic features:

* **Per-token output width 48, not a pooled single vector.** The graph emits one
  48-dimensional vector per input token, which is the defining shape of a
  late-interaction encoder — MaxSim needs per-token vectors to compare against.
* **`ReduceL2` then `Clip` at the output**, i.e. the per-token vectors leave the
  graph L2-normalised, so MaxSim reduces to a plain dot product.
* **`Cos`/`Sin`/`Neg`/`Range` nodes with no learned position embedding table**:
  rotary position embeddings, and `tok_embeddings` naming plus vocabulary 50370
  place it in the ModernBERT family.
* **No `token_type_ids`**, consistent with ModernBERT and unlike §2.1.

Note the filenames invert the intuitive reading: the *arm64* file is the dense model
and the plain *int8* file is the late-interaction one. Commit order matches the
project owner's description (dense committed first, late-interaction second).

---

## 3. Tokenizers

### 3.1 Dense: solved

`all-MiniLM-L6-v2` uses the stock `bert-base-uncased` WordPiece vocabulary. A copy
was retrieved from a reachable GitHub mirror and verified to be exactly **30522**
lines, matching the model's embedding table row count. Round-tripping
"a man is playing a guitar on stage" gives `[101, 1037, 2158, 2003, 2652, 1037,
2858, 2006, 2754, 102]` — `[CLS]`/`[SEP]` at 101/102 and `man` at 2158, all correct
for `bert-base-uncased`. The similarity table in §2.1 is the end-to-end proof.

### 3.2 Late interaction: open

`LateOn-Code-edge` needs its ModernBERT-family BPE tokenizer (vocabulary 50370),
which normally ships as `tokenizer.json` from Hugging Face — currently 403. The
project owner indicated a `tokenizer.json` is committed; as of this writing the
repository contains only `LICENSE`, `README.md` and the two `.onnx` files on
`trunk`, and no `tokenizer.json` appears anywhere in history. **Milestone 5 is
blocked on this file.** Everything else proceeds.

---

## 4. Corpora

### 4.1 Why not `shuf`

The brief asked for shuffling to be deterministic. `shuf` is the wrong tool for a
benchmark whose numbers are meant to be comparable across machines:
`shuf --random-source=FILE` is reproducible only within a single coreutils build,
and plain `shuf` also depends on locale collation of the input. Instead every random
choice flows through **ChaCha8**, seeded by `SHA-256("annlite/v1/" ‖ domain ‖ seed)`.
ChaCha8's output is fixed by the algorithm, so the same seed yields the same bytes on
any platform, architecture or library version. Fisher-Yates and uniform-integer
rejection sampling are written out explicitly rather than delegated to `rand`'s
helpers, so the permutation stays pinned even if `rand` changes its internals.

### 4.2 Vocabulary

`/usr/share/dict/words` (Debian `wamerican` 2020.12.07) holds 104,334 entries
including proper nouns and possessives. Normalising — lowercase, keep pure `[a-z]+`,
deduplicate, sort — yields **73,445 words**, mean length 8.09 characters. Sorting
before shuffling makes the result independent of the input file's own ordering.

### 4.3 Document construction

Generation proceeds in *rounds*. Each round independently shuffles the entire
vocabulary and cuts it into consecutive 50-word chunks, discarding the short tail;
73,445 / 50 gives **1,468 documents per round**. Rounds continue until the requested
document count is reached. Two properties follow, and the benchmarks depend on both:

* **Prefix property.** Document *i* depends only on *i* and the seed, never on the
  corpus size. The 100-document corpus is a byte-exact prefix of the 10k corpus,
  which is a byte-exact prefix of the 1M corpus. Verified on the real data:
  `head -100 docs-10k.txt` and `docs-100.txt` hash identically, as do
  `head -10000 docs-1m.txt` and `docs-10k.txt`. Scale curves therefore describe one
  growing collection rather than three unrelated samples.
* **Term frequency is exactly 1.** Chunks are cut from a permutation, so no word
  repeats within a document. This removes TF as a confounder from the FTS5 baseline:
  BM25 differences between documents come only from length (constant at 50) and IDF.

### 4.4 Generated corpora

| Corpus | Documents | Bytes | SHA-256 | Generation time |
|---|---:|---:|---|---:|
| `docs-100.txt` | 100 | 45,483 | `9a694c9b5f32d4fe…67065c` | <0.01 s |
| `docs-10k.txt` | 10,000 | 4,544,568 | `dfd1e23cbf75ecf3…222082` | 0.05 s |
| `docs-1m.txt` | 1,000,000 | 454,479,542 | `13df11e910ff7121…dfde80` | **5.43 s** |

Full digests are in `data/corpus/*.manifest.json`.

**Not committed.** At ~454 MB the 1M corpus would dominate the repository for no
benefit, since it regenerates byte-identically in 5.4 seconds. `make corpora`
rebuilds all three scales; manifests carry the digests to check against.

### 4.5 Query sets

A corpus of shuffled dictionary words has no topics, so the query design must not
smuggle in semantics the collection does not contain. Two families, 100 queries at
each of *k* ∈ {1, 2, 3, 5, 10} terms, 1,000 queries per scale:

* **`known_item`** — *k* words drawn from one known document. That document is a
  ground-truth answer that exists by construction, so the query measures whether a
  system can retrieve a specific document from a fragment of it. Lowering *k* from 10
  to 1 makes the task harder in a controlled way, because fewer terms means more
  documents share the query's full term set. Supports MRR and success@k.
* **`random`** — *k* words drawn from the vocabulary independently of any document.
  Typically no document contains all of them, so this measures ranking over partial
  matches, which is where dense and late-interaction systems can diverge from lexical
  matching.

Neither family needs human relevance judgments. **The gold ranking for any query is
exact brute-force scoring under the system's own scoring function**, so ANN recall@k
is always measured against an exact search rather than against an opinion.

Validated: all 500 known-item queries per scale reference in-range documents, all
query terms genuinely occur in their source document, and no query repeats a term.

---

## 5. Open items

* **Blocked:** `tokenizer.json` for `LateOn-Code-edge` (§3.2) — gates milestone 5.
* **Needed:** an emscripten toolchain for the WASM milestone.
* Out of scope this pass: LLM-written paragraph corpora (§1.1).
