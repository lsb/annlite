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

## 5. Dense embedding pipeline

### 5.1 Tokenizer

`tools/embed/tokenization.py` implements BERT WordPiece directly rather than pulling
in `transformers`. The browser target needs this same algorithm in Rust compiled to
WASM, and one readable reference implementation can be pinned against the other with
shared test vectors. `tools/tests/test_tokenization.py` holds those vectors: the
reference ids for `"a man is playing a guitar on stage"` are
`[101, 1037, 2158, 2003, 2652, 1037, 2858, 2006, 2754, 102]`, and the suite also
covers accent stripping, punctuation splitting, all-or-nothing `[UNK]` handling, and
truncation. 10/10 pass.

The vocabulary is **committed** at `models/tokenizers/bert-base-uncased-vocab.txt`
(SHA-256 `07eced37…2038a3`, 30,522 entries). Fetching it at runtime is not an option
with Hugging Face blocked, and a retrieval benchmark whose tokenizer can silently
change is not reproducible anyway.

### 5.2 Pooling

The ONNX graph emits `last_hidden_state` only. Two steps turn it into a sentence
embedding and both matter:

* **Masked** mean pooling. Averaging over padding positions pulls the embedding
  toward whatever the model emits for `[PAD]`, and the size of that distortion
  depends on how much padding the batch happens to carry — so an unmasked mean makes
  a document's vector depend on its batchmates.
* L2 normalisation, which makes the inner product equal cosine similarity and lets
  every downstream index use plain dot products.

### 5.3 Throughput

Documents tokenize to ~100–120 WordPiece tokens (50 dictionary words, many of them
rare and multi-piece). Measured on 4 cores:

| Configuration | Throughput |
|---|---:|
| 1 process × 4 intra-op threads | ~53–131 docs/s (high variance under load) |
| 1 process × 2 intra-op threads | ~79 docs/s |
| **4 processes × 1 thread** | **158 docs/s** |

A 6-layer model at batch 16 does not scale well across onnxruntime's intra-op
threads — the per-operator work is too small to amortise synchronisation — whereas
independent processes on disjoint shards scale nearly linearly. Workers write
directly into their own slice of a shared memory-mapped output file, so there is no
concatenation pass and peak memory is one batch per worker regardless of corpus size.

At 158 docs/s the 1M corpus takes ~1.8 h; the 10k corpus took **63 s**.

Embeddings are stored as headerless little-endian float32 with a JSON sidecar rather
than `.npy`, because Rust reads them during index construction and the browser reads
them at query time. A headerless matrix memory-maps from any language without a
parser, and a byte range maps to rows by arithmetic alone — which is the whole point
when the reader is fetching ranges over HTTP.

### 5.4 Finding: dense retrieval is weak on random-word documents

Exact (brute-force) cosine kNN over the 10k corpus, 1,000 queries. This is the
*ceiling* for any dense ANN index on this corpus — an approximate index can only lose
ground relative to it.

| Query kind | k terms | success@1 | success@10 | success@100 | MRR |
|---|---:|---:|---:|---:|---:|
| known_item | 1 | 0.000 | 0.060 | 0.140 | 0.023 |
| known_item | 2 | 0.040 | 0.180 | 0.310 | 0.086 |
| known_item | 3 | 0.070 | 0.160 | 0.350 | 0.101 |
| known_item | 5 | 0.080 | 0.310 | 0.590 | 0.164 |
| known_item | 10 | 0.360 | 0.620 | 0.820 | 0.435 |

Mean top-1 cosine is 0.448 against a mean median of 0.218 — the score barely
discriminates. This is the expected result and not a bug: `all-MiniLM-L6-v2` was
trained on natural sentences, and mean-pooling 50 mutually unrelated dictionary words
produces a vector near the centroid of the embedding space. Fifty random words carry
no topic for a topic model to encode.

**Implication for the results matrix.** The random-word corpus remains a perfectly
good benchmark for *index mechanics* — recall against exact search, page access
patterns, round-trips, index size, build time — because those are measured against
the same encoder's own exact ranking and so are unaffected by the encoder's semantic
quality. But it cannot support an honest *end-to-end retrieval quality* comparison
between lexical and dense retrieval: FTS5 matches query terms literally and will win
by a wide margin, for reasons that say nothing about either system's behaviour on
real text. Reporting that number as "dense loses to FTS5" would be misleading.

Measuring quality therefore needs natural-language text. The LLM-written corpus that
would have supplied it is out of scope (§1.1), so a substitute is needed; reachable
options are recorded in §6.

---

## 6. Reachable corpus sources

Re-probed after the §5.4 finding, since a natural-language corpus is now needed.

| Source | Reachable | Notes |
|---|---|---|
| NLTK corpora (`raw.githubusercontent.com/nltk/nltk_data`) | **yes**, incl. byte ranges | brown, gutenberg, reuters, inaugural, europarl among ~100 packages |
| Reuters-21578 (via NLTK) | **yes** | ships topic labels — genuine relevance judgments, not synthesised ones |
| Python standard library on disk | yes | 672 `.py` files |
| Rust crate sources in the cargo registry | yes | 1,968 `.rs` files |
| `gutenberg.org`, `dumps.wikimedia.org` | no | 403 at the proxy |

Reuters-21578 is the strongest candidate: it is a real IR benchmark whose documents
are natural English and whose topic labels give relevance judgments that were not
generated by the system under test. The local code corpora are the natural fit for
`LateOn-Code-edge`, which is a code model.

---

## 7. Product quantization

A 384-dimensional float32 embedding costs 1,536 bytes; a million of them is 1.5 GB,
which is not something a browser downloads to answer one query. PQ splits each vector
into `m` contiguous subvectors, clusters each subspace into 256 centroids, and stores
one byte per subspace. At **m = 64** — six dimensions per subquantizer — a document
costs **64 bytes**, a 24x reduction, and a million fit in 64 MB.

### 7.1 Design choices

* **Asymmetric distance computation.** The query stays in full precision; only
  documents are quantized. Quantizing the query too would discard precision on the
  one vector we have exactly, for no saving.
* **Inner product decomposes over the subspace partition**, so a per-query table of
  `<q_subspace, centroid>` values (64 x 256 floats, 64 KB) can be built once and each
  document then scores in 64 lookups and 64 additions — no multiplications, and no
  access to the original vectors at all.
* **k-means minimises squared Euclidean distance** even though scoring is by inner
  product. That is the correct objective: ADC error is bounded by how far a subvector
  sits from its centroid, which is exactly what k-means minimises.
* **k-means++ seeding, empty clusters re-seeded** onto the worst-quantized point. In
  PQ an empty cluster is a permanently wasted code point, not merely a slow start.
* **Seeded, deterministic training**, so a rebuild yields byte-identical codes and
  recall figures stay comparable across runs.

### 7.2 Measured on real MiniLM embeddings

10k documents, 200 queries, exact inner product as ground truth. `R@10/P` is the
fraction of the exact top 10 appearing in the PQ top `P`.

| m | bytes/doc | compression | train (s) | R@10/10 | R@10/50 | R@10/100 | R@100/100 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 16 | 96x | 17.3 | 0.258 | 0.525 | 0.674 | 0.346 |
| 32 | 32 | 48x | 15.8 | 0.401 | 0.747 | 0.870 | 0.493 |
| **64** | **64** | **24x** | 17.0 | **0.612** | **0.957** | **0.992** | 0.679 |
| 96 | 96 | 16x | 18.0 | 0.747 | 0.995 | 1.000 | 0.789 |

Mean signed ADC score error at m = 64 is **−0.00066** over 5,420 query-document
pairs, i.e. the approximation is essentially unbiased rather than systematically
optimistic or pessimistic. That matters because a biased score would mean thresholds
tuned on one corpus would not transfer to another.

### 7.3 Finding: PQ is a candidate generator, not a ranker

The two columns that matter are `R@10/10` = 0.612 and `R@10/100` = 0.992. At 64 bytes
per document the true top 10 is *almost always present* in the PQ top 100, but PQ
gets the ordering within that pool right only about 60% of the time.

This settles the dense architecture: **retrieve deep with PQ, then rerank a shallow
pool with exact vectors.** Trusting the PQ ordering directly would discard ~39% of
the correct top-10 results for no reason, while reranking a 100-candidate pool
recovers essentially all of them.

It also creates the central tension of this project. Reranking 100 candidates needs
100 full vectors — 150 KB, and scattered across the file rather than contiguous. Over
a network that is the dominant cost of the query, far outweighing the graph traversal
that found the candidates. Candidate-pool depth is therefore not a quality knob but a
**bandwidth knob**, and the results matrix must report recall against bytes fetched
rather than recall alone. Reranking strategies to measure:

1. Exact rerank — fetch full float32 vectors for the pool (highest quality, ~1.5 KB
   per candidate).
2. Two-level PQ — a coarse code for traversal and a finer code for reranking, keeping
   everything in the compressed domain.
3. No rerank — accept PQ ordering (cheapest, loses ~39% of top-10).

The 24x compression is what makes a million documents addressable at all; the
reranking policy is what decides whether the result is any good.

---

## 8. Network simulation (`annlite-netsim`)

An HTTP server that serves byte ranges under a simulated link, so "how many pages
does this query touch" can be converted into "how long does this query take on a
phone". Built to interoperate with `sql.js-httpvfs`: correct 206/416 semantics,
`Content-Range`, suffix and open-ended ranges, CORS with `Content-Range` exposed
(without which the browser cannot read it and httpvfs breaks), and path-traversal
refusal both lexically and after `canonicalize`.

### 8.1 The delay model

`delay = Lognormal(median = rtt, sigma) + [with probability p] Exponential(mean = spike)`.

Lognormal because measured cell RTT is right-skewed — bounded below by the radio's
scheduling floor, unbounded above. The separate spike term exists because a pure
lognormal produces no multi-hundred-millisecond stalls, and those are precisely what
destroys an index needing a dozen *dependent* round-trips; the causes are physically
distinct (scheduling jitter versus retransmission, handover, idle-state transition)
so the parameters are distinct too.

Reproducibility works the same way as the corpus generator: each request derives a
ChaCha8 stream from `SHA-256("annlite/netsim/v1/" ‖ domain ‖ seed ‖ ordinal)`, so a
replay of the same request sequence draws the same delays, on any machine. Two
separate server processes on the same seed produced byte-identical delay sequences.
Ordinals are assigned single-threaded at accept; a worker pool then does the
sleeping, so parallel fetches stay parallel without making the draws depend on
scheduling.

`/__netsim/reset` resets the ordinal, so bracketing each query makes every query see
the *same* delay sequence — two indexes then differ by access pattern rather than by
which one happened to draw a spike.

### 8.2 Profiles

| profile | RTT ms | spike p | Mbit/s | basis |
|---|---:|---:|---:|---|
| `ideal` | 0 | – | inf | control: keeps "40 round-trips" separable from "40 round-trips cost 2.8 s" |
| `wifi` | 15 | 0.005 | 50 | 802.11 plus a short hop to a well-peered edge |
| `5g` | 35 | 0.015 | 100 | NR mid-band sub-frame scheduling |
| `lte` | 70 | 0.03 | 15 | LTE RAN adds ~40-60 ms over the wired path |
| `leo` | 45 | 0.05 | 80 | ~550 km is only ~4 ms of propagation; RTT is ground network, spikes are hand-offs |
| `3g` | 200 | 0.06 | 1.6 | rate from DevTools "Fast 3G"; RTT from measured HSPA+ rather than DevTools' pessimistic 562 ms |
| `slow-3g` | 2000 | 0.08 | 0.4 | DevTools "Slow 3G" verbatim |
| `satellite` | 600 | 0.04 | 20 | physics: 35,786 km is ~477 ms at c, plus terrestrial tail |

### 8.3 Measured: latency dwarfs transfer at every scale

32 sequential 4 KiB page fetches — what a page-faulting SQLite reader does — over
128 KiB of payload:

| profile | wall | sum latency | sum transfer |
|---|---:|---:|---:|
| ideal | 225 ms | 0 | 0 |
| wifi | 836 ms | 499 ms | 21 ms |
| lte | 2,903 ms | 2,531 ms | 75 ms |
| 3g | 10,849 ms | 9,843 ms | 713 ms |
| satellite | 20,520 ms | 20,125 ms | 55 ms |

Satellite spends 20.1 s waiting and 55 ms transferring. **Bandwidth is almost
irrelevant; round-trip count is everything.** Every index decision in this project
should be read through that table. Reducing bytes fetched is worth little; reducing
*the number of dependent fetches* is worth almost everything.

---

## 9. Baseline: FTS5

Measured on this machine, SQLite 3.46.0, page size 4096, single transaction,
`journal_mode=OFF`. Every figure below comes from a run that completed; nothing is
extrapolated.

### 9.1 Build and size

| Scale | Build | Throughput | DB size | Pages | Bytes/doc |
|---|---:|---:|---:|---:|---:|
| 100 | 0.003 s | — | 139 KB | 34 | 1,393 |
| 10k | 0.39 s | 25,473 docs/s | 8.88 MB | 2,167 | 888 |
| **1M** | **56.4 s** | **17,741 docs/s** | **730.6 MB** | 178,371 | 731 |

Throughput degrades ~30% from 10k to 1M as segment merges kick in. At 1M the file
splits `docs_content` 70% / inverted index 28% — **the stored documents, not the
index, are the bulk.**

**`optimize` makes the file bigger.** It merges segments but leaves the old pages on
the freelist: at 10k the file grew 8.88 MB → 11.31 MB (+27%) while the index itself
got tighter. `VACUUM` then brought it to 7.66 MB, 14% *below* the as-built size.
Serving a post-`optimize` file over a CDN would ship 32% dead pages. The deployment
recipe is `optimize` **then** `VACUUM`, not `optimize` alone.

### 9.2 Query latency (warm cache, median, post-vacuum)

| Scale | k=1 | k=2 | k=3 | k=5 | k=10 |
|---|---:|---:|---:|---:|---:|
| 100 | 17 µs | 22 | 28 | 39 | 67 |
| 10k | 23 µs | 35 | 47 | 73 | 141 |
| 1M | 793 µs | 1,509 | 2,247 | 3,744 | **7,934** |

Linear in both term count and corpus size, and query *kind* barely matters, because
this corpus gives every term nearly the same document frequency. `LIMIT 100` prunes
nothing: `ORDER BY bm25()` must score every match first.

### 9.3 Page access — why FTS5 cannot be served over HTTP at 1M

Distinct 4 KiB pages per query, cold pager cache. Measured with a pass-through VFS
recording every `xRead`, cross-checked against `SQLITE_DBSTATUS_CACHE_MISS` — the two
agreed on **9,000 of 9,000 queries** across all scales and phases.

| Scale | median pages | p95 | mean KiB read |
|---|---:|---:|---:|
| 100 | 7 | 13 | 30.8 |
| 10k | 26 | 41 | 108.4 |
| **1M** | **1,537** | **2,402** | **6,230** |

Median pages by term count at 1M: **689 / 1,179 / 1,538 / 1,986 / 2,395**.

A single-term query at 1M touches **689 distinct pages and reads 2.8 MB**. At the
`lte` profile's 70 ms RTT and one request per page that is **48 seconds**. This is
the result the baseline existed to establish: **FTS5 as shipped is not viable over
HTTP ranges at a million documents.**

### 9.4 Finding: the ranking function scatters the access pattern, not the index

Re-running the same queries without `ORDER BY bm25()` (and `LIMIT -1`, so both
variants still enumerate every candidate):

| Scale | ranked, mean pages | unranked, mean pages | ratio |
|---|---:|---:|---:|
| 100 | 7.7 | 6.4 | 1.2x |
| 10k | 27.1 | 10.9 | 2.5x |
| **1M** | **1,557.6** | **19.8** | **79x** |

*Finding* the candidates at 1M costs 20 pages. *Scoring* them costs 1,538, because
`%_docsize` is a separate rowid-keyed table and FTS5 does one random row lookup per
matching document — a 10-term query touches 2,352 of that table's 2,454 pages.

This is the single most transferable lesson for the ANN designs: **co-locate whatever
the scorer needs with the postings, or make the scorer need nothing per candidate.**
PQ's ADC already satisfies the second form — a document's score needs only its own
64 bytes and a table held in memory.

### 9.5 Finding: `VACUUM` converts scattered pages into contiguous runs

At 1M, k=10: 2,396 pages in **2,380 runs** after `optimize`, and 2,395 pages in
**133 runs** after `VACUUM` — the same pages, one apart, in a twentieth of the
requests. A client that coalesces adjacent pages goes from ~2,380 requests
to ~133 for an identical query on an identical-size file, at zero latency cost. Page
*count* is unchanged; page *adjacency* is transformed. Given §8.3, that is a ~18x
reduction in the only quantity that matters.

### 9.6 Caveat the ANN comparison must carry: FTS5 at k=1 is not ranking

At 1M, `known_item` k=1 scores success@1 = 0.000 and success@100 = 0.130. That is not
a ranking failure, it is a **tie**. Every vocabulary term has the same document
frequency and every document the same length, so all ~681 documents containing a
single query term receive *identical* BM25 scores and the gold document's position
among them is arbitrary. success@100 = 0.130 is exactly 100/681 in expectation.

**An ANN system must not be credited for "beating FTS5 at k=1" — it would be beating
a coin flip.** A fair lexical comparison needs tie-aware scoring or a corpus with
varied term frequencies. Overall known-item figures: 100 → success@1 1.000; 10k →
0.826, MRR@100 0.8755; 1M → 0.760, MRR@100 0.7796.

Note also that the measured queries select `rowid` and score only, so `docs_content`
— 70% of the file — is never read. That is the right comparison against an ANN index
that also returns ids, but a snippet-displaying application pays roughly one extra
page per displayed result.

---

## 10. Dense index 1: HNSW

### 10.1 Implementation notes

The graph is built over **exact** vectors even when search scores with PQ codes:
graph quality depends on getting neighbour relationships right, and a bad edge is
permanent while a bad query-time score costs one comparison.

Visit tracking uses a generation-stamped buffer rather than a fresh `vec![false; n]`
per layer search. At a million documents the allocation and zeroing alone would make
construction quadratic; bumping a counter makes the reset free.

### 10.2 Validation against a reference implementation

Recall looked low, so before tuning anything the implementation was checked against
`hnswlib` on identical data, identical parameters, identical ground truth
(10k documents, 200 held-out queries, exact inner product):

| ef | this implementation | hnswlib | ms/query (this) | ms/query (hnswlib) |
|---:|---:|---:|---:|---:|
| 10 | 0.359 | 0.284 | 0.252 | 0.011 |
| 50 | 0.678 | 0.613 | 0.756 | 0.036 |
| 100 | 0.815 | 0.749 | 1.274 | 0.060 |
| 200 | **0.912** | **0.884** | 2.110 | 0.110 |

*(M=16, efConstruction=200 for both.)*

**Recall is not the problem** — this implementation matches and slightly exceeds the
reference at every `ef`. Speed is: ~19x slower per query and ~65x slower to build
(71 s versus 1.1 s), which is the expected cost of scalar Rust against heavily
SIMD-optimised, multi-threaded C++. For a research harness measuring *round-trips*
that is an acceptable trade; it would not be acceptable in the shipped WASM.

### 10.3 Finding: the corpus is intrinsically hard for graph ANN

A diagnostic over three datasets, all 10k x 384, all with **held-out** queries.
(The first version of this diagnostic used corpus members as queries and produced
recall 1.000 everywhere — a query that is itself a document is found by greedy
descent almost for free. The numbers below are after fixing that.)

| dataset | 10th-NN sim | mean sim | contrast | recall@10 at ef=10 | at ef=200 |
|---|---:|---:|---:|---:|---:|
| synthetic, tight clusters | 0.979 | 0.010 | 8.88 | 0.383 | 0.805 |
| synthetic, uniform sphere | 0.160 | 0.000 | 3.07 | 0.137 | 0.825 |
| real MiniLM word-bag embeddings | 0.647 | **0.488** | 2.69 | 0.359 | **0.912** |

The real embeddings are the *easiest* of the three for HNSW, yet still need `ef=200`
— visiting roughly 2% of a 10,000-document corpus — to reach 0.91 recall, where a
well-conditioned benchmark set reaches 0.95 at `ef=50`.

The reason is visible in the `mean sim` column: **0.488**. MiniLM embeddings occupy a
narrow cone rather than the whole sphere, so every document is somewhat similar to
every other and the gap between "nearest" and "typical" is small relative to the
spread. Graph descent has weak gradient to follow.

**Consequence for the network target.** `ef=200` at 10k means ~200 distance
evaluations against vectors that, over HTTP, live in different pages. Per §8.3 that
is not a CPU cost but a round-trip cost, and on `lte` it is minutes. Two mitigations
follow directly and are what the remaining milestones must measure:

1. **Keep candidates in the compressed domain.** PQ codes at 64 bytes put 64
   documents in a 4 KiB page, so a 200-candidate traversal can touch a handful of
   pages instead of 200 — provided the graph's neighbours are laid out together,
   which HNSW does not do.
2. **Lay the graph out for locality.** HNSW assigns node ids in insertion order and
   its neighbour lists point anywhere, so consecutive hops land on unrelated pages.
   This is what Vamana/DiskANN is for, and §9.5 already showed the size of the prize:
   the same pages in 133 runs instead of 2,380.

---

## 11. Dense index 2: Vamana, and what node ordering buys

### 11.1 Why Vamana rather than HNSW for this target

Vamana is a single flat graph of fixed out-degree, and both properties matter more
here than any recall difference:

* **Flat** means one kind of record, so node `i` sits at byte `i * record_size` and
  the page holding it follows from its id by arithmetic. HNSW's per-node level makes
  records variable-length and forces an index of indexes.
* **Fixed degree** means a node plus its entire adjacency fits a known budget, so a
  record can be sized to divide evenly into a page.

Together they make node ids the thing that decides pages — and ids are ours to
choose. That is the lever this section measures.

### 11.2 Finding: `alpha` works the opposite way round from the obvious reading

The pruning rule keeps a candidate `v` unless some already-kept neighbour `p*`
satisfies `alpha * d(p*, v) <= d(p, v)`. Raising `alpha` makes that discard test
*harder* to pass, so **fewer** candidates are occluded, the graph grows **denser**,
and its edges get **shorter** — not longer, as the "alpha keeps long-range edges"
summary suggests.

Measured on 4,000 clustered vectors, R=32:

| alpha | edges | mean edge distance | recall@10 (L=64) |
|---:|---:|---:|---:|
| 1.0 | 41,553 | 0.151 | 0.931 |
| **1.1** | 74,667 | 0.159 | **0.947** |
| 1.2 | 106,537 | 0.125 | 0.928 |
| 1.4 | 127,290 | 0.059 | 0.595 |
| 1.6 | 125,847 | 0.055 | **0.059** |
| 2.0 | 125,512 | 0.055 | 0.059 |

Past about 1.4 every node saturates at full degree with its nearest neighbours and
the graph degenerates into an approximate kNN graph — exactly the badly-navigable
structure that diversified pruning exists to prevent. Recall collapses to 0.059,
which is consistent with greedy search never escaping the medoid's own cluster
(1/40 clusters ≈ 0.025 expected by chance).

Scaling the *other* side of the inequality (`d(p*, v) <= alpha * d(p, v)`) was
implemented and measured too: it prunes ever more aggressively, stripping the graph
to mean degree 1.3 and recall 0.005. The original reading is correct; the default is
now **1.1**, the best measured value. Both halves are pinned by a test.

### 11.3 The storage format, and the lesson it inherits from FTS5

A node's record holds its PQ code *beside* its adjacency:

```text
record := pq_code[m]  degree:u16  neighbours[r]:u32
```

This is section 9.4's lesson applied directly. FTS5 was slow over the network
because scoring needed a per-document lookup in a different table; here a single
page read yields both the score of every node on that page and the ids to hop to
next. Nothing is looked up twice, and traversal never touches the full vectors at
all. Storage is ordinary SQLite tables — no virtual table, no loadable extension —
so a stock WASM build can read it.

At m=64, r=32 the record is **194 bytes** and SQLite packs **20 per 4 KiB page**
(measured via `dbstat`, against an arithmetic estimate of 21).

### 11.4 Measured: node ordering cuts pages by a third

10,000 documents, 200 queries, R=32, alpha=1.1, m=64. Graph, codes, queries and
search parameters are identical across the three orderings, so every difference is
attributable to node numbering alone. Mean distinct pages touched per query:

| L | beam | Identity | BFS | Cluster |
|---:|---:|---:|---:|---:|
| 32 | 1 | 403.1 | **271.8** | 337.3 |
| 32 | 4 | 420.5 | **280.4** | 355.7 |
| 32 | 16 | 453.4 | **299.3** | 400.2 |
| 64 | 1 | 455.0 | **339.9** | 402.2 |
| 128 | 1 | 473.9 | **403.2** | 448.1 |

BFS ordering over the graph is the best of the three, cutting pages by up to **33%**
for byte-identical results. Insertion order matches the random-access prediction
almost exactly: 500 pages and 858 reads gives an expected 410 distinct pages against
403 measured, confirming that unordered ids are simply random with respect to
locality.

But the ceiling is low. Even BFS touches 272 of 500 pages — 54% of the table — when
the query read only 858 of 10,000 records. Reordering cannot fix that, because the
problem is not *where* the records are but *how many* are read.

### 11.5 Finding: resident PQ codes cut pages by up to 18x at identical recall

The traversal above reads a record for **every node it scores**, because a node's
score lives in its record. Scoring the ~27 neighbours of each expanded node is what
turns 8 hops into 1,370 record reads.

The alternative is to make the codes resident: download every PQ code once as one
contiguous blob, and then read a record only for a node the search actually
*expands*. Same graph, same parameters, same results — only the timing of the reads
changes. Measured over the same 200 queries:

| L | beam | recall@10 | nodes read (disk) | nodes read (resident) | pages (disk) | pages (resident) | ms (disk) | ms (resident) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 0.383 | 858.0 | **37.9** | 403.1 | **36.6** | 5.94 | **0.58** |
| 32 | 4 | 0.395 | 978.0 | **44.9** | 420.5 | **43.0** | 6.78 | **0.63** |
| 32 | 16 | 0.414 | 1370.5 | **73.5** | 453.4 | **68.4** | 9.45 | **0.90** |
| 128 | 1 | 0.520 | 2197.9 | **130.7** | 473.9 | **114.7** | 15.63 | **1.80** |
| 128 | 16 | 0.523 | 2429.1 | **156.7** | 475.0 | **134.2** | 17.29 | **1.78** |
| 128 | 16 + rerank | **0.732** | 2429.1 | **156.7** | 475.0 | **134.2** | 18.71 | **3.07** |

Recall is **identical to three decimals** in every row — as it must be, since the
same nodes are scored either way. Pages fall by 6.6x to 18x and local wall-clock by
around 10x. The fixed cost is the blob: 10,000 x 64 bytes = **640 KB**, fetched once
per session as a single sequential range.

Under the `lte` profile (70 ms RTT, 15 Mbit/s, six parallel connections), at L=32
beam=16:

* on-disk: 8 hops x ceil(57 requests per hop / 6) waves x 70 ms + 0.99 s transfer ≈ **6.6 s per query**
* resident: 0.41 s preload, then 8 x ceil(8.5/6) x 70 ms + 0.15 s ≈ **1.27 s per query**

Resident wins from the very first query at this scale. It will not at every scale:
the blob is `n * m` bytes, so at a million documents it is 64 MB — about 34 s on
`lte` — and the crossover moves out to however many queries amortise that. Halving
`m` to 32 bytes halves the preload and, per section 7.2, costs recall
(R@10/100 drops 0.992 to 0.870). **The choice is not a property of the index but of
the session**, which is why `tools/analyze/netcost.py` reports a crossover rather
than a winner.

### 11.6 Caveat on comparing these numbers to section 10

The Vamana figures above score candidates with **64-byte PQ codes**, while the HNSW
figures in section 10 score with **full float32 vectors**. The gap between Vamana's
0.52 recall at L=128 and HNSW's 0.82 at ef=128 is therefore mostly the quantizer,
not the graph. They are not a like-for-like comparison and should not be read as one.

---

## 12. Late interaction

Implemented following PLAID's staging, and testable on the index mechanics even
though the encoder's tokenizer is still missing (section 3.2).

MaxSim scores a query against a document by letting every query token take its best
match among the document's tokens and summing those maxima. It is strictly more
expressive than a single dot product — a document can match one part of a query
strongly without diluting that evidence into an average — and ruinously expensive
stored naively: a 50-word document is ~120 tokens, at 48 float32 dimensions that is
23 KB per document, so a million documents would be 23 GB.

The compression exploits the fact that token vectors are highly redundant across a
corpus. Cluster them all; a token becomes a centroid id plus a residual. Retrieval
then runs in stages of increasing cost and selectivity:

1. **Candidate generation** — each query token probes its nearest centroids and
   collects documents from an inverted list. No document data is read.
2. **Centroid interaction** — rank candidates by MaxSim over centroids alone, using
   only the resident centroid table and the documents' centroid ids.
3. **Full MaxSim** — decompress and score exactly, for the surviving few.

The staging is the same shape as the dense pipeline and for the same reason: stages
1 and 2 touch only data that is resident or sequential, and only stage 3 reads
scattered per-document bytes.

Verified on synthetic multi-vector corpora with known structure: compression exceeds
10x, the centroid stage recalls over 75% of the exact top 10 into its candidate pool,
exact reranking puts the true best document first over 80% of the time, and widening
the probe never shrinks the pool. **Retrieval quality on real text remains blocked on
the `LateOn-Code-edge` tokenizer.**

---

## 13. The browser demo, and what the client does to page locality

`web/demo` runs in a real browser against a SQLite file it holds no copy of. It is
driven end to end under Playwright, and all four combinations of transport and code
residency return **identical** result ids, which is the correctness signal worth
having across three implementations of the same search (native, WASM synchronous,
WASM resumable).

Retrieval visibly works even on this corpus: "guitar music concert stage" returns
documents containing *bassists*, *concertmaster*, *saxophones*, *amplifiers*,
*auditoriums* and *guitars* — the encoder is finding topical structure inside bags
of random dictionary words.

### 13.1 The traversal API is a state machine, not a callback

`sql.js-httpvfs` is asynchronous from the page, so the synchronous callback the
native build uses only works inside a Web Worker. `SearchSession` instead hands out
the ids it needs, waits to be given the bytes, and steps forward. Three things
follow, and they are why it is the API the demo uses:

* It works with any asynchronous source without the traversal knowing.
* Each request is a **batch**, which a browser issues in parallel up to its
  per-origin limit, so one round costs about one round-trip however wide the frontier.
* The round-trip count stops being an estimate. `session.hops` is measured.

### 13.2 Finding: the httpvfs client fetches the whole database anyway

`sql.js-httpvfs` treats `requestChunkSize` as a **floor, not a cap**. Its speculative
read heads double their request size — 4 KiB, 8, 16, 32, 64, 128, 256, 512 KiB,
1 MiB, 2 MiB — until they have swallowed the file. Observed request sizes for a
single query against the 5.9 MB demo database, from the simulator's log:

```
4096  8192  16384  32768  65536  131072  262144  524288  1048576  2097152  1671168
```

The first query pulls **about 5.2 MB in six requests**: effectively the entire
database. `maxReadHeads` and `maxReadSpeed` exist but live inside its lazy file and
are not reachable through the public `createDbWorker` config — passing them was
tried and measured to change nothing.

**Any amount of page-locality work in the index is invisible to such a client**,
because it has already fetched everything before locality could matter. That is a
finding about the client, not the index, and it would have silently invalidated the
section 11.4 ordering results had they been measured through this path.

### 13.3 Bounded ranges: 56x fewer bytes, 69x more requests

So the demo also offers the transport the fixed-size record format was designed for.
Record `i` begins at byte `i * record_bytes`, so a client can compute the offset and
issue a bounded `Range` request, coalescing adjacent records into single requests.

Traversal measured separately from result display, because the latter always goes
through SQLite here and its read-ahead would otherwise swamp the comparison:

| transport | codes | round-trips | records | traversal requests | traversal bytes |
|---|---|---:|---:|---:|---:|
| SQLite httpvfs | on-disk | 39 | 863 | 2 | 528.0 KB |
| SQLite httpvfs | resident | 20 | 72 | 1 | 512.0 KB |
| bounded Range | on-disk | 39 | 863 | 726 | 109.6 KB |
| **bounded Range** | **resident** | **20** | **72** | **69** | **9.1 KB** |

Bounded ranges with resident codes move **9.1 KB per query against SQLite's 512 KB,
a 56x reduction** — in 69 requests rather than one.

### 13.4 This does not resolve in favour of either side at this scale

Applying the cost model: on the `lte` profile (70 ms RTT, 15 Mbit/s, six parallel
connections), 69 requests spread over 20 dependent rounds cost roughly
20 x 70 ms = **1.4 s**, while one 512 KB request costs 70 ms + 0.27 s = **0.34 s**.

**At two thousand documents the naive client wins**, and by a factor of four. Fetching
9.1 KB instead of 512 KB is worth nothing when the 9.1 KB arrives as 69 dependent
round-trips on a link where a round-trip costs more than 130 KB of transfer.

That is not an argument against the index design; it is a statement of where the
design earns its keep. A 5.9 MB database can be swallowed whole, so nothing cleverer
than swallowing it is needed. A database that cannot be swallowed is the case the
whole project exists for, and it is what the million-document scale measures — at
which point section 9.3's figure for FTS5 (689 pages and 2.8 MB for a *single term*)
is the thing to beat, and "just download it" is not available at roughly a gigabyte.

The honest summary of the demo is therefore: it proves the machinery works
end to end in a browser, it produces identical results across every path, and it
establishes the crossover question rather than answering it.

---

## 14. Late interaction, with the real tokenizer

`tokenizer.json` arrived mid-project and unblocks section 3.2.

### 14.1 What it is, and that it matches

ByteLevel BPE, NFC normalisation, 50,280 vocabulary entries plus 118 added tokens
reaching id 50,369 — so **50,370 tokens, exactly the row count of the checkpoint's
`tok_embeddings.weight [50370, 256]`**. The template wraps input as
`[CLS] … [SEP]` with `[CLS]`=50281, `[SEP]`=50282. This is the ModernBERT family, as
the graph's rotary embeddings and `tok_embeddings` naming implied.

The reference `tokenizers` library is used rather than a hand-written ByteLevel BPE.
The WordPiece port in section 5.1 was worth writing because it is 150 lines and had
to run in WASM; ByteLevel BPE with a 50k merge table is neither.

Per-token output norms measure **exactly 1.0000**, confirming the structural reading
in section 2.2 that the graph's trailing `ReduceL2`/`Clip` normalises its output.
MaxSim is therefore a plain dot product with no normalisation step.

### 14.2 Finding: the `[MASK]` pad token is a trap

`tokenizer.json` names **`[MASK]` (50284) as its padding token**. That is exactly
what ColBERT's query augmentation looks like: pad the query with `[MASK]` and
*attend to* those positions, so the model fills them with learned query expansion.

Following that convention here is wrong, and measurably so. On a five-query
code-retrieval probe:

| convention | top-1 |
|---|---:|
| pad queries to 32 with `[MASK]`, attend to padding | **2/5** |
| no padding | **5/5** |
| no padding, special tokens dropped | 5/5 |
| pad to 32 with `[MASK]`, special tokens dropped | 2/5 |

With padding attended to, one long document won every query it was not the answer
to. The reason is structural: MaxSim sums a maximum over document tokens *for each
query token*, so 15 meaningless `[MASK]` vectors appended to a 17-token query add 15
more maxima, which are largest for whichever document has the most tokens to offer.
The augmentation is a trained behaviour, not a free one, and this model was not
trained with it.

The encoder therefore excludes padded positions from both the attention mask and the
returned vectors. Special tokens are kept, since dropping them measured identically
and keeping them stays faithful to the tokenizer's own template.

This is the kind of error that does not announce itself: every vector still looks
plausible, every norm is still 1.0, and retrieval quality quietly halves.

---

## 15. A corpus the code model can actually be measured on

`LateOn-Code-edge` is a **code** model. Evaluating it on bags of random dictionary
words measures nothing it was trained to do, so a second corpus was added.

`tools/corpus/code.py` builds the standard docstring-to-code benchmark from the
local Python standard library, the same construction CodeSearchNet uses:

* a **document** is a function's source with its docstring removed;
* a **query** is the first sentence of that docstring;
* the **gold answer** is the function the docstring came from.

Relevance is established by construction rather than by judgment. Removing the
docstring is essential: leaving it in makes the query a literal substring of its own
answer, which degenerates the task into exact matching and would flatter every
lexical baseline. Ordering is by content hash, not filesystem order, so the corpus
does not depend on how the standard library happens to be laid out, and identical
function bodies are deduplicated so "gold" is never ambiguous.

Result: **3,366 functions**, mean 13.5 lines, 1.79 MB, SHA-256 `df079baf…f4a818`.

### 15.1 Head to head, 500 queries

| system | success@1 | success@10 | success@100 | MRR@10 | index build | bytes/doc |
|---|---:|---:|---:|---:|---:|---:|
| BM25 (FTS5) | 0.280 | 0.542 | 0.762 | 0.362 | 0.0 s | — |
| Dense (MiniLM, mean-pooled) | 0.350 | 0.698 | 0.932 | 0.463 | 55.6 s | 1,536 |
| **Late interaction (LateOn)** | **0.454** | **0.780** | **0.950** | **0.567** | 123.4 s | **28,240** |

Per-query latency is deliberately omitted from this table. Both runs shared the
machine with a million-document index build, and the figures moved by more than a
factor of two between runs that differed only in how much else was executing. The
quality columns are unaffected by contention; the timing columns would be dishonest.

**What success@1 means here, and its ceiling.** Each query has exactly one correct
document — the function its docstring was taken from — so success@1 is the fraction
of queries whose gold document is ranked first out of 3,366, identical to precision@1
and to plain accuracy. This is *known-item* retrieval, not topical relevance: a
system returning a genuinely better-matching function scores zero unless it is the
source one. That harshness is deliberate, since it is what removes human judgment
from the benchmark, but it has a consequence worth stating.

**7.4% of the eval queries cannot be answered at all.** Their docstring appears
verbatim on two or three different functions, so the query text carries nothing that
could distinguish them:

| docstring shared by | queries in the 500-query eval set |
|---|---:|
| 1 function (answerable) | 463 |
| 2 functions | 35 |
| 3 functions | 2 |

For a docstring shared by `m` functions, any fixed ranking answers exactly one of
those `m` queries correctly, so the **maximum attainable success@1 is 0.962**, not
1.0. Against that ceiling the measured scores are BM25 **0.291**, dense **0.364**,
late interaction **0.472**. The ordering is unchanged and the correction is under
five points, but quoting the raw figures against 1.0 overstates the remaining
headroom, and `code_eval.py` now computes and prints the ceiling rather than leaving
it to a footnote.

Two things that are *not* confounds here, both checked: BM25 hits a tie at rank 1 on
only **0.2%** of queries — unlike the word corpus, where ties were the dominant
effect and made FTS5's single-term score meaningless (section 9.6) — and no query
lacks usable query terms.

*Sensitivity to the sequence cap.* An earlier version of this table capped documents
at 512 tokens, which silently truncated 91 of the 3,366 -- a number picked without
grounding, when the tokenizer declares 2047 and the model, being RoPE-based, accepts
a 740-token input unchanged. Re-running uncapped moved success@1 from 0.456 to 0.454
and MRR@10 from 0.568 to 0.567, which on 500 queries is one query either way, and
raised storage 1.6% (27,797 to 28,240 bytes per document). **The cap was a real
defect with no measurable effect**, which is worth stating in both directions: the
configuration is now principled, and anyone reproducing the earlier numbers should
not expect the difference to show.

Late interaction wins on every quality measure: **62% better success@1 than BM25 and
30% better than dense**, with MRR@10 of 0.567 against 0.463 and 0.362. This is what
the random-word corpus could not show, and it is the result that justifies the model
being in the repository at all.

It also shows what late interaction costs. **28,240 bytes per document** — 18x dense
and, at a mean of 144.8 tokens per document, the dominant term in any storage budget.
Extrapolated to a million documents that is **28 GB**, against 1.5 GB for dense
float32 and 64 MB for dense PQ. Section 15.2 brings it to 646 MB. Exact MaxSim is two to three orders of magnitude
slower per query than a dense dot product; the exact ratio is not quoted, for the
contention reason above.

So the ranking on quality and the ranking on cost are exactly inverted, and neither
number alone decides anything. That is the case PLAID compression exists to
address, and section 12's staged pipeline is measured against this corpus next.

The exact-MaxSim figures were computed independently in Python and in Rust and agreed
to three decimals on the capped run (0.456 / 0.782, and 27,796 against 27,797 bytes
per document from integer rounding), which is the cross-check that the two
implementations of the scoring function agree.

### 15.2 PLAID compression on the code corpus

Staged retrieval per section 12, measured against the exact MaxSim ceiling on the
same 3,366 documents and 500 queries, and on the same embeddings (see the note at
the end of this section). `cand@k` is the centroid-only ranking; `rerank@1` is after
exact rescoring of a 100-document pool.

| centroids | bytes/doc | compression | cand@1 | cand@10 | **rerank@1** | build |
|---:|---:|---:|---:|---:|---:|---:|
| exact (no compression) | 28,239 | 1.0x | — | — | **0.454** | — |
| 512 | 617 | **45.7x** | 0.144 | 0.452 | 0.444 | 24 s |
| **1,024** | **646** | **43.7x** | 0.240 | 0.578 | **0.454** | 48 s |
| 2,048 | 705 | 40.0x | 0.302 | 0.630 | **0.454** | 97 s |

At 1,024 centroids the *centroid representation* is **43.7x smaller** than the raw
token vectors, and reranking a pool against exact vectors recovers exact quality to
three decimals. Doubling to 2,048 centroids buys a better *first-stage* ranking
(cand@1 climbs 0.240 to 0.302) but nothing after reranking: the first stage only has
to get the right document into the pool, and by 1,024 it already does.

**But the compression figure and the quality figure describe different
configurations, and pairing them overstates the system.** `footprint()` counts the
centroid codes and the centroid table — exactly what PLAID compresses — and nothing
else. A file that can actually *answer* a query carries the inverted lists and an
offsets directory too, and a configuration that reranks exactly must also store the
uncompressed token vectors it reranks against. Measured from `dbstat` on the stored
databases (section 15.4):

| k = 1,024 configuration | success@1 | bytes/doc, as stored | index |
|---|---:|---:|---:|
| centroid stages only, no rerank | **0.240** | 928 | 3.1 MB |
| with exact rerank of 100 | **0.454** | **29,196** | 98.3 MB |

So this implementation is either cheap and weak or strong and expensive, with nothing
in between. The missing piece is **residual quantization**: real ColBERTv2/PLAID
stores a few bits of residual per token so that reranking happens in the compressed
domain, which is what would make an accurate configuration also a small one. Sections
12 and 15.2 implement PLAID's *staging* and its centroid compression; they do not
implement its residuals, and the 43.7x figure should be read as applying to the
candidate-generation data alone.

Extrapolated to a million documents: the no-rerank configuration is ~928 MB, the same
order as the FTS5 baseline's 730 MB; the exact-rerank configuration is ~29 GB, which
is not deployable. Closing that gap is the single most valuable follow-up in the
project.

*Cross-check.* Exact MaxSim was computed independently in Python and in Rust, on the
same embeddings, and agrees to three decimals: 0.454 success@1 and 0.780 success@10
in both, with 28,240 against 28,239 bytes per document from integer rounding. That
is the check that the two implementations of the scoring function agree.

*Provenance.* An earlier version of this table was computed from a different
encoding than the head-to-head above it. Re-running the encoder without the sequence
cap rewrote the vector file, while the per-document token-length table that the Rust
benchmark uses to slice it was produced by a separate manual step and stayed behind
— 495,075 vectors described by a table covering 487,313. A bounds assertion would
have caught the mismatch before it produced wrong output, but the *reported* figures
had already been mixed across two encodings. Both files are now written in the same
pass, so they cannot diverge, and every number in this section comes from one run.

### 15.3 Null result: probe width does nothing at this corpus size

Probing 4, 16 or 32 centroids per query token changes the numbers in the third
decimal place. That is not a bug, and the arithmetic says why.

A document here holds a mean of 144.8 tokens. With `k` centroids, a document
therefore touches on the order of 145 of them — at `k = 512` that is **28% of every
centroid in the index**. So any single centroid's posting list already contains
roughly a quarter of the corpus, and a query with 17 tokens probing even one centroid
each retrieves nearly all 3,366 documents. There is nothing left for a wider probe to
add.

**PLAID's inverted list provides no pruning at this scale**, and the speedup measured
above comes entirely from the other two mechanisms: centroid-only scoring is cheaper
per candidate than full MaxSim, and exact rescoring runs over 100 documents instead
of 3,366. Selectivity would require a corpus large enough that a centroid appears in
a small fraction of documents — which, at 145 tokens per document, means `k` far
above the document count. This corpus cannot show that, and the honest reading is
that the candidate-generation stage is untested here rather than that it is useless.

### 15.4 Late interaction in SQLite, with page accounting

Sections 15.1 and 15.2 measured quality and size. Both were measured in memory, so
neither said anything about the axis this project is about. This section stores the
`LateIndex` in an ordinary SQLite file and reports what a client holding no copy of
that file pays, per stage, so late interaction can be set beside FTS5 and the dense
index on the same terms.

**The storage problem, and the trade made.** The Vamana format gets its page
arithmetic free: fixed-size records, so node `i` is at byte `i * record_bytes`
(section 11.3). A late-interaction document is a list of one centroid id per token,
and on this corpus that is a mean of 147 and a maximum of 830 — there is no record
size that is both correct and predictable. Three options were weighed:

| option | cost | verdict |
|---|---|---|
| offsets **table**, looked up per candidate | one extra *dependent* round-trip per candidate | rejected: this is FTS5's `%_docsize` pattern, the 79x penalty of section 9.4 |
| pad to a fixed size | 5.64x to the corpus maximum; 2.00x with four quantile buckets; 1.22x with sixteen | rejected: gives back most of PLAID's 43.7x |
| **contiguous arena + resident offsets directory** | **1.007x** (4 bytes per document) plus a one-time preload | **chosen** |

So each variable-length collection is one BLOB — an arena — holding every document's
data end to end in document order, with an `n + 1` entry `u32` offsets directory
carried in the metadata a client downloads once. Document `i` is the byte range
`[off[i] * 4, off[i+1] * 4)`, and the pages covering it follow by the same
arithmetic a fixed record would use. The directory is 13 KB here, 0.7% of the arena
it indexes, and 4 MB at a million documents. Three arenas — postings, centroid codes,
exact token vectors — one per stage, so the stages cannot share a page and their
costs cannot be confused. Reads go through `sqlite3_blob_open` at a byte offset:
core SQLite, present in a stock WASM build, no virtual table and no extension.

SQLite allocates an oversized BLOB's overflow pages consecutively within one insert,
so a byte range is a *run* of consecutive pages. That is asserted rather than
assumed: the arena's real page numbers are read out of `dbstat` at open, checked
against the local/overflow split, and a test fails if the chain is not consecutive.

**Measured, 3,366 documents, 500 queries, page size 4096.** Mean per query.

| k | probe | rerank | succ@1 | succ@10 | MRR@10 | pages | requests | hops | bytes/doc | build |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 0 | 0.140 | 0.450 | 0.229 | 585 | 33 | 2 | 852 | 22.5 s |
| 512 | 8 | 100 | 0.444 | 0.716 | 0.540 | 1,806 | 129 | 3 | 29,120 | 22.5 s |
| 1,024 | 4 | 0 | 0.240 | 0.578 | 0.343 | 550 | 33 | 2 | 928 | 44.9 s |
| **1,024** | **4** | **100** | **0.454** | **0.760** | **0.563** | **1,749** | **130** | **3** | 29,196 | 44.9 s |
| 1,024 | 8 | 0 | 0.240 | 0.578 | 0.343 | 592 | 42 | 2 | 928 | 44.9 s |
| 1,024 | 8 | 100 | 0.454 | 0.760 | 0.563 | 1,791 | 139 | 3 | 29,196 | 44.9 s |
| 1,024 | 32 | 0 | 0.242 | 0.580 | 0.344 | 680 | 25 | 2 | 928 | 44.9 s |
| 1,024 | 32 | 100 | 0.454 | 0.760 | 0.563 | 1,878 | 122 | 3 | 29,196 | 44.9 s |
| 2,048 | 8 | 0 | 0.304 | 0.630 | 0.406 | 589 | 49 | 2 | 1,025 | 89.2 s |
| 2,048 | 8 | 100 | 0.454 | 0.774 | 0.565 | 1,771 | 146 | 3 | 29,293 | 89.2 s |

Quality reproduces section 15.2 out of the database to three decimals — 0.444 at
512 centroids, 0.454 at 1,024 and 2,048, against the exact-MaxSim ceiling of 0.454 —
which is the check that the stored form is the index and not something adjacent to
it. Against the attainable ceiling of 0.962 (section 15.1), 0.454 is 0.472.

**Per stage.** Mean per query, probe 8.

| k | rerank | stage | pages | requests | payload | arena |
|---:|---:|---|---:|---:|---:|---:|
| 1,024 | 100 | postings | 108.1 | 39.8 | 175 KB | 226 pg |
| 1,024 | 100 | centroid | **484.0** | **2.0** | 1.98 MB | **484 pg** |
| 1,024 | 100 | rerank | **1,198.7** | **97.0** | 4.51 MB | 23,230 pg |

Three findings, in order of how much they matter.

*Reranking is the expensive stage, as it was for the dense index.* It is **67% of
the pages and 70% of the requests**, and nearly all of the bytes: 4.5 MB against the
other two stages' 2.2 MB combined. Its 97 requests for 100 documents is the worst
ratio in the table — the pool is in score order, not id order, so essentially every
reranked document is its own range request. This is the same shape as section 11.5,
where full vectors dominated a traversal that had already been made cheap, and for
the same reason: the pool is scattered by construction.

*The centroid stage is a full scan, and should therefore be resident.* Stage 2
touches **484 of the code arena's 484 pages** on every query, because stage 1's
candidate pool is **100.0% of the corpus** — the arithmetic of section 15.3, now
measured in pages. But it touches them in **2 requests**, because they are one
contiguous run. A client that fetches the 1.98 MB arena once per session pays that
twice and gets every later query's stage 2 for nothing, which is exactly the
resident-codes trade of section 11.5 and is strictly better here than it is there, since the
"traversal" reads the whole thing anyway.

*Wider probing costs bytes and saves round-trips.* From probe 4 to 32 at k=1,024,
postings pages go 66 → 196 but requests go **33 → 25**: more of the postings arena is
read, and the spans coalesce into fewer runs. Section 15.3 found probe width does
nothing for quality at this scale; it is not neutral for cost, and the sign is the
opposite of the obvious guess.

**What `bytes_per_doc` means here, and why it is not 646.** Section 15.2's 646 bytes
counts `codes * 4 + centroids * 4`. A file that can actually answer a query also
needs the inverted lists and the offsets directory, and pays SQLite's page rounding:
**928 bytes per document** at 1,024 centroids without reranking. With reranking it
needs the exact token vectors too, and the figure is **29,196** — the compression is
in the *first two stages only*, and a configuration that reranks exactly is storing
the uncompressed corpus whatever the index costs. That is the honest statement of
what PLAID buys and does not buy, and it is the thing the 43.7x headline hides.

**Hops are 2 or 3, flat.** There is no graph to walk: stage 1 knows every range it
wants once the resident centroid table is scored, stage 2 once stage 1 returns,
stage 3 once stage 2 sorts. The dense traversal pays eight or more dependent rounds
at 10,000 documents. This is the structural advantage late interaction has over a
graph index on a high-latency link, and it does not degrade with corpus size.

*Timing.* `cpu_ms_per_query` is 24 ms without reranking and 61 ms with, taken from
`/proc/self/stat` rather than the wall clock because the machine was shared; the
records carry `contended: true`. Pages, requests, bytes and quality are unaffected by
load. Build times are wall clock and did move: the same k-means that took 89 s here
is quoted at 97 s in section 15.2.

*Not measured.* The obvious next lever is capping the candidate pool before stage 2
rather than scoring all 3,366; that changes the algorithm whose quality sections 15.1
and 15.2 published, so it was left out rather than mixed in. Results are in
`bench/results/tri-late.jsonl`.

---

## 16. The scale story

Three scales of the same growing collection — each corpus is a byte-exact prefix of
the next (section 4.3), so these are points on one curve rather than three separate
experiments. Dense index: Vamana R=32, alpha=1.1, PQ m=64. Query costs are the
measured counts put through the model in `tools/analyze/netcost.py`.

### 16.1 What ordering and residency buy, by scale

Mean per query at L=32, beam=4, no rerank. Identical graph, codes, queries and
parameters within each scale.

| scale | records read | pages, insertion order | pages, BFS | pages, cluster | requests, insertion order | requests, BFS |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 978.0 | 420.5 | **280.4** | 355.7 | 49.0 | 64.8 |
| 100,000 | 1,291.1 | 1,135.3 | **659.5** | 916.3 | 861.3 | **355.4** |
| 1,000,000 | 1,543.5 | 1,518.9 | **994.0** | 1,384.4 | 1,470.0 | **755.7** |

At 10,000 documents BFS ordering cuts pages by a third but *raises* the coalesced
request count, 49.0 to 64.8. That is not a contradiction: a query there touches most
of the 500-page table, so insertion order's pages happen to be one long contiguous
sweep, while BFS's smaller set is scattered across it. The metric that improves
depends on whether the client fetches pages or ranges, and at that scale the answer
is "neither matters much". By 100,000 documents both improve together and decisively.

And with codes resident, which changes *when* records are read rather than which:

| scale | records (on disk) | records (resident) | pages (BFS, on disk) | pages (BFS, resident) | requests (BFS, resident) |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 978.0 | 44.9 | 280.4 | 37.1 | 30.7 |
| 100,000 | 1,291.1 | 51.9 | 659.5 | 44.9 | 41.2 |
| 1,000,000 | 1,543.5 | **59.2** | 994.0 | **50.6** | **47.3** |

Records read grows only from 978 to 1,543 across a hundredfold increase in corpus
size — the graph is doing its job. What grows is the number of *pages* those records
are scattered over, from 421 to 1,519 under insertion order, which is the cost
ordering and residency exist to attack.

Recall is identical to three decimals in every resident/on-disk pair and across all
three orderings, as it must be: the same nodes are scored either way, and a test
pins it (`ordering_changes_pages_but_not_results`).

**At a million documents both levers hold and compose.** Breadth-first ordering takes
pages from 1,519 to 994 and coalesced requests from 1,470 to 756; resident codes take
records read from 1,543 to 59. Together, **1,470 requests become 47 — a factor of
31** — at recall identical to three decimals. Cluster ordering again lands between
insertion order and BFS.

**Ordering pays more as the corpus grows.** At 10,000 documents a query touched most
of the node table, so BFS ordering cut pages by a third while leaving requests no
better. At 100,000 it cuts pages by 42% *and* requests by 59%, because there is now
enough table for locality to be a meaningful property rather than a rounding error.
Cluster ordering lands consistently between the two: grouping by similarity helps,
but grouping by the graph the search actually walks helps more.

**Residency pays more still, and independently.** It is the larger lever at both
scales — 1,291 records down to 51.9 at 100,000, a factor of 25 — because it attacks
a different quantity. Ordering changes *where* records sit; residency changes *how
many* have to be read at all. Composing them takes 861 requests to 41.

### 16.2 The cost of the things that are not the traversal

Two costs are easy to leave out of a comparison and both are charged here.

**Reranking.** PQ is a candidate generator (section 7.3), so reaching useful recall
means rescoring a pool against full float32 vectors. At 100,000 documents, L=128,
beam=16, that lifts recall@10 from 0.342 to **0.434** and adds 99 pages of scattered
1.5 KB reads — on `lte`, 2.4 s becomes 3.9 s. Reranking is not free and is not
optional; it is the difference between a mediocre index and a usable one, bought
with about 60% more time.

**Preloading.** Resident codes cost `n * m` bytes once: 0.64 MB at 10,000, 6.4 MB at
100,000, 64 MB at a million. The per-query tables exclude that; the session tables in
`docs/RESULTS.md` include it, and they are the ones that decide the design. On `lte`
at 100,000 documents the preload is 3.5 s, so it repays after a handful of queries
and is pure loss for a single one.

### 16.3 Against the baseline

Both now measured at the same scale. A median FTS5 query at a million documents
costs 1,537 pages and 6.1 MB; the dense index at the same scale, breadth-first with
resident codes, costs 82 pages and 79 requests.

| system (1M documents) | pages | requests | `lte` | `3g` | satellite |
|---|---:|---:|---:|---:|---:|
| FTS5 | 1,537 | 630 | **111 s** | 339 s | 925 s |
| dense, insertion order, on disk | 1,551 | 1,502 | 21.0 s | 82.2 s | 154 s |
| dense, BFS, on disk | 1,026 | 787 | 12.3 s | 49.8 s | 88.1 s |
| **dense, BFS, resident** | **82** | **79** | **1.4 s** | **5.3 s** | **10.9 s** |

**79x faster than the baseline on `lte`, 85x on satellite.** The mechanism is the one
section 9.4 identified: FTS5's cost is a per-match random lookup that grows with the
number of matches, while the graph index's cost is a hop count that grows with the
logarithm of the corpus. Records read rose only 978 to 1,543 from ten thousand
documents to a million.

Two things this table does **not** say, and both matter.

*It is not a quality comparison.* Recall@10 here runs 0.213 to 0.366 while FTS5
scores 0.760 success@1 on the same corpus. Dense retrieval loses badly on
random-word documents, for the reason established in section 5.4: MiniLM embeddings
of fifty unrelated dictionary words sit in a narrow cone with almost no
discrimination. This is a cost result. The quality comparison lives on the code
corpus (section 15.1), where late interaction reaches 0.454 against BM25's 0.280.

*The preload is not free.* The resident row excludes a 64 MB code blob, which on
`lte` is 34 seconds. For a single query the on-disk variant wins on wifi, `lte` and
`3g`; resident wins from about ten queries onward, and by a hundred it is not close.
The session tables in `docs/RESULTS.md` carry the crossover for every profile. Even
counting the preload in full, one query costs 35 s against the baseline's 111 s.

---

## 17. Three systems, one corpus, and what parallelism does to the ranking

Sections 9 to 16 measured FTS5, the dense index and late interaction on different
corpora with different metrics, so no honest three-way comparison existed. All three
are now measured on the **code corpus** (3,366 documents, 500 queries), with the same
single-gold quality definition and the same pass-through VFS counting real file
pages.

### 17.1 Cost against quality

`success@1` ceiling is 0.962 (section 15.1). CPU is process time under contention —
ordering is meaningful, absolute values are an upper bound.

| system | configuration | success@1 | of max | MRR@10 | pages | requests | bytes/doc | cpu ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FTS5 | post-vacuum | 0.280 | 0.291 | 0.362 | **25.5** | **12.3** | **764** | 2.09 |
| dense | L=128, m=64, no rerank | 0.348 | 0.362 | 0.456 | 67.4 | 66.7 | 2,420 | **1.06** |
| dense | L=128, m=32, rerank 100 | 0.348 | 0.362 | 0.461 | 158.6 | 125.9 | 2,352 | 1.61 |
| late | k=1,024, no rerank | 0.240 | 0.249 | 0.343 | 592.1 | 41.8 | 928 | 23.76 |
| **late** | **k=1,024, rerank 100** | **0.454** | **0.472** | **0.563** | 1,790.8 | 138.7 | 29,196 | 61.92 |

The three axes disagree, which is the useful part. Late interaction wins quality
outright and loses every cost measure — 70x the pages of FTS5, 38x the bytes, 30x the
CPU. FTS5 is cheapest on all three and worst on quality. Dense sits between, and is
cheapest on CPU.

One result worth isolating: **dense at m=64 without reranking matches m=32 with
reranking on quality (0.348 either way) at 67 pages instead of 159.** Spending 32
more bytes per document to avoid fetching a hundred scattered vectors is strictly
better here.

### 17.2 Finding: requests in flight reorder the systems completely

Page counts hide the distinction that decides latency. Requests *within* a hop can be
overlapped, because their addresses are known together; hops cannot, because the next
hop's addresses are unknown until the current returns. Parallelism divides the first
and leaves the second, so latency is bounded below by `hops x RTT` however wide the
pipe — and the three architectures differ enormously in hop count.

| system | hops | why |
|---|---:|---|
| FTS5 | 12 | equal to its requests: `xRead` is synchronous, so SQLite asks for the next page only after the current returns. **It cannot batch at all** — a property of the client, not the index. |
| dense | 34–35 | one per beam round; each frontier depends on the last. |
| **late interaction** | **2–3** | fixed stages: postings, centroid scoring, optional rerank. Independent of corpus size. |

Seconds per query on **`satellite`** (600 ms RTT, 20 Mbit/s):

| system | c=1 | c=6 | c=32 | c=128 | floor |
|---|---:|---:|---:|---:|---:|
| FTS5 | 7.24 | 7.24 | 7.24 | 7.24 | 7.24 |
| dense m=64, no rerank | 40.91 | 20.51 | 20.51 | 20.51 | 20.51 |
| dense m=32, rerank 100 | 84.26 | 21.26 | 21.26 | 21.26 | 21.26 |
| **late, no rerank** | 26.17 | 5.77 | **2.17** | **2.17** | **2.17** |
| late, rerank 100 | 87.53 | 17.33 | 6.53 | **4.73** | 4.73 |

**Serially, FTS5 wins by 3.6x. With 32 requests in flight, late interaction wins by
3.3x** — a complete reversal, from the same measurements, purely by allowing what a
browser already does. The dense index barely improves past six, because its 34
dependent rounds are a floor no parallelism touches.

The profile decides which term dominates:

| link | winner | why |
|---|---|---|
| `3g` (200 ms, 1.6 Mbit/s) | FTS5 at every concurrency | bandwidth-bound; late interaction moves 7.3 MB |
| `lte` (70 ms, 15 Mbit/s) | FTS5 (0.90 s) | corpus too small for posting lists to hurt |
| `satellite` (600 ms, 20 Mbit/s) | **late interaction, given c >= 32** | latency-bound, and hops are what latency charges for |

So "how many pages does a query touch" is the wrong single question. **Pages bound
the bytes; hops bound the latency; and only hops are immune to parallelism.** Late
interaction's flat 2–3 hops is a structural property — no graph to walk — that no
amount of page-locality work on the graph index can match, and it is invisible in
every table before this one.

*Caveat.* Late interaction's floor on `satellite` is transfer-bound rather than
latency-bound once reranking is on: 4.73 s for 3 hops is 1.8 s of round-trips and
2.9 s of moving 7.3 MB. Section 15.2's unresolved gap — no residual quantization, so
reranking reads uncompressed vectors — is what puts it there.

---

## 18. Open items

* **Blocked:** `tokenizer.json` for `LateOn-Code-edge` (§3.2) — gates milestone 5.
* **Needed:** an emscripten toolchain for the WASM milestone.
* Out of scope this pass: LLM-written paragraph corpora (§1.1).
