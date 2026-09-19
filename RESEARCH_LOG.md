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

At 1M, k=10: 2,395 pages in **2,380 runs** before vacuum, the same 2,395 pages in
**133 runs** after. A client that coalesces adjacent pages goes from ~2,380 requests
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

## 11. Open items

* **Blocked:** `tokenizer.json` for `LateOn-Code-edge` (§3.2) — gates milestone 5.
* **Needed:** an emscripten toolchain for the WASM milestone.
* Out of scope this pass: LLM-written paragraph corpora (§1.1).
