# annlite

annlite is a set of approximate-nearest-neighbour (ANN) search indexes for SQLite. It
implements dense vector search and late-interaction (multi-vector) search over a
SQLite database file that the client does not hold locally, reading it over HTTP range
requests. The intended deployment is a static database file on a CDN with a
WebAssembly client and no search server, in the manner of
[sql.js-httpvfs](https://github.com/phiresky/sql.js-httpvfs).

In that setting the dominant cost is the number of network round-trips, not CPU time
or bandwidth. An index that performs well on a local disk can be unusable over a
mobile connection if its access pattern is scattered across many database pages. Every
index in this repository is therefore measured in pages read, HTTP requests issued and
dependent round-trips required, as well as in milliseconds.

![The demo application running in a browser](docs/demo.png)

Methodology and full tables are in [RESEARCH_LOG.md](RESEARCH_LOG.md). A generated
summary of all results is in [docs/RESULTS.md](docs/RESULTS.md). Every figure in
either document is derived from a measurement file committed under `bench/results/`,
and `make matrix` regenerates the summary from those files without re-running any
benchmark.

## Background

The project's design follows from the relative cost of latency and transfer. The
table below gives the time to fetch 32 sequential 4 KiB database pages (128 KiB in
total) under three simulated network profiles, separated into time spent waiting for
responses and time spent transferring data.

| Profile | Waiting | Transferring |
|---|---:|---:|
| wifi | 499 ms | 21 ms |
| lte | 2,531 ms | 75 ms |
| satellite | 20,125 ms | 55 ms |

Transfer time is a small fraction of the total in each case. The design work in this
repository is consequently directed at reducing the number of *dependent* fetches —
requests whose addresses are not known until an earlier request returns — rather than
the number of bytes transferred.

## Requirements

| Component | Version used | Required for |
|---|---|---|
| Rust | 1.94.1 | all indexes, corpus generation, benchmarks |
| Python | 3.11.15 | encoders, corpus builder, reporting scripts |
| `wamerican` | 2020.12.07 | supplies `/usr/share/dict/words` for corpus generation |
| Node.js | 22.22.2 | browser parity tests and the WebAssembly client |
| `wasm32-unknown-unknown` target | — | `make wasm`, `make demo`, `make wasm-test` |
| `wasm-bindgen-cli` | 0.2.128 | `make wasm`, `make demo`, `make wasm-test` |
| Emscripten | 6.0.9 | `make sqlite-wasm`, `make range-demo` |
| SQLite | 3.46.0 | bundled by `libsqlite3-sys`; no system SQLite is used |

Python dependencies are pinned in `requirements.txt`. The onnxruntime version
determines the embeddings, and the embeddings determine every quality figure in this
repository, so it is pinned rather than left unconstrained.

Rust, Python and `wamerican` are sufficient to reproduce every measurement result.
Node.js, the WebAssembly target, `wasm-bindgen-cli` and Emscripten are needed only for
the browser targets. SQLite requires no action, as it is compiled from the
amalgamation vendored by `libsqlite3-sys`.

## Installation

On Debian or Ubuntu:

```sh
sudo apt-get update
sudo apt-get install wamerican python3-venv    # dictionary and venv support
make venv                                      # creates .venv from requirements.txt
```

Rust is installed through [rustup](https://rustup.rs). No further setup is needed for
the measurement targets.

For the browser targets, add the WebAssembly target and the matching `wasm-bindgen`
command-line tool:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.128
```

`make sqlite-wasm` additionally requires Emscripten and the vendored SQLite
amalgamation:

```sh
git clone https://github.com/emscripten-core/emsdk.git ~/emsdk
cd ~/emsdk && ./emsdk install latest && ./emsdk activate latest
cargo fetch        # vendors libsqlite3-sys, whose sqlite3.c is compiled to WASM
```

An SDK installed elsewhere can be named on the command line, as
`make sqlite-wasm EMSDK=/path/to/emsdk`, or the compiler named directly with
`EMCC=/path/to/emcc`.

Each target checks its own prerequisites and reports the missing one with the command
that installs it.

## Usage

`make` with no arguments lists all targets.

```sh
# Data. Corpora are generated rather than stored; see "Corpus generation".
make corpora queries           # word corpora at 100, 10k and 1M documents
make code-corpus               # docstring-to-code corpus from the Python standard library

# Measurement. Each target writes to bench/results/.
make fts5                      # FTS5 baseline at all scales
make ann                       # dense index sweep at 10k, 100k and 1M
make code-eval                 # BM25, dense and late interaction on the code corpus
make late-pages                # late interaction in SQLite, with page accounting
make late-words SCALE=10k      # late interaction on the word corpus
make residual                  # residual quantization: storage against quality

# Reporting. These read committed results only and re-run no benchmark.
make matrix                    # regenerate docs/RESULTS.md
make concurrency               # seconds per query against requests in flight

# Browser
make sqlite-wasm               # SQLite 3.46.0 with the bounded-range VFS
make range-demo                # page and request counts for the WebAssembly client
make demo && make demo-serve   # demo application, served through the network simulator

# Checks
make test                      # Rust and Python test suites
make tokenizer-parity          # compares the Rust and Python tokenizers
make wasm-test                 # compares the browser build against the native one
```

Build times vary considerably. The 1M-document dense index takes approximately two
hours to build and is available as a separate `make ann-1m` target for that reason.

## Results

Each subsection names the target that produces its figures.

### Full-text search baseline

SQLite's FTS5 extension provides the baseline. Page counts were measured with a
pass-through virtual file system that intercepts `xRead`; its counts agreed with
`SQLITE_DBSTATUS_CACHE_MISS` on all 9,000 queries measured.

| Documents | Median pages per query | KB read | Time on `lte` |
|---|---:|---:|---:|
| 10,000 | 26 | 104 | 1.9 s |
| 1,000,000 | 1,537 | 6,148 | 111 s |

Produced by `make fts5`.

The cost is attributable to the ranking function rather than to the index. The same
queries run without `ORDER BY bm25()` read 19.8 pages instead of 1,557.6, a factor of
79, because FTS5 performs one random lookup in its `%_docsize` table for every match.
The record format used by the dense index follows from this observation: a node's
quantized vector is stored adjacent to its adjacency list, so a single page read
yields both the score and the next set of candidates.

Running `VACUUM` on the FTS5 database leaves the page count unchanged at 2,395 but
reduces the number of contiguous runs from 2,380 to 133. A client that combines
adjacent pages into single range requests therefore issues about one-twentieth as many
requests for an identical query.

### Dense vector index

The dense index is a Vamana (DiskANN) graph with 64-byte product-quantized vectors.
Within each scale, the graph, quantized codes, queries and search parameters are
identical across orderings, so differences are attributable to node numbering alone.

| Documents | Records read | Pages, insertion order | Pages, BFS order | Requests, insertion order | Requests, BFS order |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 978 | 421 | 280 | 49 | 65 |
| 100,000 | 1,291 | 1,135 | 660 | 861 | 355 |
| 1,000,000 | 1,544 | 1,519 | 994 | 1,470 | 756 |

Produced by `make ann`.

The number of records read grows from 978 to 1,544 across a hundredfold increase in
corpus size, which is the expected behaviour for a graph index. What grows faster is
the number of distinct pages those records occupy. Two independent techniques reduce
it.

The first is node ordering. Because the index is a table of fixed-size records in
rowid order, the page a node occupies is determined by the identifier it is assigned.
Numbering nodes in breadth-first order from the graph's medoid places nodes near the
neighbours through which a search reaches them, which reduces pages read by up to 33%
for identical results.

The second, and larger, is making the quantized codes resident: downloading all codes
once as a single contiguous blob, after which a record is read only for a node the
search expands rather than for every node it scores.

| Documents | Records read, on disk | Records read, resident | Requests, BFS and resident |
|---:|---:|---:|---:|
| 10,000 | 978 | 45 | 31 |
| 100,000 | 1,291 | 52 | 41 |
| 1,000,000 | 1,544 | 59 | 47 |

Combined, the two techniques reduce the request count at one million documents from
1,470 to 47. Recall is identical to three decimal places in every pair, as it must be
because the same nodes are scored either way; a test asserts this.

The resident configuration requires a 64 MB preload at one million documents, which
takes approximately 34 s on the `lte` profile. The on-disk configuration is therefore
faster for a single query and the resident configuration is faster from roughly ten
queries onward. `tools/analyze/netcost.py` reports this crossover per profile rather
than declaring a single winner, because it is a property of the session rather than of
the index.

### Comparison at one million documents

| System | Pages | Requests | `lte` | `satellite` |
|---|---:|---:|---:|---:|
| FTS5 | 1,537 | 630 | 111 s | 925 s |
| Dense, insertion order, on disk | 1,551 | 1,502 | 21.0 s | 154 s |
| Dense, BFS order, resident codes | 82 | 79 | 1.4 s | 10.9 s |

Two qualifications apply to this table. First, it compares cost and not quality:
recall on this corpus is between 0.213 and 0.366, against FTS5's success@1 of 0.760,
because embeddings of fifty unrelated dictionary words carry little discriminating
information. Second, the resident row excludes the 64 MB preload described above.

### Retrieval quality on the code corpus

The code corpus is a known-item retrieval benchmark built from the Python standard
library: each document is a function with its docstring removed, each query is that
docstring, and the correct answer is the originating function. Ground truth follows
from the construction, so no relevance judgements are involved.

Because 7.4% of queries share a docstring with another function, and are therefore
indistinguishable from one another, the highest attainable success@1 on this query set
is 0.962 rather than 1.0. Figures below are stated against that ceiling where
relevant.

| System | success@1 | MRR@10 | Bytes per document |
|---|---:|---:|---:|
| BM25 (FTS5) | 0.280 | 0.362 | — |
| Dense (MiniLM) | 0.350 | 0.463 | 1,536 |
| Late interaction (LateOn) | 0.454 | 0.567 | 28,240 |
| PLAID staging, 1,024 centroids, no reranking | 0.240 | 0.343 | 928 |
| With 2-bit residuals, reranking compressed | 0.388 | 0.493 | 2,411 |
| With 4-bit residuals, reranking compressed | 0.400 | 0.520 | 4,176 |
| With exact reranking against float32 | 0.454 | 0.563 | 29,196 |

As a fraction of the 0.962 ceiling: BM25 0.291, dense 0.364 and late interaction
0.472. Produced by `make code-eval` and `make residual`.

Storage and accuracy are not independent. PLAID's centroid representation is 43.7
times smaller than raw token vectors but scores 0.240 on its own; reaching 0.454
requires reranking against uncompressed vectors, which returns the stored size to
about 29 KB per document.

Residual quantization reduces this gap. Storing a few bits of the residual
`v - centroid[c]` per dimension allows reranking against a reconstruction rather than
a stored vector. At two bits this requires 2,411 bytes per document, which is 8.5% of
the exact configuration's storage for 85% of its success@1. The gap does not close
entirely: doubling from two bits to four adds 0.012 to success@1 and still does not
reach 0.454, indicating that beyond two bits the quantizer is no longer the limiting
factor. See [RESEARCH_LOG §18.1](RESEARCH_LOG.md).

Per-query latency is omitted from this table because these runs shared a machine with
a one-million-document index build. [RESEARCH_LOG §18.2](RESEARCH_LOG.md) re-measures
CPU time on an otherwise idle machine.

Late interaction also has a SQLite storage format with the same page accounting as the
dense index, which allows its cost to be compared and not only its quality. Mean
figures per query over the same 500 queries, with 1,024 centroids:

| Stage | Pages | Requests | Payload |
|---|---:|---:|---:|
| Candidate generation (postings) | 108 | 40 | 175 KB |
| Centroid interaction | 484 | 2 | 1.98 MB |
| Exact reranking of 100 candidates | 1,199 | 97 | 4.51 MB |

Produced by `make late-pages`.

A late-interaction query requires two dependent round-trips without reranking and
three with it, and this does not vary with corpus size because there is no graph to
traverse. Reranking accounts for 67% of pages read, as it does for the dense index.
The centroid stage reads every page of its arena on every query, because at this
corpus size the inverted list eliminates almost no candidates; the arena would be
better downloaded once and held resident. See
[RESEARCH_LOG §15.4](RESEARCH_LOG.md).

### Retrieval quality on the word corpus

The table above uses the code corpus, which is the case late interaction is designed
for. The word corpus is the opposite case: 10,000 documents of fifty randomly selected
dictionary words each. All three systems were scored on the same 500 known-item
queries against the same gold documents.

| System | success@1 | MRR@10 | Pages | Requests | Hops |
|---|---:|---:|---:|---:|---:|
| BM25 (FTS5) | 0.826 | 0.875 | 26.6 | 14.4 | 14 |
| Dense (Vamana and PQ, resident) | 0.106 | 0.128 | 103.7 | 72.0 | 35 |
| Late interaction (reranking 100) | 0.536 | 0.550 | 1,890.0 | 156.7 | 3 |

Produced by `make fts5-10k`, `make ann-gold SCALE=10k` and
`make late-words SCALE=10k`.

Lexical matching is more accurate and less expensive than either vector method on this
corpus. This is the expected result: a document of fifty unrelated dictionary words
has no subject matter for an embedding to represent. The dense result is not an
indexing failure, since the same run recalls 0.757 of its own exact search's top ten;
the limitation is in the representation. The case for a graph index begins at the
scale where FTS5's per-match lookups become expensive, which is nearer one million
documents than ten thousand. See [RESEARCH_LOG §17.3](RESEARCH_LOG.md).

### Browser client over HTTP range requests

`make sqlite-wasm` compiles SQLite 3.46.0 to WebAssembly using Emscripten, together
with a read-only virtual file system that issues one HTTP range request per `xRead`
call for exactly the bytes requested. The amalgamation compiled is the one
`libsqlite3-sys` vendors for the native benchmarks, so the browser and native
measurements use the same database engine.

Each measurement is counted twice, by the virtual file system and by the request log
of the network simulator, and is discarded unless the two counts agree.

| Operation | 5.9 MB, 2,000 documents | 274.5 MB, 100,000 documents |
|---|---:|---:|
| Point lookup | 3 pages, 12 KB | 4 pages, 16 KB |
| Four records, scattered identifiers | 6 pages | 7 pages |
| Four records, adjacent identifiers | 3 pages | 4 pages |
| Whole file | 1,438 pages | 67,024 pages |

Produced by `make range-demo`.

A 46-fold increase in database size adds one page to a point lookup, which is the
B-tree acquiring an additional level. The two four-record rows show the effect of node
ordering: four adjacent records share a leaf page and cost no more than a single
lookup, whereas four scattered records do not.

Applying the cost model to these counts, a traversal of the 274.5 MB database takes
5.9 s on `lte`, against 146 s to download the file in full. On the `satellite`
profile the comparison reverses with scale: downloading the 2,000-document file is
faster than range requests by a factor of 4.1, whereas for the 100,000-document file
range requests are faster by a factor of 2.4. See
[RESEARCH_LOG §19](RESEARCH_LOG.md).

## Implementation notes

The following results affected the design and may be of general interest.

Product quantization at 64 bytes is suitable for candidate generation but not for
ranking. 99.2% of the exact top ten appears in the quantized top 100, but only 61%
appears in the quantized top ten. Retrieving a deep candidate list and reranking a
shallow one is therefore necessary.

The `LateOn` tokenizer designates `[MASK]` as its padding token, which resembles
ColBERT query augmentation but is not. Attending to that padding reduced accuracy from
5/5 to 2/5 on a small probe, because MaxSim treats each padded position as another
candidate maximum. The output vectors retain unit norm in both cases, so the error
produces no visible symptom.

Vamana's `alpha` parameter has the opposite effect to the one its description
suggests. Increasing it prunes fewer edges, producing a denser graph with shorter
edges. Above 1.4 the graph degenerates towards a k-nearest-neighbour graph and recall
falls from 0.93 to 0.06.

The `sql.js-httpvfs` client makes page-locality work unobservable. It treats
`requestChunkSize` as a lower bound and escalates its read-ahead past one megabyte,
retrieving a 5.9 MB database almost in full on the first query. The WebAssembly client
described above reads 12 KB for a point lookup on the same database, a factor of
approximately 430, and makes the effect of node ordering measurable through SQLite.

An append-only results file concealed its own re-measurements. One benchmark appended
to its JSONL output while the reporting scripts selected the first row matching a
given configuration, so each re-run was recorded behind the measurement it was
intended to replace. The file is now truncated per run and the readers select the last
match. See [RESEARCH_LOG §18.3](RESEARCH_LOG.md).

## Models

Both encoders are committed to the repository. Their filenames do not indicate which
model each contains; they were identified from graph structure and confirmed by
running them, as described in [RESEARCH_LOG §2](RESEARCH_LOG.md).

| File | Model | Parameters | Output shape | Use |
|---|---|---:|---|---|
| `model_qint8_arm64.onnx` | `all-MiniLM-L6-v2` | 22.6M | `[batch, seq, 384]` | dense; mean-pooled and L2-normalised |
| `model_int8.onnx` | `lightonai/LateOn-Code-edge` | 17.0M | `[batch, seq, 48]` | late interaction; per-token, pre-normalised |

Tokenizers are also committed rather than downloaded, because a benchmark whose
tokenizer can change is not reproducible. The WordPiece tokenizer is implemented twice,
in Python for corpus preparation and in Rust for the browser; `make tokenizer-parity`
compares their output over 615 lines of input chosen to exercise edge cases.

## Corpus generation

Corpora are generated rather than stored. The one-million-document corpus is
approximately 454 MB and regenerates byte-for-byte in 5.4 seconds, so committing it
would add size without adding reproducibility.

| Corpus | Documents | Size | Generation time |
|---|---:|---:|---:|
| `docs-100.txt` | 100 | 45 KB | under 0.01 s |
| `docs-10k.txt` | 10,000 | 4.5 MB | 0.05 s |
| `docs-1m.txt` | 1,000,000 | 454 MB | 5.43 s |

Determinism is a property of the generator rather than of the machine. Every random
choice is drawn from ChaCha8 seeded with `SHA-256("annlite/v1/" ‖ domain ‖ seed)`,
which is fixed by the algorithm and reproduces on any platform. `shuf` is deliberately
not used: `shuf --random-source` is reproducible only within a single coreutils build,
and `shuf` without that option also depends on locale collation.

Each corpus is a byte-exact prefix of the next, so measurements across scales describe
one growing collection rather than three unrelated samples. Benchmark output is
committed, since it is the evidence for the figures above and totals only a few
megabytes.

The pipeline was verified end to end on a clean container: regenerating the corpora
from the system dictionary reproduced the SHA-256 digests recorded in
[RESEARCH_LOG §4.4](RESEARCH_LOG.md), and re-running `make ann-10k` from those corpora
reproduced all 72 rows of the committed `bench/results/ann-10k.jsonl` with no
differences in any field, including recall, pages, runs, hops and records read.

## Repository layout

```
crates/annlite-core      PQ, HNSW, Vamana/DiskANN, late interaction, layout, tokenizer
crates/annlite-corpus    deterministic corpus and query-set generation
crates/annlite-fts5      FTS5 baseline: build cost, latency, quality, page access
crates/annlite-netsim    HTTP range server with latency and throughput simulation
crates/annlite-sqlite    SQLite storage format and the page-locality experiment
crates/annlite-wasm      browser bindings: tokenizer, PQ scoring, resumable search
web/sqlite-wasm          SQLite compiled to WASM with a bounded-range HTTP VFS
web/demo                 demo application
tools/embed              ONNX encoders and the Python WordPiece tokenizer
tools/corpus             docstring-to-code corpus builder
tools/analyze            network cost model, tokenizer comparison, results matrix
models/tokenizers        committed tokenizer files
bench/results            committed measurement output
docs/RESULTS.md          generated results summary
```

## Project status

All indexes described above are implemented and measured at the scales given in the
tables. The one-million-document dense measurements and the FTS5 baseline at the same
scale are complete, all three retrieval systems have been measured on both corpora,
and the WebAssembly client reads a remote database through a virtual file system
maintained in this repository.

The principal gap is late interaction at one million documents. Encoding would take
approximately one hour, but the current pipeline writes a 21 GB uncompressed
intermediate file before quantizing. A streaming encode-and-quantize path would reduce
the index to approximately 1.8 GB at two-bit residuals and make the measurement
practical. [RESEARCH_LOG §20](RESEARCH_LOG.md) lists the remaining open items.

## License

Apache-2.0. See [LICENSE](LICENSE).
