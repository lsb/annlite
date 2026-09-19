# annlite

ANN extensions for SQLite — dense and late-interaction vector search that works from
a browser, reading a remote SQLite file over HTTP range requests.

The target is [sql.js-httpvfs](https://github.com/phiresky/sql.js-httpvfs)-style
deployment: a static SQLite file on a CDN, a WASM client, and no server-side search
infrastructure. That makes **HTTP round-trips the cost that matters**, not CPU. An
index that flies on a local SSD can be unusable over a cell link if its access
pattern scatters across pages, so everything here is measured in round-trips and
bytes fetched as well as milliseconds.

## Status

| Milestone | State |
|---|---|
| Deterministic corpora + query sets | **done** |
| HTTP byte-range server with latency/throughput simulation | **done** |
| FTS5 baseline (to 1M docs) | **done** |
| Product quantization (64 B/doc) | **done** |
| HNSW, validated against `hnswlib` | **done** |
| DiskANN / Vamana page-local index | **done** |
| WASM build + browser demo | **done** |
| Late interaction index | **done** (quality blocked — see below) |
| Million-document dense measurements | in progress |

### Headline results so far

* **Round-trips are the whole cost.** 32 sequential 4 KiB page fetches over
  satellite spend 20.1 s waiting and 55 ms transferring. Reducing bytes is worth
  little; reducing *dependent fetches* is worth almost everything.
* **FTS5 does not survive the network at 1M documents.** A single-term query touches
  689 distinct pages and reads 2.8 MB — about 48 s at a 70 ms RTT.
* **BM25, not the inverted index, is what scatters the reads.** Finding candidates
  costs 20 pages; scoring them costs 1,538, a 79x difference, because FTS5 does one
  random `%_docsize` lookup per match.
* **`VACUUM` turns 2,380 scattered requests into 133 contiguous ones** for the same
  query on the same pages — free, and the single biggest lever found so far.
* **PQ at 64 bytes is a candidate generator, not a ranker**: 99.2% of the exact
  top-10 lands in its top-100, but only 61% in its top-10.
* **Resident PQ codes cut pages per query by 6.6–18x at identical recall**, in
  exchange for one bulk download whose cost depends on corpus size.
* **The httpvfs client can defeat the whole exercise.** `sql.js-httpvfs` escalates
  its read-ahead past a megabyte and pulls a 5.9 MB database whole on the first
  query. Bounded Range requests against the flat record format move 9.1 KB per query
  instead of 512 KB — but in 69 requests rather than one, which is slower on a
  70 ms link. At small scale, downloading everything wins.

See [RESEARCH_LOG.md](RESEARCH_LOG.md) for methodology and full tables.

**Blocked:** late interaction needs the `tokenizer.json` for `LateOn-Code-edge`.
`huggingface.co` is unreachable from the build environment and no `tokenizer.json`
is committed. See [RESEARCH_LOG.md §3.2](RESEARCH_LOG.md).

## Models

Both encoders are committed to the repository. Their filenames do not indicate which
is which; they were identified from graph structure and verified by running them
(see [RESEARCH_LOG.md §2](RESEARCH_LOG.md)).

| File | Model | Params | Output | Use |
|---|---|---:|---|---|
| `model_qint8_arm64.onnx` | `all-MiniLM-L6-v2` | 22.6M | `[batch, seq, 384]` | dense — mean-pool + L2 normalise |
| `model_int8.onnx` | `lightonai/LateOn-Code-edge` | 17.0M | `[batch, seq, 48]` | late interaction — per-token, pre-normalised |

![the demo running in a browser](docs/demo.png)

## Quick start

```sh
apt-get install wamerican      # supplies /usr/share/dict/words
make corpora queries           # regenerate all benchmark data
make fts5                      # run the FTS5 baseline at all scales
make test                      # run the Rust test suite
make tokenizer-parity          # diff the Rust and Python tokenizers
make wasm-test                 # check the browser build against the native one
make demo && make demo-serve   # build and serve the browser demo
```

## Data is generated, not committed

Benchmark corpora are reproducible rather than stored. The 1M-document corpus is
~454 MB and regenerates byte-identically in 5.4 seconds, so committing it would
dominate the repository for no benefit.

Determinism is a property of the generator, not of the machine. Every random choice
flows through ChaCha8 seeded by `SHA-256("annlite/v1/" ‖ domain ‖ seed)`, which is
fixed by the algorithm and so reproduces on any platform. `shuf` is deliberately not
used: `shuf --random-source` is stable only within one coreutils build, and plain
`shuf` also depends on locale collation.

| Corpus | Documents | Bytes | Generation |
|---|---:|---:|---:|
| `docs-100.txt` | 100 | 45 KB | <0.01 s |
| `docs-10k.txt` | 10,000 | 4.5 MB | 0.05 s |
| `docs-1m.txt` | 1,000,000 | 454 MB | 5.43 s |

Each corpus is a byte-exact prefix of the next, so scale curves describe one growing
collection rather than three unrelated samples. Digests land in
`data/corpus/*.manifest.json`; the same values are quoted in
[RESEARCH_LOG.md §4.4](RESEARCH_LOG.md).

## Layout

```
crates/annlite-core      index algorithms: PQ, HNSW, Vamana/DiskANN, late interaction
crates/annlite-corpus    deterministic corpus and query-set generation
crates/annlite-fts5      FTS5 baseline: build cost, latency, quality, page access
crates/annlite-netsim    HTTP range server with latency/throughput simulation
data/corpus              generated (gitignored); rebuild with `make corpora`
bench/results            measurement output
```

## License

Apache-2.0. See [LICENSE](LICENSE).
