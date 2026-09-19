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
| HTTP byte-range server with latency/throughput simulation | in progress |
| FTS5 baseline | in progress |
| Dense ANN (HNSW + PQ, 64 B/doc) | planned |
| DiskANN / Vamana page-local index | planned |
| Late interaction (fast-plaid style) | blocked — see below |
| WASM browser demo | planned |

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

## Quick start

```sh
apt-get install wamerican      # supplies /usr/share/dict/words
make corpora queries           # regenerate all benchmark data
make test                      # run the Rust test suite
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
crates/annlite-netsim    HTTP range server with latency/throughput simulation
data/corpus              generated (gitignored); rebuild with `make corpora`
bench/results            measurement output
```

## License

Apache-2.0. See [LICENSE](LICENSE).
