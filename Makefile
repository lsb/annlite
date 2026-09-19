# annlite -- reproducible benchmark pipeline.
#
# Generated corpora are not committed; they are large (the 1M-document corpus is
# ~450 MB) and exactly reproducible from the seeds baked into annlite-corpus.
# Every target below rebuilds byte-identical output on any platform. Digests are
# recorded in data/corpus/*.manifest.json and quoted in RESEARCH_LOG.md.

CARGO     ?= cargo
PYTHON    ?= .venv/bin/python
DICT      ?= /usr/share/dict/words
DATA      := data
CORPUS    := $(DATA)/corpus
VOCAB     := $(CORPUS)/vocab.txt
CORPUS_BIN := target/release/annlite-corpus

SCALES    := 100 10k 1m
100_N     := 100
10k_N     := 10000
1m_N      := 1000000

.DEFAULT_GOAL := help
.PHONY: all corpora queries clean clean-data help check-dict test matrix fts5

help:
	@echo "annlite benchmark pipeline"
	@echo ""
	@echo " data"
	@echo "  make corpora        generate word corpora at all scales (100 / 10k / 1M)"
	@echo "  make queries        generate query sets for each corpus scale"
	@echo "  make code-corpus    build the docstring-to-code corpus from the stdlib"
	@echo ""
	@echo " measurement"
	@echo "  make fts5           FTS5 baseline at all scales"
	@echo "  make code-eval      BM25 vs dense vs late interaction on the code corpus"
	@echo "  make fts5-code      FTS5 on the code corpus: pages, requests, quality"
	@echo "  make ann-code       dense index on the code corpus (embeds first)"
	@echo "  make tri-code       join both into bench/results/tri-code.jsonl"
	@echo "  make late-pages     late interaction in SQLite: pages, requests, hops per stage"
	@echo "  make matrix         join every result into docs/RESULTS.md"
	@echo ""
	@echo " browser"
	@echo "  make wasm           build the WebAssembly module"
	@echo "  make demo           build the browser demo"
	@echo "  make demo-serve     serve it through the network simulator"
	@echo ""
	@echo " checks"
	@echo "  make test              Rust test suite"
	@echo "  make tokenizer-parity  diff the Rust and Python tokenizers"
	@echo "  make wasm-test         check the browser build against the native one"
	@echo "  make clean-data        remove generated corpora (they rebuild identically)"
	@echo ""
	@echo "Individual scales: make corpus-100 corpus-10k corpus-1m fts5-100 fts5-10k fts5-1m"

all: corpora queries

# The vocabulary is derived from the system dictionary; on Debian/Ubuntu that is
# the 'wamerican' package. Fail with the fix rather than with a missing-file error.
check-dict:
	@test -r $(DICT) || { \
	  echo "error: $(DICT) not found."; \
	  echo "  Debian/Ubuntu: apt-get install wamerican"; \
	  echo "  macOS already ships /usr/share/dict/words"; \
	  exit 1; }

$(CORPUS_BIN): $(shell find crates/annlite-corpus/src -name '*.rs' 2>/dev/null)
	$(CARGO) build --release -p annlite-corpus

$(VOCAB): $(CORPUS_BIN) check-dict
	@mkdir -p $(CORPUS)
	$(CORPUS_BIN) vocab --dict $(DICT) --out $@

vocab: $(VOCAB)

define CORPUS_RULE
corpus-$(1): $$(CORPUS)/docs-$(1).txt
$$(CORPUS)/docs-$(1).txt: $$(VOCAB) $$(CORPUS_BIN)
	$$(CORPUS_BIN) docs --vocab $$(VOCAB) --n $$($(1)_N) --out $$@

queries-$(1): $$(CORPUS)/queries-$(1).jsonl
$$(CORPUS)/queries-$(1).jsonl: $$(VOCAB) $$(CORPUS_BIN)
	$$(CORPUS_BIN) queries --vocab $$(VOCAB) --corpus-size $$($(1)_N) --per-k 100 --out $$@
endef
$(foreach s,$(SCALES),$(eval $(call CORPUS_RULE,$(s))))

# The FTS5 baseline every ANN index is compared against. Each scale is its own
# target because the 1M run takes minutes and is worth starting on its own.
FTS5_BIN := target/release/annlite-fts5

$(FTS5_BIN): $(shell find crates/annlite-fts5/src -name '*.rs' 2>/dev/null)
	$(CARGO) build --release -p annlite-fts5

define FTS5_RULE
fts5-$(1): $$(FTS5_BIN) $$(CORPUS)/docs-$(1).txt $$(CORPUS)/queries-$(1).jsonl
	$$(FTS5_BIN) --scale $(1)
endef
$(foreach s,$(SCALES),$(eval $(call FTS5_RULE,$(s))))

fts5: $(foreach s,$(SCALES),fts5-$(s))

corpora: $(foreach s,$(SCALES),corpus-$(s))
queries: $(foreach s,$(SCALES),queries-$(s))

test:
	$(CARGO) test --release

clean-data:
	rm -rf $(CORPUS)

clean: clean-data
	$(CARGO) clean

# --- WebAssembly -----------------------------------------------------------
WASM_TARGET := wasm32-unknown-unknown
WASM_OUT    := web/pkg

.PHONY: wasm wasm-test tokenizer-parity

wasm:
	$(CARGO) build --release -p annlite-wasm --target $(WASM_TARGET)
	@command -v wasm-bindgen >/dev/null 2>&1 || { \
	  echo "error: wasm-bindgen not found."; \
	  echo "  cargo install wasm-bindgen-cli --version $$(grep -m1 -A1 '^name = \"wasm-bindgen\"' Cargo.lock | grep version | cut -d'\"' -f2)"; \
	  exit 1; }
	wasm-bindgen --target nodejs --out-dir $(WASM_OUT) \
	  target/$(WASM_TARGET)/release/annlite_wasm.wasm
	@echo "wasm -> $(WASM_OUT)"

# Proves the browser build returns exactly what the native build returns, using a
# fixture that carries a real index and the native answer for the same bytes.
wasm-test: wasm web/testfixture.json
	node web/test/parity.js

web/testfixture.json: $(VOCAB)
	$(CARGO) run --release -p annlite-sqlite --example wasm_fixture

# The corpus is tokenized offline in Python and queries are tokenized in the
# browser in Rust. If they ever disagree, every similarity is computed between
# vectors from two different input distributions, so the two are diffed directly.
tokenizer-parity: $(CORPUS)/docs-10k.txt
	@$(CARGO) build --release -p annlite-core --example tokenize_dump 2>/dev/null
	@$(PYTHON) tools/analyze/tokenizer_sample.py > /tmp/annlite-tok-sample.txt
	@./target/release/examples/tokenize_dump models/tokenizers/bert-base-uncased-vocab.txt \
	   < /tmp/annlite-tok-sample.txt > /tmp/annlite-tok-rust.txt
	@$(PYTHON) tools/analyze/tokenizer_dump.py < /tmp/annlite-tok-sample.txt > /tmp/annlite-tok-py.txt
	@diff -q /tmp/annlite-tok-rust.txt /tmp/annlite-tok-py.txt >/dev/null \
	  && echo "tokenizers agree on $$(wc -l < /tmp/annlite-tok-sample.txt) lines" \
	  || { echo "TOKENIZERS DISAGREE:"; diff /tmp/annlite-tok-rust.txt /tmp/annlite-tok-py.txt | head; exit 1; }

# --- Browser demo ----------------------------------------------------------
DEMO_DIR  := web/demo
DEMO_DOCS ?= 2000

.PHONY: demo demo-serve

demo: wasm $(CORPUS)/docs-10k.txt
	$(CARGO) run --release -p annlite-sqlite --example build_demo -- $(DEMO_DOCS) $(DEMO_DIR)/annlite-demo.db
	wasm-bindgen --target web --out-dir $(DEMO_DIR)/vendor/annlite \
	  target/$(WASM_TARGET)/release/annlite_wasm.wasm
	$(PYTHON) tools/analyze/demo_queries.py
	@echo "demo built -- run 'make demo-serve' and open http://127.0.0.1:8099/index.html"

# Served through the network simulator so the page can report real HTTP costs from
# the server's own counters rather than guessing at them.
demo-serve:
	./target/release/annlite-netsim --root $(DEMO_DIR) --addr 127.0.0.1:8099 \
	  --profile $(or $(PROFILE),ideal) --log /tmp/annlite-netsim.jsonl

# --- Code-search corpus ----------------------------------------------------
# A code model cannot be evaluated on bags of dictionary words, so this builds the
# docstring-to-code benchmark from the local Python standard library. Ground truth
# is the corpus construction itself: a docstring's own function.
.PHONY: code-corpus code-eval fts5-code code-dense ann-code tri-code late-pages

code-corpus: $(CORPUS)/code-docs.txt
$(CORPUS)/code-docs.txt:
	$(PYTHON) -m tools.corpus.code --out-dir $(CORPUS)

code-eval: code-corpus
	$(PYTHON) tools/analyze/code_eval.py $(or $(NQ),500)
	$(CARGO) run --release -p annlite-core --example late_eval

# The same corpus measured the way the word corpora are, so the three retrieval
# systems can be compared on cost as well as on quality. NQ is fixed at 500 to match
# bench/results/code-eval.json; raising it makes nothing comparable.
CODE_NQ   ?= 500
EMBED     := $(DATA)/embeddings

fts5-code: $(FTS5_BIN) $(CORPUS)/code-docs.txt
	$(FTS5_BIN) --docs $(CORPUS)/code-docs.txt --queries $(CORPUS)/code-queries.jsonl \
	  --name code --sample $(CODE_NQ)

# Documents carry escaped newlines; without --unescape the encoder sees a backslash
# and an `n` at every line break.
$(EMBED)/code-dense.f32: $(CORPUS)/code-docs.txt
	$(PYTHON) -m tools.embed --input $< --out $@ --unescape
$(EMBED)/code-dense-q.f32: $(CORPUS)/code-queries.jsonl
	$(PYTHON) -m tools.embed --input $< --out $@

code-dense: $(EMBED)/code-dense.f32 $(EMBED)/code-dense-q.f32

# Parameters are scaled to 3,366 documents rather than inherited from the 10k-1M
# runs: R=24 keeps the record at 130 bytes, and PQ trains on the whole corpus because
# at this size a sample buys nothing.
ann-code: code-dense
	$(CARGO) build --release -p annlite-sqlite
	./target/release/annlite-sqlite --docs $(EMBED)/code-dense.f32 \
	  --queries $(EMBED)/code-dense-q.f32 --gold $(CORPUS)/code-queries.jsonl \
	  --scale code --dim 384 --m 32 --r 24 --l-build 64 --pq-train 3366 \
	  --n-queries $(CODE_NQ) --k 100 --search-l 100,128,256 --beams 4 \
	  --orderings identity,bfs
	./target/release/annlite-sqlite --docs $(EMBED)/code-dense.f32 \
	  --queries $(EMBED)/code-dense-q.f32 --gold $(CORPUS)/code-queries.jsonl \
	  --scale code-m64 --dim 384 --m 64 --r 24 --l-build 64 --pq-train 3366 \
	  --n-queries $(CODE_NQ) --k 100 --search-l 128 --beams 4 --orderings bfs

# Flattens both into one row per (system, configuration). Rows written by other
# systems are preserved, so the three can be measured independently.
tri-code:
	$(PYTHON) tools/analyze/tri_code.py

# Late interaction with page accounting, so it can be set against FTS5 and dense on
# the axis the project is about rather than on quality alone. Needs the multi-vector
# embeddings, which `code-eval` writes.
late-pages: data/embeddings/code-late.f32
	$(CARGO) run --release -p annlite-sqlite --example late_pages -- $(or $(NQ),500)

data/embeddings/code-late.f32:
	$(MAKE) code-eval

# --- Reporting -------------------------------------------------------------
# Joins the separate benchmark outputs and converts measured counts into seconds
# per network profile. Safe to run with only some benchmarks completed; missing
# sections are simply omitted.
matrix:
	$(PYTHON) tools/analyze/matrix.py
