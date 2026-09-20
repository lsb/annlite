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
EMBED     := $(DATA)/embeddings
VOCAB     := $(CORPUS)/vocab.txt
CORPUS_BIN := target/release/annlite-corpus

SCALES    := 100 10k 1m
100_N     := 100
10k_N     := 10000
1m_N      := 1000000

.DEFAULT_GOAL := help
.PHONY: all corpora queries clean clean-data help check-dict test matrix fts5 venv

help:
	@echo "annlite benchmark pipeline"
	@echo ""
	@echo " setup"
	@echo "  make venv           create .venv and install the pinned Python deps"
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
	@echo "  make ann            dense sweep at 10k / 100k / 1M (embeds first)"
	@echo "  make ann-gold       dense scored against gold, comparable with the others"
	@echo "  make tri-code       join both into bench/results/tri-code.jsonl"
	@echo "  make late-pages     late interaction in SQLite: pages, requests, hops per stage"
	@echo "  make late-words     late interaction on a word corpus (SCALE=10k)"
	@echo "  make residual       residual quantization: storage against quality"
	@echo "  make matrix         join every result into docs/RESULTS.md"
	@echo ""
	@echo " browser"
	@echo "  make sqlite-wasm    build SQLite to WASM with the bounded-range VFS"
	@echo "  make range-demo     bounded ranges vs sql.js-httpvfs on one database"
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

# --- Python environment ----------------------------------------------------
# The Python half of the pipeline (both encoders, the code-corpus builder, every
# reporting script) runs out of .venv rather than the system interpreter, because
# onnxruntime's version decides the embeddings and the embeddings decide every
# quality number in RESEARCH_LOG.md. requirements.txt pins what bench/results was
# measured with.
VENV_STAMP := .venv/.stamp

venv: $(VENV_STAMP)
$(VENV_STAMP): requirements.txt
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@touch $@

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

# Both halves of the pipeline. The Python tests cover the network cost model, the
# tokenizer, and the file-naming conventions the Rust readers depend on by name.
test: $(VENV_STAMP)
	$(CARGO) test --release
	$(PYTHON) -m pytest tools/tests -q

clean-data:
	rm -rf $(CORPUS)

clean: clean-data
	$(CARGO) clean

# --- WebAssembly -----------------------------------------------------------
WASM_TARGET := wasm32-unknown-unknown
WASM_OUT    := web/pkg

.PHONY: wasm wasm-test tokenizer-parity check-wasm-target

# Without the target installed, cargo fails with a raw "can't find crate for `core`"
# from deep inside a dependency, which says nothing about the actual cause.
check-wasm-target:
	@rustup target list --installed 2>/dev/null | grep -qx '$(WASM_TARGET)' || { \
	  echo "error: the $(WASM_TARGET) target is not installed."; \
	  echo "  rustup target add $(WASM_TARGET)"; \
	  exit 1; }

wasm: check-wasm-target
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

# --- Dense sweep on the word corpora ---------------------------------------
# These produced bench/results/ann-{10k,100k,1m}.jsonl, the evidence behind the
# headline cost numbers in README.md. Until now they had no committed command and
# were run by hand -- which is the one thing a file like this exists to prevent.
# Every parameter below is read off the `meta` record of the file it reproduces.
# `--k 10` matters more than it looks: the search widens L to at least k, so leaving
# the CLI default of 100 turns the L=32 half of the sweep into an L=100 one and
# changes every page, hop and recall figure it produces.
SQLITE_BIN := target/release/annlite-sqlite

$(SQLITE_BIN): $(shell find crates/annlite-sqlite/src -name '*.rs' 2>/dev/null)
	$(CARGO) build --release -p annlite-sqlite

10k_BATCH   := 1024
10k_PQTRAIN := 10000
10k_BEAMS   := 1,4,16
10k_NQ      := 200
100k_BATCH   := 4096
100k_PQTRAIN := 100000
100k_BEAMS   := 4,16
100k_NQ      := 100
1m_BATCH   := 8192
1m_PQTRAIN := 100000
1m_BEAMS   := 4,16
1m_NQ      := 100

# The 100k corpus is the first 100,000 lines of docs-1m.txt. The corpora are
# byte-exact prefixes of one another (RESEARCH_LOG 4.3), so this is the same
# collection truncated, not a different sample.
$(EMBED)/docs-10k.f32: $(CORPUS)/docs-10k.txt $(VENV_STAMP)
	$(PYTHON) -m tools.embed --input $< --out $@
$(EMBED)/docs-100k.f32: $(CORPUS)/docs-1m.txt $(VENV_STAMP)
	$(PYTHON) -m tools.embed --input $< --out $@ --limit 100000
$(EMBED)/docs-1m.f32: $(CORPUS)/docs-1m.txt $(VENV_STAMP)
	$(PYTHON) -m tools.embed --input $< --out $@

define QEMBED_RULE
$$(EMBED)/queries-$(1).f32: $$(CORPUS)/queries-$(1).jsonl $$(VENV_STAMP)
	$$(PYTHON) -m tools.embed --input $$< --out $$@
endef
$(foreach s,10k 100k 1m,$(eval $(call QEMBED_RULE,$(s))))
$(CORPUS)/queries-100k.jsonl: $(VOCAB) $(CORPUS_BIN)
	$(CORPUS_BIN) queries --vocab $(VOCAB) --corpus-size 100000 --per-k 100 --out $@

define ANN_RULE
ann-$(1): $$(SQLITE_BIN) $$(EMBED)/docs-$(1).f32 $$(EMBED)/queries-$(1).f32
	$$(SQLITE_BIN) --docs $$(EMBED)/docs-$(1).f32 --queries $$(EMBED)/queries-$(1).f32 \
	  --scale $(1) --dim 384 --m 64 --r 32 --alpha 1.1 --l-build 100 \
	  --build-batch $$($(1)_BATCH) --pq-train $$($(1)_PQTRAIN) \
	  --n-queries $$($(1)_NQ) --k 10 --search-l 32,128 --beams $$($(1)_BEAMS) \
	  --orderings identity,bfs,cluster
endef
$(foreach s,10k 100k 1m,$(eval $(call ANN_RULE,$(s))))

.PHONY: ann ann-10k ann-100k ann-1m ann-gold
ann: ann-10k ann-100k ann-1m

# The sweeps above score recall@10 against exact brute force, which is the right
# ground truth for an ANN index but is not the metric FTS5 and late interaction
# report. This run scores the same index against the query set's gold document, so
# all three systems can be read on one quality scale at 10k.
ann-gold: $(SQLITE_BIN) $(EMBED)/docs-$(SCALE).f32 $(EMBED)/queries-$(SCALE).f32
	$(SQLITE_BIN) --docs $(EMBED)/docs-$(SCALE).f32 --queries $(EMBED)/queries-$(SCALE).f32 \
	  --gold $(CORPUS)/queries-$(SCALE).jsonl \
	  --scale $(SCALE)-gold --dim 384 --m 64 --r 32 --alpha 1.1 --l-build 100 \
	  --build-batch $($(SCALE)_BATCH) --pq-train $($(SCALE)_PQTRAIN) \
	  --n-queries 500 --search-l 128 --beams 4 --orderings bfs

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
$(CORPUS)/code-docs.txt: $(VENV_STAMP)
	$(PYTHON) -m tools.corpus.code --out-dir $(CORPUS)

code-eval: code-corpus $(VENV_STAMP)
	$(PYTHON) tools/analyze/code_eval.py $(or $(NQ),500)
	$(CARGO) run --release -p annlite-core --example late_eval

# The same corpus measured the way the word corpora are, so the three retrieval
# systems can be compared on cost as well as on quality. NQ is fixed at 500 to match
# bench/results/code-eval.json; raising it makes nothing comparable.
CODE_NQ   ?= 500

fts5-code: $(FTS5_BIN) $(CORPUS)/code-docs.txt
	$(FTS5_BIN) --docs $(CORPUS)/code-docs.txt --queries $(CORPUS)/code-queries.jsonl \
	  --name code --sample $(CODE_NQ)

# Documents carry escaped newlines; without --unescape the encoder sees a backslash
# and an `n` at every line break.
$(EMBED)/code-dense.f32: $(CORPUS)/code-docs.txt $(VENV_STAMP)
	$(PYTHON) -m tools.embed --input $< --out $@ --unescape
$(EMBED)/code-dense-q.f32: $(CORPUS)/code-queries.jsonl $(VENV_STAMP)
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
tri-code: $(VENV_STAMP)
	$(PYTHON) tools/analyze/tri_code.py

# Late interaction with page accounting, so it can be set against FTS5 and dense on
# the axis the project is about rather than on quality alone. Needs the multi-vector
# embeddings, which `code-eval` writes.
late-pages: data/embeddings/code-late.f32
	$(CARGO) run --release -p annlite-sqlite --example late_pages -- $(or $(NQ),500)

data/embeddings/code-late.f32:
	$(MAKE) code-eval

# Late interaction on a word corpus, so all three systems can be read side by side
# at the scales the FTS5 and dense baselines already cover. The first 500 queries of
# every word query set are known-item with a gold document, which is the same set the
# dense sweep scores against.
#
#   make late-words SCALE=10k
SCALE ?= 10k

.PHONY: late-words residual

late-words: $(VENV_STAMP) $(CORPUS)/docs-$(SCALE).txt $(CORPUS)/queries-$(SCALE).jsonl
	$(PYTHON) -m tools.embed.late_corpus \
	  --input $(CORPUS)/docs-$(SCALE).txt --out $(EMBED)/words-$(SCALE)-late.f32
	$(PYTHON) -m tools.embed.late_corpus \
	  --input $(CORPUS)/queries-$(SCALE).jsonl --out $(EMBED)/words-$(SCALE)-late-q.f32 \
	  --lengths $(EMBED)/words-$(SCALE)-late-qlengths.i32 --limit 500
	$(CARGO) run --release -p annlite-sqlite --example late_pages -- \
	  --corpus words-$(SCALE) --k 1024

# Storage against quality for late interaction, the gap section 15.2 left open.
# Quality and bytes only -- nothing here is timed, so it is safe under load.
residual: data/embeddings/code-late.f32
	$(CARGO) run --release -p annlite-core --example residual_eval

# --- SQLite compiled to WASM ------------------------------------------------
# Section 13.2 found that sql.js-httpvfs escalates its read-ahead until it has
# fetched the whole database, so the index's access pattern never reaches the wire
# and section 13.4's crossover question could not be answered through a real SQLite
# client. This builds one that fetches exactly what SQLite asks for.
#
# The amalgamation is the one libsqlite3-sys already vendored for the native
# benchmarks, so the browser and native measurements run the same engine (3.46.0)
# and any difference between them is the VFS rather than SQLite.
EMSDK      ?= $(HOME)/emsdk
SQLITE_AMALG := $(firstword $(wildcard $(HOME)/.cargo/registry/src/*/libsqlite3-sys-0.30.1/sqlite3))
WASM_SQLITE := web/sqlite-wasm/annlite-sqlite.wasm

.PHONY: sqlite-wasm check-emsdk range-demo

# emcc is invoked by path rather than through emsdk_env.sh: that script is written
# for bash, and make runs recipes under /bin/sh, where sourcing it silently fails to
# set PATH and the build dies with "emcc: not found" several lines later.
EMCC ?= $(EMSDK)/upstream/emscripten/emcc

check-emsdk:
	@test -x "$(EMCC)" || { \
	  echo "error: no emcc at $(EMCC)."; \
	  echo "  git clone https://github.com/emscripten-core/emsdk.git ~/emsdk"; \
	  echo "  cd ~/emsdk && ./emsdk install latest && ./emsdk activate latest"; \
	  echo "  or point at an existing one: make sqlite-wasm EMSDK=/path/to/emsdk"; \
	  echo "  or name the compiler directly:  make sqlite-wasm EMCC=/path/to/emcc"; \
	  exit 1; }
	@test -n "$(SQLITE_AMALG)" || { \
	  echo "error: no bundled SQLite amalgamation found."; \
	  echo "  cargo fetch     # vendors libsqlite3-sys, whose sqlite3.c this builds"; \
	  exit 1; }

sqlite-wasm: $(WASM_SQLITE)
$(WASM_SQLITE): web/sqlite-wasm/vfs_httprange.c | check-emsdk
	@echo "building SQLite $$(grep -m1 '\#define SQLITE_VERSION ' $(SQLITE_AMALG)/sqlite3.h \
	  | sed 's/.*"\(.*\)".*/\1/') to WASM"
	cd web/sqlite-wasm && "$(abspath $(EMCC))" -O2 \
	  -I"$(abspath $(SQLITE_AMALG))" "$(abspath $(SQLITE_AMALG))/sqlite3.c" vfs_httprange.c \
	  -DSQLITE_ENABLE_FTS5 -DSQLITE_OMIT_LOAD_EXTENSION -DSQLITE_THREADSAFE=0 \
	  -DSQLITE_DEFAULT_MEMSTATUS=0 -DSQLITE_OMIT_DEPRECATED -DSQLITE_ENABLE_DBSTAT_VTAB \
	  -sASYNCIFY=1 -sASYNCIFY_STACK_SIZE=16384 \
	  '-sASYNCIFY_EXPORTS=["sqlite3_open_v2","sqlite3_prepare_v2","sqlite3_step","sqlite3_exec","sqlite3_close_v2","sqlite3_finalize","annlite_register_vfs"]' \
	  '-sEXPORTED_FUNCTIONS=["_malloc","_free","_sqlite3_open_v2","_sqlite3_prepare_v2","_sqlite3_step","_sqlite3_column_int","_sqlite3_column_int64","_sqlite3_column_text","_sqlite3_column_count","_sqlite3_finalize","_sqlite3_close_v2","_sqlite3_errmsg","_sqlite3_exec","_annlite_register_vfs","_annlite_vfs_name","_annlite_stats_reset","_annlite_stat_reads","_annlite_stat_requests","_annlite_stat_pages","_annlite_stat_bytes"]' \
	  '-sEXPORTED_RUNTIME_METHODS=["ccall","cwrap","UTF8ToString","stringToUTF8","getValue","setValue","HEAPU8"]' \
	  -sALLOW_MEMORY_GROWTH=1 -sMODULARIZE=1 -sEXPORT_NAME=createAnnliteSqlite \
	  -sENVIRONMENT=node,web -sSTACK_SIZE=1048576 \
	  -o annlite-sqlite.js
	@echo "-> $(WASM_SQLITE)"

# Both clients against the same database, same query: what read-ahead costs.
range-demo: sqlite-wasm
	$(PYTHON) tools/analyze/range_compare.py

# --- Reporting -------------------------------------------------------------
# Joins the separate benchmark outputs and converts measured counts into seconds
# per network profile. Safe to run with only some benchmarks completed; missing
# sections are simply omitted.
matrix: $(VENV_STAMP)
	$(PYTHON) tools/analyze/matrix.py

# Seconds per query against requests in flight, for each system. Reads only the
# committed measurement files, so it needs no benchmark re-run.
.PHONY: concurrency
concurrency: $(VENV_STAMP)
	@$(PYTHON) tools/analyze/concurrency.py lte
	@echo
	@$(PYTHON) tools/analyze/concurrency.py satellite
