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

.PHONY: all corpora queries clean clean-data help check-dict test fts5

help:
	@echo "annlite benchmark pipeline"
	@echo "  make corpora    generate document corpora at all scales (100 / 10k / 1M)"
	@echo "  make queries    generate query sets for each corpus scale"
	@echo "  make fts5       run the FTS5 baseline at all scales (results in bench/results)"
	@echo "  make test       run the Rust test suite"
	@echo "  make clean-data remove generated corpora (they regenerate byte-identically)"
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
