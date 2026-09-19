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

.PHONY: all corpora queries clean clean-data help check-dict test

help:
	@echo "annlite benchmark pipeline"
	@echo "  make corpora    generate document corpora at all scales (100 / 10k / 1M)"
	@echo "  make queries    generate query sets for each corpus scale"
	@echo "  make test       run the Rust test suite"
	@echo "  make clean-data remove generated corpora (they regenerate byte-identically)"
	@echo ""
	@echo "Individual scales: make corpus-100 corpus-10k corpus-1m"

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

corpora: $(foreach s,$(SCALES),corpus-$(s))
queries: $(foreach s,$(SCALES),queries-$(s))

test:
	$(CARGO) test --release

clean-data:
	rm -rf $(CORPUS)

clean: clean-data
	$(CARGO) clean
