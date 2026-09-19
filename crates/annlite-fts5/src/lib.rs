//! FTS5 baseline for annlite.
//!
//! Every ANN index in this project is compared against SQLite's own full-text search,
//! so this crate measures FTS5 the way the ANN indexes will be measured: build cost,
//! query latency, retrieval quality, and — the number that decides whether an index is
//! usable from a browser — how many distinct database pages a single query touches.
//! See [`vfs`] for how the page counts are obtained and what they do and do not mean.

pub mod cpu;
pub mod db;
pub mod index;
pub mod metrics;
pub mod pages;
pub mod query;
pub mod vfs;

/// The three corpus scales, as named in `data/corpus/`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scale {
    pub name: &'static str,
    pub n_docs: usize,
}

pub const SCALES: [Scale; 3] = [
    Scale { name: "100", n_docs: 100 },
    Scale { name: "10k", n_docs: 10_000 },
    Scale { name: "1m", n_docs: 1_000_000 },
];

pub fn scale_by_name(name: &str) -> Option<Scale> {
    SCALES.iter().copied().find(|s| s.name == name)
}

/// The conventions a corpus pair follows.
///
/// These are not tuning knobs: reading a corpus with the wrong one measures a
/// different collection. The generated word corpora store one literal document per
/// line and ask bare `[a-z]+` terms; the docstring-to-code corpus escapes the
/// newlines inside a function so a document still occupies one line, and its queries
/// are English sentences. Both halves of each convention travel together, so they
/// are one setting rather than two that can be mismatched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CorpusFormat {
    /// `docs-*.txt` / `queries-*.jsonl`: literal lines, whitespace-separated terms.
    Words,
    /// `code-docs.txt` / `code-queries.jsonl`: backslash-escaped lines, prose queries
    /// split on non-alphanumerics.
    Code,
}

impl CorpusFormat {
    pub fn parse(s: &str) -> Option<CorpusFormat> {
        match s.to_lowercase().as_str() {
            "words" => Some(CorpusFormat::Words),
            "code" => Some(CorpusFormat::Code),
            _ => None,
        }
    }

    pub fn term_split(self) -> query::TermSplit {
        match self {
            CorpusFormat::Words => query::TermSplit::Whitespace,
            CorpusFormat::Code => query::TermSplit::Alphanumeric,
        }
    }

    /// Whether a document line carries `\n` and `\\` escapes that must be undone
    /// before indexing.
    pub fn escaped_lines(self) -> bool {
        self == CorpusFormat::Code
    }
}

/// Where one measurement run reads its corpus and writes its results.
///
/// A named scale and an arbitrary file pair differ only in how this is filled in,
/// which is what keeps the page counter, the cache-miss cross-check and the
/// build/optimize/vacuum phases on a single code path.
#[derive(Clone, Debug)]
pub struct Corpus {
    pub name: String,
    pub docs: std::path::PathBuf,
    pub queries: std::path::PathBuf,
    pub manifest: std::path::PathBuf,
    pub db: std::path::PathBuf,
    pub format: CorpusFormat,
    /// Document count the generator promises, when the corpus is one of the named
    /// scales. `None` for an arbitrary file, where the build reports what it found.
    pub n_docs_expected: Option<usize>,
}

impl Corpus {
    /// The manifest a corpus generator writes beside its documents, if it wrote one.
    fn manifest_for(docs: &std::path::Path) -> std::path::PathBuf {
        docs.with_extension("manifest.json")
    }

    pub fn scale(scale: Scale, data_dir: &std::path::Path) -> Corpus {
        let docs = data_dir.join(format!("corpus/docs-{}.txt", scale.name));
        Corpus {
            manifest: Self::manifest_for(&docs),
            docs,
            queries: data_dir.join(format!("corpus/queries-{}.jsonl", scale.name)),
            db: data_dir.join(format!("db/fts5-{}.db", scale.name)),
            name: scale.name.to_string(),
            format: CorpusFormat::Words,
            n_docs_expected: Some(scale.n_docs),
        }
    }

    pub fn files(
        name: &str,
        docs: std::path::PathBuf,
        queries: std::path::PathBuf,
        data_dir: &std::path::Path,
    ) -> Corpus {
        Corpus {
            manifest: Self::manifest_for(&docs),
            docs,
            queries,
            db: data_dir.join(format!("db/fts5-{name}.db")),
            name: name.to_string(),
            format: CorpusFormat::Code,
            n_docs_expected: None,
        }
    }
}
