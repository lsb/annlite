//! Query construction for FTS5.
//!
//! The query sets are plain whitespace-separated words, but handing them to `MATCH`
//! verbatim would measure the wrong thing. FTS5 treats a bare sequence of terms as an
//! implicit AND, so a 10-term query would return only documents containing all ten —
//! for the `random` family, essentially always zero rows, and the benchmark would
//! report microsecond latencies for an empty search. An explicit `OR` makes partial
//! matches candidates and lets BM25 do the ranking, which is the comparison the ANN
//! systems need: they also score every partial match rather than filtering.
//!
//! Terms are double-quoted so that anything FTS5 reads as syntax — `AND`, `OR`,
//! `NOT`, `NEAR`, a leading digit, a `*` — is taken as a string instead. The generated
//! vocabulary is pure `[a-z]+` and contains none of these, but a query builder that
//! only works on well-behaved input is a trap for the next corpus.
//!
//! What the quotes do *not* do is make the contents literal. FTS5 runs its tokenizer
//! inside a double-quoted string and the result is a **phrase**: `"load_module()."`
//! is the phrase `load module`, matching only documents where those two tokens are
//! adjacent and in that order. That is invisible on the word corpora, where every
//! whitespace-delimited chunk is one token and a phrase of one token is just a term.
//! It is not invisible on prose — which is what [`TermSplit`] exists to handle.

/// How a query string is cut into terms.
///
/// The split is a property of the query set, not a preference. Getting it wrong does
/// not merely change the numbers, it changes which query was asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TermSplit {
    /// Split on whitespace and keep everything else. The generated word corpora are
    /// pure `[a-z]+`, so this reproduces the query text exactly.
    Whitespace,
    /// Keep maximal runs of alphanumeric characters and drop the rest.
    ///
    /// A docstring is a sentence, and splitting it on whitespace would quote chunks
    /// like `load_module().` — which FTS5 reads as the *phrase* `load module` rather
    /// than as two independent terms. The query would then silently demand adjacency
    /// it has no reason to demand, and a document mentioning only `module` would stop
    /// being a candidate. Splitting on non-alphanumerics instead produces exactly the
    /// tokens the `unicode61` tokenizer put in the index, each free to match on its
    /// own, which is the comparison the ANN systems need.
    Alphanumeric,
}

/// Build an FTS5 `MATCH` expression that ORs the query's terms.
///
/// Returns `None` for a query with no terms, which has no meaningful `MATCH` form —
/// under [`TermSplit::Alphanumeric`] that includes a query of pure punctuation.
pub fn match_expression_with(text: &str, split: TermSplit) -> Option<String> {
    let raw: Vec<&str> = match split {
        TermSplit::Whitespace => text.split_whitespace().collect(),
        TermSplit::Alphanumeric => text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .collect(),
    };
    if raw.is_empty() {
        return None;
    }
    Some(
        raw.iter()
            .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" OR "),
    )
}

/// Whitespace-split form, which is what the word corpora use.
pub fn match_expression(text: &str) -> Option<String> {
    match_expression_with(text, TermSplit::Whitespace)
}

/// One query from `data/corpus/queries-*.jsonl` or `data/corpus/code-queries.jsonl`.
///
/// `k` is absent from the code query set — those queries carry no per-query result
/// depth, because every one of them has exactly one right answer — so it defaults to
/// zero and the grouping below collapses to a single bucket.
#[derive(serde::Deserialize, Clone, Debug)]
pub struct Query {
    pub qid: usize,
    pub kind: String,
    #[serde(default)]
    pub k: usize,
    pub text: String,
    #[serde(default)]
    pub source_doc: Option<usize>,
}

impl Query {
    /// Whether this query has a gold document, and so counts toward known-item
    /// quality.
    ///
    /// Tested on the presence of the gold id rather than on `kind`, because the kind
    /// names belong to a particular corpus generator ("known_item"/"random" for the
    /// word sets, "docstring" for the code set) while having an answer does not.
    pub fn is_known_item(&self) -> bool {
        self.source_doc.is_some()
    }
}

pub fn load(path: &std::path::Path) -> anyhow::Result<Vec<Query>> {
    use std::io::BufRead;
    let f = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("cannot read query set {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for line in std::io::BufReader::new(f).lines() {
        let line = line?;
        if !line.trim().is_empty() {
            out.push(serde_json::from_str(&line)?);
        }
    }
    Ok(out)
}
