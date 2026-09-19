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
//! `NOT`, `NEAR`, a leading digit, a `*` — is taken as a literal string instead. The
//! generated vocabulary is pure `[a-z]+` and contains none of these, but a query
//! builder that only works on well-behaved input is a trap for the next corpus.

/// Build an FTS5 `MATCH` expression that ORs the query's terms.
///
/// Returns `None` for a query with no terms, which has no meaningful `MATCH` form.
pub fn match_expression(text: &str) -> Option<String> {
    let terms: Vec<String> = text
        .split_whitespace()
        .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
        .collect();
    if terms.is_empty() {
        None
    } else {
        Some(terms.join(" OR "))
    }
}

/// One query from `data/corpus/queries-*.jsonl`.
#[derive(serde::Deserialize, Clone, Debug)]
pub struct Query {
    pub qid: usize,
    pub kind: String,
    pub k: usize,
    pub text: String,
    #[serde(default)]
    pub source_doc: Option<usize>,
}

impl Query {
    pub fn is_known_item(&self) -> bool {
        self.kind == "known_item"
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
