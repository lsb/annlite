//! Vocabulary extraction from a system dictionary.

use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::path::Path;

/// Normalise a system word list into the benchmark vocabulary.
///
/// `/usr/share/dict/words` ships proper nouns and possessives ("Aaron", "Aaron's").
/// We lowercase, keep only pure `[a-z]+` runs, and deduplicate into sorted order.
/// Sorting before shuffling is what makes the result independent of the input file's
/// own ordering and of locale collation, so a dictionary from a different
/// distribution yields the same vocabulary as long as it holds the same words.
pub fn build(dict: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(dict)
        .with_context(|| format!("reading dictionary {}", dict.display()))?;
    let mut set = BTreeSet::new();
    for line in text.lines() {
        let w = line.trim().to_lowercase();
        if !w.is_empty() && w.chars().all(|c| c.is_ascii_lowercase()) {
            set.insert(w);
        }
    }
    anyhow::ensure!(!set.is_empty(), "dictionary {} yielded no usable words", dict.display());
    Ok(set.into_iter().collect())
}
