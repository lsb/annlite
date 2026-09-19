//! Query-set construction.
//!
//! A corpus of shuffled dictionary words has no topics, so "what is a sensible
//! query?" needs an answer that does not smuggle in semantics that the collection
//! does not contain. Two families are generated, and they measure different things:
//!
//! * **`known_item`** — take `k` words from one known document. That document is a
//!   ground-truth answer that exists by construction, so the query measures whether
//!   a system can find a specific document from a fragment of it. As `k` falls from
//!   10 to 1 the query gets harder in a controlled way, because fewer words means
//!   more documents share the query's full term set.
//! * **`random`** — draw `k` words from the vocabulary independently of any document.
//!   Usually no document contains all of them, so this measures ranking quality over
//!   partial matches rather than known-item lookup, and it is the case where a dense
//!   or late-interaction system can behave differently from lexical matching.
//!
//! Neither family needs human relevance judgments. The gold ranking for any query is
//! whatever exact brute-force scoring produces under the system's own scoring
//! function, so recall@k is always measured against an exact search rather than
//! against an opinion. `known_item` additionally carries a single unambiguously
//! correct document id, which is what makes MRR meaningful.

use crate::{docs, rng};
use anyhow::Result;
use serde::Serialize;
use std::io::Write;

#[derive(Serialize)]
pub struct Query {
    pub qid: usize,
    /// `known_item` or `random`.
    pub kind: &'static str,
    /// Number of query terms.
    pub k: usize,
    pub text: String,
    /// For `known_item`, the document the terms were drawn from; `null` for `random`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_doc: Option<usize>,
}

/// Term counts probed by the query set, from single-word up to a fifth of a document.
pub const TERM_COUNTS: [usize; 5] = [1, 2, 3, 5, 10];

/// Build `per_k` queries of each kind at each term count in [`TERM_COUNTS`].
///
/// `corpus_size` bounds which documents `known_item` queries may reference, so a
/// query set built for the 10k corpus never points at a document only the 1M corpus
/// contains.
pub fn generate<W: Write>(
    vocab: &[String],
    corpus_size: usize,
    per_k: usize,
    out: &mut W,
) -> Result<usize> {
    let mut qid = 0usize;

    for &k in TERM_COUNTS.iter() {
        let mut r = rng::stream(&format!("queries/known_item/k{k}"), 0);
        for _ in 0..per_k {
            let doc_idx = rng::below(&mut r, corpus_size as u64) as usize;
            let mut words = docs::document_at(vocab, doc_idx);
            // Take k distinct words from the document by partial Fisher-Yates, so the
            // chosen terms are a uniform k-subset rather than a biased prefix.
            for i in 0..k {
                let j = i + rng::below(&mut r, (words.len() - i) as u64) as usize;
                words.swap(i, j);
            }
            write_one(
                out,
                Query {
                    qid,
                    kind: "known_item",
                    k,
                    text: words[..k].join(" "),
                    source_doc: Some(doc_idx),
                },
            )?;
            qid += 1;
        }
    }

    for &k in TERM_COUNTS.iter() {
        let mut r = rng::stream(&format!("queries/random/k{k}"), 0);
        for _ in 0..per_k {
            let mut picked: Vec<&str> = Vec::with_capacity(k);
            while picked.len() < k {
                let w = vocab[rng::below(&mut r, vocab.len() as u64) as usize].as_str();
                if !picked.contains(&w) {
                    picked.push(w);
                }
            }
            write_one(
                out,
                Query { qid, kind: "random", k, text: picked.join(" "), source_doc: None },
            )?;
            qid += 1;
        }
    }

    out.flush()?;
    Ok(qid)
}

fn write_one<W: Write>(out: &mut W, q: Query) -> Result<()> {
    serde_json::to_writer(&mut *out, &q)?;
    out.write_all(b"\n")?;
    Ok(())
}
