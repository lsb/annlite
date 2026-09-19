//! Document generation: shuffled dictionary words cut into fixed-size chunks.

use crate::rng;
use crate::WORDS_PER_DOC;
use anyhow::Result;
use sha2::{Digest, Sha256};
use std::io::Write;

/// Generate `n_docs` documents of [`WORDS_PER_DOC`] words each, writing one document
/// per line, and return the SHA-256 of the output.
///
/// Generation proceeds in *rounds*. Each round independently shuffles the whole
/// vocabulary and cuts it into consecutive chunks of [`WORDS_PER_DOC`]; the tail
/// shorter than a full chunk is discarded so every document has exactly the same
/// length. Rounds continue until `n_docs` documents exist.
///
/// Two consequences are worth stating because the benchmarks rely on them:
///
/// * **Prefix property.** Document `i` depends only on `i` and the base seed, never
///   on `n_docs`. The 100-document corpus is therefore a byte-exact prefix of the
///   10k corpus, which is a prefix of the 1M corpus. Index-size and latency curves
///   across the three scales describe the same growing collection rather than three
///   unrelated samples.
/// * **No repeated word within a document.** Chunks are cut from a permutation, so
///   every term in a document has frequency exactly 1. This removes term-frequency
///   as a confounder from the FTS5 baseline: BM25 differences between documents come
///   only from document length (constant here) and inverse document frequency.
pub fn generate<W: Write>(vocab: &[String], n_docs: usize, out: &mut W) -> Result<String> {
    let chunks_per_round = vocab.len() / WORDS_PER_DOC;
    anyhow::ensure!(
        chunks_per_round > 0,
        "vocabulary of {} words is too small for {}-word documents",
        vocab.len(),
        WORDS_PER_DOC
    );

    let mut hasher = Sha256::new();
    let mut buf: Vec<u8> = Vec::with_capacity(WORDS_PER_DOC * 16);
    let mut scratch: Vec<&str> = Vec::with_capacity(vocab.len());
    let mut written = 0usize;
    let mut round = 0u64;

    while written < n_docs {
        scratch.clear();
        scratch.extend(vocab.iter().map(|s| s.as_str()));
        let mut r = rng::stream(&format!("docs/round/{round}"), 0);
        rng::shuffle(&mut scratch, &mut r);

        for chunk in scratch.chunks_exact(WORDS_PER_DOC) {
            if written == n_docs {
                break;
            }
            buf.clear();
            for (i, w) in chunk.iter().enumerate() {
                if i > 0 {
                    buf.push(b' ');
                }
                buf.extend_from_slice(w.as_bytes());
            }
            buf.push(b'\n');
            hasher.update(&buf);
            out.write_all(&buf)?;
            written += 1;
        }
        round += 1;
    }
    out.flush()?;
    Ok(hex(&hasher.finalize()))
}

/// Reproduce a single document without generating the ones before it.
///
/// Used by the query builder, which needs the words of a chosen source document in
/// order to ask for it by name. Regenerating one round is far cheaper than
/// regenerating the corpus prefix.
pub fn document_at(vocab: &[String], index: usize) -> Vec<String> {
    let chunks_per_round = vocab.len() / WORDS_PER_DOC;
    let round = (index / chunks_per_round) as u64;
    let within = index % chunks_per_round;

    let mut scratch: Vec<&str> = vocab.iter().map(|s| s.as_str()).collect();
    let mut r = rng::stream(&format!("docs/round/{round}"), 0);
    rng::shuffle(&mut scratch, &mut r);

    scratch[within * WORDS_PER_DOC..(within + 1) * WORDS_PER_DOC]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
