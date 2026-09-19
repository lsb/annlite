use anyhow::{Context, Result};
use annlite_corpus::{docs, queries, vocab};
use clap::{Parser, Subcommand};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "annlite-corpus", about = "Deterministic benchmark corpora for annlite")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Normalise a system dictionary into the benchmark vocabulary.
    Vocab {
        #[arg(long, default_value = "/usr/share/dict/words")]
        dict: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Generate a document corpus, one 50-word document per line.
    Docs {
        #[arg(long)]
        vocab: PathBuf,
        #[arg(long)]
        n: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Generate a query set as JSONL.
    Queries {
        #[arg(long)]
        vocab: PathBuf,
        /// Corpus the queries are built against; bounds known-item document ids.
        #[arg(long)]
        corpus_size: usize,
        #[arg(long, default_value_t = 100)]
        per_k: usize,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Vocab { dict, out } => {
            let words = vocab::build(&dict)?;
            let mut w = BufWriter::new(File::create(&out)?);
            for word in &words {
                writeln!(w, "{word}")?;
            }
            w.flush()?;
            eprintln!("vocab: {} words -> {}", words.len(), out.display());
        }
        Cmd::Docs { vocab: vpath, n, out } => {
            let words = read_vocab(&vpath)?;
            let mut w = BufWriter::with_capacity(1 << 20, File::create(&out)?);
            let t0 = std::time::Instant::now();
            let digest = docs::generate(&words, n, &mut w)?;
            let secs = t0.elapsed().as_secs_f64();
            eprintln!(
                "docs: {n} documents in {secs:.2}s -> {}\n  sha256 {digest}",
                out.display()
            );
            write_manifest(&out, n, &digest, &words)?;
        }
        Cmd::Queries { vocab: vpath, corpus_size, per_k, out } => {
            let words = read_vocab(&vpath)?;
            let mut w = BufWriter::new(File::create(&out)?);
            let n = queries::generate(&words, corpus_size, per_k, &mut w)?;
            eprintln!("queries: {n} queries -> {}", out.display());
        }
    }
    Ok(())
}

fn read_vocab(p: &PathBuf) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(p)
        .with_context(|| format!("reading vocabulary {} (run `make vocab` first)", p.display()))?;
    Ok(text.lines().map(str::to_string).collect())
}

/// Record what was generated alongside it, so a regenerated corpus can be checked
/// against the digest quoted in RESEARCH_LOG.md without shipping the corpus itself.
fn write_manifest(out: &PathBuf, n: usize, digest: &str, words: &[String]) -> Result<()> {
    let manifest = serde_json::json!({
        "documents": n,
        "words_per_doc": annlite_corpus::WORDS_PER_DOC,
        "vocabulary_size": words.len(),
        "sha256": digest,
        "bytes": std::fs::metadata(out)?.len(),
        "generator": "annlite-corpus v1 (ChaCha8, domain-separated)",
    });
    let path = out.with_extension("manifest.json");
    std::fs::write(&path, serde_json::to_string_pretty(&manifest)? + "\n")?;
    Ok(())
}
