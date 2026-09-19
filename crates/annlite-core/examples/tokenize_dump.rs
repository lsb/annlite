//! Dump token ids for lines on stdin, for differential testing against the
//! Python implementation. One line of comma-separated ids per input line.
use annlite_core::tokenize::WordPiece;
use std::io::{BufRead, Write};

fn main() -> anyhow::Result<()> {
    let vocab = std::fs::read_to_string(
        std::env::args().nth(1).unwrap_or_else(|| {
            "models/tokenizers/bert-base-uncased-vocab.txt".to_string()
        }),
    )?;
    let t = WordPiece::from_vocab_text(&vocab, true)?;
    let stdin = std::io::stdin();
    let out = std::io::stdout();
    let mut w = std::io::BufWriter::new(out.lock());
    for line in stdin.lock().lines() {
        let ids = t.encode(&line?, 512);
        writeln!(w, "{}", ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","))?;
    }
    Ok(())
}
