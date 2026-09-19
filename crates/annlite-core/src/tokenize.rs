//! BERT WordPiece tokenization.
//!
//! A port of `tools/embed/tokenization.py`, which prepares the corpus offline. Both
//! must agree exactly: if the browser tokenizes a query differently from how the
//! corpus was tokenized, every similarity is computed between vectors from two
//! different input distributions and the retrieval quality figures mean nothing.
//! `tests/tokenize.rs` asserts the same reference ids the Python suite does, so the
//! two are pinned to a shared set of vectors rather than to each other's behaviour.

use std::collections::HashMap;

/// Words longer than this become `[UNK]` without being decomposed, as in the
/// reference implementation.
pub const MAX_CHARS_PER_WORD: usize = 100;

pub struct WordPiece {
    vocab: Vec<String>,
    ids: HashMap<String, u32>,
    lowercase: bool,
    pub pad_id: u32,
    pub unk_id: u32,
    pub cls_id: u32,
    pub sep_id: u32,
}

impl WordPiece {
    /// Parse a newline-separated vocabulary file.
    pub fn from_vocab_text(text: &str, lowercase: bool) -> anyhow::Result<Self> {
        let vocab: Vec<String> = text.lines().map(str::to_string).collect();
        let ids: HashMap<String, u32> =
            vocab.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let need = |n: &str| -> anyhow::Result<u32> {
            ids.get(n).copied().ok_or_else(|| anyhow::anyhow!("vocabulary is missing {n}"))
        };
        Ok(Self {
            pad_id: need("[PAD]")?,
            unk_id: need("[UNK]")?,
            cls_id: need("[CLS]")?,
            sep_id: need("[SEP]")?,
            vocab,
            ids,
            lowercase,
        })
    }

    pub fn len(&self) -> usize {
        self.vocab.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }

    pub fn token(&self, id: u32) -> &str {
        &self.vocab[id as usize]
    }

    /// Token ids without special tokens.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for word in self.basic_tokenize(text) {
            self.wordpiece_into(&word, &mut out);
        }
        out
    }

    /// Token ids wrapped in `[CLS]` and `[SEP]`, truncated to `max_length` total.
    pub fn encode(&self, text: &str, max_length: usize) -> Vec<u32> {
        let mut ids = self.tokenize(text);
        ids.truncate(max_length.saturating_sub(2));
        let mut out = Vec::with_capacity(ids.len() + 2);
        out.push(self.cls_id);
        out.extend(ids);
        out.push(self.sep_id);
        out
    }

    fn basic_tokenize(&self, text: &str) -> Vec<String> {
        let mut cleaned = String::with_capacity(text.len());
        for ch in text.chars() {
            if ch == '\0' || ch == '\u{FFFD}' || is_control(ch) {
                continue;
            }
            if is_whitespace(ch) {
                cleaned.push(' ');
            } else if is_cjk(ch) {
                // Isolate CJK so each character becomes its own token.
                cleaned.push(' ');
                cleaned.push(ch);
                cleaned.push(' ');
            } else {
                cleaned.push(ch);
            }
        }

        let mut out = Vec::new();
        for word in cleaned.split_whitespace() {
            let word = if self.lowercase {
                strip_accents(&word.to_lowercase())
            } else {
                word.to_string()
            };
            split_punctuation(&word, &mut out);
        }
        out
    }

    fn wordpiece_into(&self, token: &str, out: &mut Vec<u32>) {
        let chars: Vec<char> = token.chars().collect();
        if chars.len() > MAX_CHARS_PER_WORD {
            out.push(self.unk_id);
            return;
        }
        let mut pieces = Vec::new();
        let mut start = 0usize;
        while start < chars.len() {
            let mut end = chars.len();
            let mut found = None;
            while start < end {
                let mut sub: String = chars[start..end].iter().collect();
                if start > 0 {
                    sub.insert_str(0, "##");
                }
                if let Some(&id) = self.ids.get(&sub) {
                    found = Some(id);
                    break;
                }
                end -= 1;
            }
            match found {
                Some(id) => {
                    pieces.push(id);
                    start = end;
                }
                // All-or-nothing: a word with any unmatchable piece becomes [UNK]
                // entirely, rather than yielding a partial decomposition.
                None => {
                    out.push(self.unk_id);
                    return;
                }
            }
        }
        out.extend(pieces);
    }
}

fn is_control(ch: char) -> bool {
    if ch == '\t' || ch == '\n' || ch == '\r' {
        return false;
    }
    ch.is_control()
}

fn is_whitespace(ch: char) -> bool {
    ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch.is_whitespace()
}

/// BERT treats every non-alphanumeric ASCII character as punctuation, which is
/// broader than Unicode's `P*` categories: `$`, `+`, `^` and friends split too.
fn is_punctuation(ch: char) -> bool {
    let cp = ch as u32;
    if (33..=47).contains(&cp)
        || (58..=64).contains(&cp)
        || (91..=96).contains(&cp)
        || (123..=126).contains(&cp)
    {
        return true;
    }
    matches!(
        unicode_category(ch),
        Category::Punctuation
    )
}

fn is_cjk(ch: char) -> bool {
    let cp = ch as u32;
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B820..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

fn split_punctuation(word: &str, out: &mut Vec<String>) {
    let mut buf = String::new();
    for ch in word.chars() {
        if is_punctuation(ch) {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            out.push(ch.to_string());
        } else {
            buf.push(ch);
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
}

enum Category {
    Punctuation,
    Mark,
    Other,
}

/// Just enough of the Unicode character database for BERT's uncased pipeline:
/// whether a character is a combining mark (dropped when stripping accents) or
/// punctuation. Pulling in a full UCD crate would add megabytes to the WASM build
/// for two predicates, and the ranges that matter for Latin-script text are small.
fn unicode_category(ch: char) -> Category {
    let cp = ch as u32;
    // Combining diacritical marks and the common extensions.
    if (0x0300..=0x036F).contains(&cp)
        || (0x1AB0..=0x1AFF).contains(&cp)
        || (0x1DC0..=0x1DFF).contains(&cp)
        || (0x20D0..=0x20FF).contains(&cp)
        || (0xFE20..=0xFE2F).contains(&cp)
    {
        return Category::Mark;
    }
    if (0x2010..=0x2027).contains(&cp)
        || (0x2030..=0x205E).contains(&cp)
        || (0x00A1..=0x00BF).contains(&cp) && matches!(cp, 0x00A1 | 0x00AB | 0x00B6 | 0x00B7 | 0x00BB | 0x00BF)
        || (0x3001..=0x3003).contains(&cp)
        || (0xFF01..=0xFF0F).contains(&cp)
    {
        return Category::Punctuation;
    }
    Category::Other
}

/// NFD then drop combining marks, matching the uncased models' accent stripping.
///
/// Decomposition is table-driven for Latin-1 and Latin Extended-A, which covers the
/// accented characters that appear in practice for these corpora. Anything outside
/// those ranges passes through unchanged; the alternative is a full normalisation
/// crate, which is not worth its size in a WASM bundle for this.
fn strip_accents(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(unicode_category(*c), Category::Mark))
        .map(decompose_latin)
        .collect()
}

fn decompose_latin(c: char) -> char {
    match c {
        'à'..='å' | 'ā' | 'ă' | 'ą' => 'a',
        'è'..='ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => 'e',
        'ì'..='ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' => 'i',
        'ò'..='ö' | 'ō' | 'ŏ' | 'ő' => 'o',
        'ù'..='ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => 'u',
        'ý' | 'ÿ' => 'y',
        'ñ' | 'ń' | 'ņ' | 'ň' => 'n',
        'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => 'c',
        'ß' => 'ß',
        other => other,
    }
}
