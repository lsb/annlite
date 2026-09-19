//! `Range` header parsing, to the letter of RFC 7233 §2.1 and §4.4.
//!
//! sql.js-httpvfs is the client that matters, and it is unforgiving in one
//! specific way: it learns the database's length from the `Content-Range` of an
//! early request and then addresses pages by absolute offset. A server that
//! clamps, rounds or silently widens a range does not fail loudly — it returns a
//! 200 with the whole file, or a page of the wrong bytes, and SQLite reports a
//! corrupt database several layers away from the cause. Hence the pedantry here.
//!
//! The distinction the RFC draws, and that this module keeps:
//!
//! * A **syntactically invalid** range (`bytes=5-3`, `bytes=abc`, a unit other than
//!   bytes) MUST be ignored, giving a normal `200` with the full body.
//! * A **valid but unsatisfiable** range (first byte past the end of the file) gets
//!   `416` with `Content-Range: bytes */TOTAL`, which is how a client discovers the
//!   length it guessed wrong about.
//!
//! `If-Range` is not implemented: conditional range retrieval only matters to a
//! client resuming a transfer of an entity that may have changed underneath it,
//! and the files here are static for the life of a benchmark.
//!
//! Multi-range requests (`bytes=0-9,20-29`) are answered with the full body rather
//! than a `multipart/byteranges` document. The RFC permits a server to ignore a
//! Range header, no SQLite VFS emits multi-range, and a half-correct multipart
//! implementation would be a liability rather than a feature.

/// What a `Range` header resolves to against a file of known length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolved {
    /// Serve the whole entity with `200`.
    Full,
    /// Serve `[start, end]` inclusive with `206`.
    Partial { start: u64, end: u64 },
    /// Serve `416` with `Content-Range: bytes */total`.
    Unsatisfiable,
}

impl Resolved {
    /// Byte offset and length of what will actually be sent.
    pub fn span(&self, total: u64) -> (u64, u64) {
        match *self {
            Resolved::Full => (0, total),
            Resolved::Partial { start, end } => (start, end - start + 1),
            Resolved::Unsatisfiable => (0, 0),
        }
    }
}

/// Resolve a `Range` header value against an entity of `total` bytes.
pub fn resolve(header: Option<&str>, total: u64) -> Resolved {
    let Some(raw) = header else {
        return Resolved::Full;
    };
    let Some(spec) = raw.trim().strip_prefix("bytes=") else {
        // Other range units are not supported; RFC 7233 §3.1 says to ignore them.
        return Resolved::Full;
    };
    if spec.contains(',') {
        return Resolved::Full;
    }
    let spec = spec.trim();
    let Some((first, last)) = spec.split_once('-') else {
        return Resolved::Full;
    };
    let (first, last) = (first.trim(), last.trim());

    if first.is_empty() {
        // Suffix form `bytes=-N`: the last N bytes.
        let Some(n) = parse_u64(last) else {
            return Resolved::Full;
        };
        // `bytes=-0` is valid syntax asking for nothing, which nothing can satisfy.
        if n == 0 || total == 0 {
            return Resolved::Unsatisfiable;
        }
        let start = total.saturating_sub(n);
        return Resolved::Partial { start, end: total - 1 };
    }

    let Some(start) = parse_u64(first) else {
        return Resolved::Full;
    };
    let end = if last.is_empty() {
        // Open-ended `bytes=S-`: to the end of the entity. This is what a client
        // uses when it does not yet know the length.
        total.saturating_sub(1)
    } else {
        match parse_u64(last) {
            // A last-byte-pos beyond the end is clamped, not rejected (§2.1).
            Some(e) => e.min(total.saturating_sub(1)),
            None => return Resolved::Full,
        }
    };

    if total == 0 || start >= total {
        return Resolved::Unsatisfiable;
    }
    if start > end {
        // Only reachable as `bytes=5-3`, which is invalid syntax, so ignore it.
        return Resolved::Full;
    }
    Resolved::Partial { start, end }
}

/// Strict digits-only parse. `u64::from_str` would accept a leading `+`, and the
/// RFC's grammar is `1*DIGIT`.
fn parse_u64(s: &str) -> Option<u64> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: u64 = 1000;

    #[test]
    fn closed_open_and_suffix_forms() {
        assert_eq!(resolve(Some("bytes=0-99"), N), Resolved::Partial { start: 0, end: 99 });
        assert_eq!(resolve(Some("bytes=900-"), N), Resolved::Partial { start: 900, end: 999 });
        assert_eq!(resolve(Some("bytes=-100"), N), Resolved::Partial { start: 900, end: 999 });
        assert_eq!(resolve(Some("bytes=0-0"), N), Resolved::Partial { start: 0, end: 0 });
    }

    #[test]
    fn over_long_end_is_clamped_not_rejected() {
        assert_eq!(resolve(Some("bytes=990-99999"), N), Resolved::Partial { start: 990, end: 999 });
        assert_eq!(resolve(Some("bytes=-99999"), N), Resolved::Partial { start: 0, end: 999 });
    }

    #[test]
    fn unsatisfiable_versus_ignored() {
        assert_eq!(resolve(Some("bytes=1000-1010"), N), Resolved::Unsatisfiable);
        assert_eq!(resolve(Some("bytes=-0"), N), Resolved::Unsatisfiable);
        assert_eq!(resolve(Some("bytes=0-99"), 0), Resolved::Unsatisfiable);
        // Invalid syntax is ignored, which means a 200, not a 416.
        assert_eq!(resolve(Some("bytes=5-3"), N), Resolved::Full);
        assert_eq!(resolve(Some("bytes=abc"), N), Resolved::Full);
        assert_eq!(resolve(Some("items=0-9"), N), Resolved::Full);
        assert_eq!(resolve(Some("bytes=0-9,20-29"), N), Resolved::Full);
        assert_eq!(resolve(None, N), Resolved::Full);
    }
}
