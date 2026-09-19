//! The stored late index must be byte-for-byte the index that was written.
//!
//! The variable-length form is more fragile than the dense one. A fixed-size record
//! that is off by an element fails loudly; an offsets directory that is off by one
//! hands back a document's codes shifted by a token, which still scores, still
//! ranks, and still looks entirely plausible. So the arenas are checked against the
//! in-memory index directly rather than through retrieval quality, and the page
//! mapping is checked against `dbstat` rather than against the arithmetic that
//! produced it.

use annlite_core::late::{LateIndex, MultiVector};
use annlite_sqlite::late_search::{late_search, LateDb};
use annlite_sqlite::late_store::{write_late_index, Arena, OVERFLOW_USABLE};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rusqlite::{Connection, DatabaseName};

const DIM: usize = 16;

/// Documents of varying length drawn from a shared concept pool -- varying length is
/// the property under test, so a fixed token count would defeat the point.
fn corpus(n: usize, seed: u64) -> Vec<MultiVector> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let pool: Vec<Vec<f32>> = (0..40).map(|_| unit(&mut rng)).collect();
    (0..n)
        .map(|_| {
            let tokens = rng.gen_range(3..120);
            let mut data = Vec::with_capacity(tokens * DIM);
            for _ in 0..tokens {
                let c = &pool[rng.gen_range(0..pool.len())];
                let mut v: Vec<f32> = c.iter().map(|x| x + rng.gen_range(-0.05f32..0.05)).collect();
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= n);
                data.extend_from_slice(&v);
            }
            MultiVector { data, dim: DIM }
        })
        .collect()
}

fn unit(rng: &mut ChaCha8Rng) -> Vec<f32> {
    let mut v: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
    v
}

struct Built {
    conn: Connection,
    db: LateDb,
    idx: LateIndex,
    docs: Vec<MultiVector>,
}

fn build(n: usize, k: usize) -> Built {
    let docs = corpus(n, 11);
    let idx = LateIndex::build(&docs, k, 8, 0xBEEF).unwrap();
    // A file, not `:memory:`: `dbstat` reports page numbers either way, but the
    // overflow-chain contiguity this format relies on is a property of how a real
    // file is allocated.
    let dir = std::env::temp_dir().join(format!("annlite-late-{n}-{k}-{}", std::process::id()));
    let _ = std::fs::remove_file(&dir);
    let mut conn = Connection::open(&dir).unwrap();
    write_late_index(&mut conn, &idx, &docs).unwrap();
    let db = LateDb::open(&conn).unwrap();
    let _ = std::fs::remove_file(&dir);
    Built { conn, db, idx, docs }
}

fn read(conn: &Connection, table: &str, off: usize, len: usize) -> Vec<u8> {
    let blob = conn.blob_open(DatabaseName::Main, table, "data", 0, true).unwrap();
    let mut buf = vec![0u8; len];
    if len > 0 {
        blob.read_at_exact(&mut buf, off).unwrap();
    }
    buf
}

fn as_u32(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
fn every_document_reads_back_the_codes_it_was_given() {
    let b = build(400, 64);
    for d in 0..b.idx.len() as u32 {
        let (a, z) = (b.db.doc_offsets[d as usize], b.db.doc_offsets[d as usize + 1]);
        let got = as_u32(&read(&b.conn, "late_codes", a as usize * 4, (z - a) as usize * 4));
        assert_eq!(got, b.idx.doc_codes(d), "document {d} came back with the wrong codes");
    }
}

#[test]
fn every_posting_list_reads_back_unchanged() {
    let b = build(400, 64);
    for c in 0..b.db.k as u32 {
        let (a, z) = (b.db.posting_offsets[c as usize], b.db.posting_offsets[c as usize + 1]);
        let got = as_u32(&read(&b.conn, "late_postings", a as usize * 4, (z - a) as usize * 4));
        assert_eq!(got, b.idx.postings(c), "centroid {c} has the wrong posting list");
    }
}

#[test]
fn every_token_vector_reads_back_unchanged() {
    // Stage 3 is the only stage that can be wrong without stages 1 and 2 noticing,
    // because nothing else ever reads these bytes.
    let b = build(300, 64);
    let stride = DIM * 4;
    for d in 0..b.idx.len() {
        let (a, z) = (b.db.doc_offsets[d], b.db.doc_offsets[d + 1]);
        let raw = read(&b.conn, "late_tokens", a as usize * stride, (z - a) as usize * stride);
        let got: Vec<f32> =
            raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(got, b.docs[d].data, "document {d}'s token vectors were stored wrong");
    }
}

#[test]
fn the_offsets_directory_describes_the_whole_arena() {
    // Off-by-one in either direction is the failure this format is most exposed to:
    // the spans must tile the arena exactly, with no gap and no overlap.
    let b = build(400, 64);
    assert_eq!(b.db.doc_offsets[0], 0);
    for d in 0..b.idx.len() {
        assert_eq!(
            b.db.doc_offsets[d + 1] - b.db.doc_offsets[d],
            b.docs[d].tokens() as u32,
            "document {d}'s span is the wrong length"
        );
    }
    assert_eq!(*b.db.doc_offsets.last().unwrap() as usize, b.db.tokens);
    assert_eq!(b.db.codes_arena.len, b.db.tokens * 4);
    assert_eq!(b.db.tokens_arena.len, b.db.tokens * DIM * 4);
}

#[test]
fn the_page_mapping_agrees_with_what_sqlite_actually_did() {
    // The page attribution is arithmetic over byte offsets, anchored on page numbers
    // read out of dbstat. If the anchor or the local/overflow split were wrong, the
    // mapping would still return page numbers -- plausible ones -- so it is checked
    // against the pages dbstat reports for the same table.
    let b = build(600, 64);
    for arena in [&b.db.postings_arena, &b.db.codes_arena, &b.db.tokens_arena] {
        let mut from_dbstat: Vec<usize> = b
            .conn
            .prepare("SELECT pageno FROM dbstat WHERE name = ?1 AND pagetype IN ('leaf','overflow')")
            .unwrap()
            .query_map([arena.table.as_str()], |r| r.get::<_, i64>(0))
            .unwrap()
            .map(|x| x.unwrap() as usize)
            .collect();
        from_dbstat.sort_unstable();

        let mut mapped = arena.pages_for(0, arena.len);
        mapped.sort_unstable();
        mapped.dedup();
        assert_eq!(mapped, from_dbstat, "{}: page mapping disagrees with dbstat", arena.table);

        // Byte zero is on the leaf page and the last byte is on the last overflow
        // page; anything else means the local/overflow split is misplaced.
        assert_eq!(arena.pages_for(0, 1), vec![arena.leaf_page]);
        assert_eq!(
            arena.pages_for(arena.len - 1, 1),
            vec![*arena.overflow.last().unwrap()],
            "{}: last byte is not on the last overflow page",
            arena.table
        );
        assert!(
            arena.is_contiguous(),
            "{}: overflow chain is not consecutive, so runs are inflated",
            arena.table
        );
        // A range crossing a page boundary must report both pages.
        let cross = arena.local_bytes + OVERFLOW_USABLE - 2;
        assert_eq!(arena.pages_for(cross, 4).len(), 2, "{}: boundary range", arena.table);
    }
}

#[test]
fn sqlite_search_returns_what_the_in_memory_index_returns() {
    // The two read the same bytes through completely different paths -- one from a
    // Vec, one from byte ranges into a BLOB -- and share only the scoring helpers.
    // Any disagreement is a storage bug, not an approximation.
    let b = build(500, 64);
    for qi in [0usize, 7, 88, 301] {
        let q = &b.docs[qi];
        let want = b.idx.candidates(q, 4, 50);
        let got = late_search(&b.conn, &b.db, q, 4, 50, 0).unwrap();
        let ids = |v: &[(u32, f32)]| -> Vec<u32> { v.iter().map(|x| x.0).collect() };
        assert_eq!(ids(&got.results), ids(&want), "query {qi}: candidate ranking differs");
        for (a, c) in got.results.iter().zip(want.iter()) {
            assert!((a.1 - c.1).abs() < 1e-4, "query {qi}: document {} scored differently", a.0);
        }
    }
}

#[test]
fn reranking_out_of_the_database_matches_exact_maxsim_in_memory() {
    let b = build(500, 64);
    for qi in [0usize, 13, 222] {
        let q = &b.docs[qi];
        let got = late_search(&b.conn, &b.db, q, 4, 100, 20).unwrap();
        for &(d, s) in got.results.iter().take(20) {
            let want = annlite_core::late::maxsim(q, &b.docs[d as usize]);
            assert!(
                (s - want).abs() < 1e-3,
                "query {qi}: document {d} scored {s} from disk against {want} in memory"
            );
        }
        // A document retrieves itself: the weakest check there is, and the one that
        // catches a directory applied to the wrong arena.
        assert_eq!(got.results[0].0 as usize, qi, "query {qi} did not retrieve itself");
    }
}

#[test]
fn stage_costs_are_attributed_to_their_own_arena() {
    // The per-stage numbers are the entire output of this benchmark. If two stages
    // could share a page the split would be meaningless, so the arenas must occupy
    // disjoint page ranges and each stage must only ever touch its own.
    let b = build(600, 64);
    let q = &b.docs[3];
    let res = late_search(&b.conn, &b.db, q, 4, 100, 25).unwrap();
    let c = &res.cost;
    assert_eq!(c.hops, 3, "three stages, three dependent rounds");
    assert!(c.postings.distinct_pages > 0 && c.centroid.distinct_pages > 0);
    assert!(c.rerank.reads == 25, "reranked {} documents, expected 25", c.rerank.reads);
    assert_eq!(
        c.distinct_pages,
        c.postings.distinct_pages + c.centroid.distinct_pages + c.rerank.distinct_pages
    );

    let page_set = |a: &Arena| -> std::collections::HashSet<usize> {
        a.pages_for(0, a.len).into_iter().collect()
    };
    let (p, cd, t) =
        (page_set(&b.db.postings_arena), page_set(&b.db.codes_arena), page_set(&b.db.tokens_arena));
    assert!(p.is_disjoint(&cd) && cd.is_disjoint(&t) && p.is_disjoint(&t), "arenas share pages");

    // Stage 3 reads dim*4 bytes per token where stage 2 reads 4, so with the same
    // documents its byte count is dim times larger. Anything else means one stage is
    // reading the other's arena.
    let per_doc: usize = res.results[..25].iter().map(|&(d, _)| b.db.doc_tokens(d)).sum();
    assert_eq!(c.rerank.bytes, per_doc * DIM * 4);
}
