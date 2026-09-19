//! The database must give back exactly the index that was written to it.
//!
//! Everything downstream -- the browser client, the page-access measurements, the
//! results matrix -- assumes the stored form is faithful. These tests check that
//! directly rather than inferring it from search quality, because a permutation
//! applied inconsistently would still return plausible-looking neighbours.

use annlite_core::layout::{bfs_order, Ordering, Permutation};
use annlite_core::pq::ProductQuantizer;
use annlite_core::vamana::{Vamana, VamanaParams};
use annlite_core::vectors::Vectors;
use annlite_sqlite::format::RecordFormat;
use annlite_sqlite::search::{search, search_resident, Index, ResidentCodes};
use annlite_sqlite::store::{node_page_stats, write_index};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rusqlite::Connection;

const DIM: usize = 64;
const M: usize = 16;
const R: usize = 16;

fn corpus(n: usize, seed: u64) -> Vectors {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let centres: Vec<Vec<f32>> = (0..20)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect();
    let mut data = Vec::with_capacity(n * DIM);
    for i in 0..n {
        let c = &centres[i % centres.len()];
        let mut v: Vec<f32> = c.iter().map(|x| x + rng.gen_range(-0.2f32..0.2)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        data.extend_from_slice(&v);
    }
    Vectors { data, dim: DIM }
}

struct Built {
    conn: Connection,
    idx: Index,
    perm: Permutation,
    graph: Vamana,
    docs: Vectors,
}

fn build(n: usize, ordering: Ordering) -> Built {
    let docs = corpus(n, 7);
    let pq = ProductQuantizer::train(&docs, M, 10, 1).unwrap();
    let codes = pq.encode_all(&docs);
    let graph = Vamana::build(&docs, VamanaParams { r: R, l_build: 48, alpha: 1.1, seed: 2 });
    let perm = match ordering {
        Ordering::Bfs => bfs_order(n, graph.medoid, |x| graph.neighbors(x).to_vec()),
        _ => Permutation::identity(n),
    };
    let mut conn = Connection::open_in_memory().unwrap();
    write_index(&mut conn, &graph, &pq, &codes, &docs, &perm, ordering, None).unwrap();
    let idx = Index::open(&conn).unwrap();
    Built { conn, idx, perm, graph, docs }
}

#[test]
fn stored_adjacency_matches_the_graph_under_the_permutation() {
    let b = build(600, Ordering::Bfs);
    let fmt = RecordFormat { m: M, r: R };
    let mut stmt = b.conn.prepare("SELECT rec FROM annlite_nodes WHERE id = ?1").unwrap();
    for old in 0..b.docs.len() as u32 {
        let new = b.perm.new_id_of[old as usize];
        let rec: Vec<u8> = stmt.query_row([new as i64], |r| r.get(0)).unwrap();
        let (_, stored) = fmt.decode(&rec).unwrap();
        let mut expected: Vec<u32> = b
            .graph
            .neighbors(old)
            .iter()
            .map(|&o| b.perm.new_id_of[o as usize])
            .collect();
        let mut got = stored.clone();
        expected.sort_unstable();
        got.sort_unstable();
        assert_eq!(got, expected, "node {old} (stored as {new}) has the wrong neighbours");
    }
}

#[test]
fn stored_vectors_match_under_the_permutation() {
    let b = build(400, Ordering::Bfs);
    let mut stmt = b.conn.prepare("SELECT v FROM annlite_vectors WHERE id = ?1").unwrap();
    for old in [0usize, 1, 17, 199, 399] {
        let new = b.perm.new_id_of[old];
        let raw: Vec<u8> = stmt.query_row([new as i64], |r| r.get(0)).unwrap();
        let v: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(v, b.docs.row(old), "vector for document {old} was stored wrong");
    }
}

#[test]
fn code_blob_agrees_with_the_records() {
    // The blob and the records hold the same codes in the same numbering; if they
    // drifted apart, resident and on-disk search would silently disagree.
    let b = build(500, Ordering::Bfs);
    let resident = ResidentCodes::load(&b.conn, &b.idx).unwrap();
    let fmt = RecordFormat { m: M, r: R };
    let mut stmt = b.conn.prepare("SELECT rec FROM annlite_nodes WHERE id = ?1").unwrap();
    for new in 0..b.docs.len() {
        let rec: Vec<u8> = stmt.query_row([new as i64], |r| r.get(0)).unwrap();
        let (code, _) = fmt.decode(&rec).unwrap();
        assert_eq!(code, &resident.codes[new * M..(new + 1) * M], "codes differ at node {new}");
    }
}

#[test]
fn resident_and_on_disk_search_agree_exactly() {
    // They score the same candidates from the same bytes, so any difference is a
    // bug in one of the two traversals, not an approximation.
    let b = build(800, Ordering::Bfs);
    let resident = ResidentCodes::load(&b.conn, &b.idx).unwrap();
    for qi in [0usize, 5, 50, 300] {
        let q = b.docs.row(qi);
        let a = search(&b.conn, &b.idx, q, 10, 64, 4, 0).unwrap();
        let c = search_resident(&b.conn, &b.idx, &resident, q, 10, 64, 4, 0).unwrap();
        let ids = |r: &annlite_sqlite::search::SearchResult| -> Vec<u32> {
            r.results.iter().map(|x| x.0).collect()
        };
        assert_eq!(ids(&a), ids(&c), "query {qi} ranked differently");
        assert!(
            c.cost.nodes_read < a.cost.nodes_read,
            "resident read {} records against on-disk's {}",
            c.cost.nodes_read, a.cost.nodes_read
        );
    }
}

#[test]
fn search_finds_a_document_by_its_own_vector() {
    // The weakest possible sanity check, and the one that catches a permutation
    // applied in the wrong direction: a document must retrieve itself.
    let b = build(600, Ordering::Bfs);
    let mut found = 0;
    for old in [0usize, 11, 123, 456, 599] {
        let hits = search(&b.conn, &b.idx, b.docs.row(old), 10, 128, 8, 10).unwrap();
        let want = b.perm.new_id_of[old];
        if hits.results.iter().any(|x| x.0 == want) {
            found += 1;
        }
    }
    assert!(found >= 4, "only {found}/5 documents retrieved themselves");
}

#[test]
fn ordering_changes_pages_but_not_results() {
    let ident = build(800, Ordering::Identity);
    let bfs = build(800, Ordering::Bfs);
    for qi in [0usize, 37, 500] {
        let q = ident.docs.row(qi);
        let a = search(&ident.conn, &ident.idx, q, 10, 64, 4, 0).unwrap();
        let c = search(&bfs.conn, &bfs.idx, q, 10, 64, 4, 0).unwrap();
        // Results are the same documents, named in each database's own numbering.
        let a_old: Vec<u32> =
            a.results.iter().map(|x| ident.perm.old_id_of[x.0 as usize]).collect();
        let c_old: Vec<u32> = c.results.iter().map(|x| bfs.perm.old_id_of[x.0 as usize]).collect();
        assert_eq!(a_old, c_old, "query {qi} returned different documents under reordering");
        assert_eq!(a.cost.nodes_read, c.cost.nodes_read, "traversal differed");
    }
}

#[test]
fn records_pack_densely_into_pages() {
    // The id-to-page arithmetic only holds if SQLite really is packing fixed-size
    // rows in rowid order.
    let b = build(2000, Ordering::Identity);
    let (pages, per_page) = node_page_stats(&b.conn).unwrap();
    let fmt = RecordFormat { m: M, r: R };
    let predicted = fmt.records_per_page(4096) as f64;
    assert!(pages > 0);
    assert!(
        (per_page - predicted).abs() / predicted < 0.25,
        "measured {per_page:.1} records per page against a predicted {predicted:.1}"
    );
}
