//! Emit a small real index as JSON, so the browser bindings can be tested against
//! results the native implementation produced from the same bytes.

use annlite_core::pq::ProductQuantizer;
use annlite_core::vamana::{Vamana, VamanaParams};
use annlite_core::vectors::Vectors;
use annlite_sqlite::format::RecordFormat;
use annlite_sqlite::search::{search_resident, Index, ResidentCodes};
use annlite_sqlite::store::write_index;
use annlite_core::layout::{Ordering, Permutation};
use rusqlite::Connection;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let n = 500usize;
    let dim = 384usize;
    let all = Vectors::load(Path::new("data/embeddings/docs-10k.f32"), dim)?;
    let docs = Vectors { data: all.data[..n * dim].to_vec(), dim };
    let queries = Vectors::load(Path::new("data/embeddings/queries-10k.f32"), dim)?;

    let m = 16usize;
    let r = 16usize;
    let pq = ProductQuantizer::train(&docs, m, 15, 0xA11CE)?;
    let codes = pq.encode_all(&docs);
    let graph = Vamana::build(&docs, VamanaParams { r, l_build: 64, alpha: 1.1, seed: 0xDA7A });
    let fmt = RecordFormat { m, r };

    let records: Vec<Vec<u8>> = (0..n)
        .map(|i| fmt.encode(&codes[i * m..(i + 1) * m], graph.neighbors(i as u32)))
        .collect();

    // The expected answer must come from the SAME scoring function the browser uses.
    // `Vamana::search` scores with full float32 vectors, while the browser scores
    // with PQ codes, so comparing against it would be comparing two different
    // rankings and the parity check would be meaningless. Build the real index and
    // run the real PQ beam search.
    let query = queries.row(0).to_vec();
    let mut conn = Connection::open_in_memory()?;
    let perm = Permutation::identity(n);
    write_index(&mut conn, &graph, &pq, &codes, &docs, &perm, Ordering::Identity, None)?;
    let idx = Index::open(&conn)?;
    let resident = ResidentCodes::load(&conn, &idx)?;
    let expected: Vec<u32> = search_resident(&conn, &idx, &resident, &query, 5, 32, 4, 0)?
        .results
        .into_iter()
        .map(|x| x.0)
        .collect();

    let mut codebook = Vec::new();
    for x in &pq.centroids {
        codebook.extend_from_slice(&x.to_le_bytes());
    }

    let json = serde_json::json!({
        "dim": dim, "m": m, "dsub": pq.dsub, "r": r, "count": n,
        "medoid": graph.medoid,
        "codebook": codebook,
        "codes": codes,
        "records": records,
        "query": query,
        "expected": expected,
    });
    std::fs::write("web/testfixture.json", serde_json::to_string(&json)?)?;
    eprintln!("fixture: {n} docs, expected top-5 {expected:?}");
    Ok(())
}
