//! Browser bindings.
//!
//! The division of labour is deliberate. Reading bytes out of the remote database is
//! JavaScript's job, because `sql.js-httpvfs` already does it and does it well; the
//! arithmetic and the traversal policy are Rust's, because they are hot and because
//! they must behave identically to the offline build that produced the index.
//!
//! Node fetching enters through a **synchronous** callback. That works, and is much
//! simpler than threading async through the traversal, because `sql.js-httpvfs` runs
//! inside a Web Worker where it issues synchronous `XMLHttpRequest` calls. The
//! traversal therefore reads like ordinary blocking code while the page stays
//! responsive, since the worker is not the UI thread.

use annlite_core::pq::{ProductQuantizer, ScoreTable, CENTROIDS};
use annlite_core::tokenize::WordPiece;
use wasm_bindgen::prelude::*;

/// BERT WordPiece, so a query is tokenized in the browser exactly as the corpus was
/// tokenized offline. `annlite-core`'s test suite pins both to shared vectors.
#[wasm_bindgen]
pub struct Tokenizer {
    inner: WordPiece,
}

#[wasm_bindgen]
impl Tokenizer {
    /// Build from the contents of a `vocab.txt`.
    #[wasm_bindgen(constructor)]
    pub fn new(vocab_text: &str) -> Result<Tokenizer, JsValue> {
        WordPiece::from_vocab_text(vocab_text, true)
            .map(|inner| Tokenizer { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Token ids with `[CLS]`/`[SEP]`, ready to feed to onnxruntime-web.
    pub fn encode(&self, text: &str, max_length: usize) -> Vec<u32> {
        self.inner.encode(text, max_length)
    }

    #[wasm_bindgen(getter)]
    pub fn vocab_size(&self) -> usize {
        self.inner.len()
    }
}

/// Mean-pool a model's `last_hidden_state` over its attention mask and L2-normalise.
///
/// Exposed because getting this wrong is the most common way to break a dense
/// retrieval demo, and the failure is silent: an unmasked mean still returns
/// plausible-looking vectors whose similarities are quietly wrong, and wrong by an
/// amount that depends on how much padding the batch carried.
#[wasm_bindgen]
pub fn pool_and_normalize(hidden: &[f32], mask: &[u32], seq_len: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; dim];
    let mut count = 0f32;
    for t in 0..seq_len {
        if mask.get(t).copied().unwrap_or(0) == 0 {
            continue;
        }
        count += 1.0;
        for d in 0..dim {
            out[d] += hidden[t * dim + d];
        }
    }
    if count > 0.0 {
        out.iter_mut().for_each(|x| *x /= count);
    }
    let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    out.iter_mut().for_each(|x| *x /= norm);
    out
}

/// A Vamana index served out of a remote SQLite file.
#[wasm_bindgen]
pub struct Index {
    pq: ProductQuantizer,
    r: usize,
    count: usize,
    medoid: u32,
    /// All PQ codes, when the caller chose to preload them. See the resident-codes
    /// measurement in RESEARCH_LOG.md section 11.5: it trades one bulk download for
    /// roughly an order of magnitude fewer page reads per query.
    codes: Option<Vec<u8>>,
}

#[wasm_bindgen]
pub struct SearchStats {
    pub nodes_read: usize,
    pub hops: usize,
}

#[wasm_bindgen]
impl Index {
    /// `codebook` is the raw `pq_centroids` blob from `annlite_meta`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        codebook: &[u8],
        dim: usize,
        m: usize,
        dsub: usize,
        r: usize,
        count: usize,
        medoid: u32,
    ) -> Result<Index, JsValue> {
        let expect = m * CENTROIDS * dsub * 4;
        if codebook.len() != expect {
            return Err(JsValue::from_str(&format!(
                "codebook is {} bytes, expected {expect}",
                codebook.len()
            )));
        }
        let centroids = codebook
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(Index {
            pq: ProductQuantizer { dim, m, dsub, centroids },
            r,
            count,
            medoid,
            codes: None,
        })
    }

    /// Supply the whole code blob, switching to resident-codes traversal.
    pub fn load_codes(&mut self, codes: &[u8]) -> Result<(), JsValue> {
        let expect = self.count * self.pq.m;
        if codes.len() != expect {
            return Err(JsValue::from_str(&format!(
                "code blob is {} bytes, expected {expect}",
                codes.len()
            )));
        }
        self.codes = Some(codes.to_vec());
        Ok(())
    }

    #[wasm_bindgen(getter)]
    pub fn record_bytes(&self) -> usize {
        self.pq.m + 2 + self.r * 4
    }

    #[wasm_bindgen(getter)]
    pub fn resident(&self) -> bool {
        self.codes.is_some()
    }

    /// Beam search. `fetch` is `(id: number) => Uint8Array` returning that node's
    /// record; it is called only for nodes the search expands when codes are
    /// resident, and for every node scored otherwise.
    ///
    /// Returns candidate ids best-first. The caller reranks with full vectors if it
    /// wants the last 30% of accuracy PQ alone does not give (section 7.3).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        l: usize,
        beam: usize,
        fetch: &js_sys::Function,
    ) -> Result<Vec<u32>, JsValue> {
        if query.len() != self.pq.dim {
            return Err(JsValue::from_str(&format!(
                "query has {} dimensions, index has {}",
                query.len(),
                self.pq.dim
            )));
        }
        let table = self.pq.score_table(query);
        let l = l.max(k).max(1);
        let beam = beam.max(1);

        let read = |id: u32| -> Result<Vec<u8>, JsValue> {
            let v = fetch.call1(&JsValue::NULL, &JsValue::from_f64(id as f64))?;
            let arr = js_sys::Uint8Array::new(&v);
            Ok(arr.to_vec())
        };
        let decode = |rec: &[u8]| -> (usize, Vec<u32>) {
            let m = self.pq.m;
            let deg = u16::from_le_bytes([rec[m], rec[m + 1]]) as usize;
            let nb = (0..deg.min(self.r))
                .map(|i| {
                    let o = m + 2 + i * 4;
                    u32::from_le_bytes([rec[o], rec[o + 1], rec[o + 2], rec[o + 3]])
                })
                .collect();
            (deg, nb)
        };

        let mut seen = vec![false; self.count];
        let mut list: Vec<(u32, f32, bool)> = Vec::new();
        let mut nodes_read = 0usize;

        let score_of = |id: u32, rec: &[u8], table: &ScoreTable| -> f32 {
            match &self.codes {
                Some(c) => table.score(&c[id as usize * self.pq.m..(id as usize + 1) * self.pq.m]),
                None => table.score(&rec[..self.pq.m]),
            }
        };

        let seed_rec = read(self.medoid)?;
        nodes_read += 1;
        seen[self.medoid as usize] = true;
        list.push((self.medoid, score_of(self.medoid, &seed_rec, &table), false));

        loop {
            let frontier: Vec<u32> = list
                .iter_mut()
                .filter(|e| !e.2)
                .take(beam)
                .map(|e| {
                    e.2 = true;
                    e.0
                })
                .collect();
            if frontier.is_empty() {
                break;
            }
            let mut discovered = Vec::new();
            for id in frontier {
                let rec = read(id)?;
                nodes_read += 1;
                let (_, nb) = decode(&rec);
                for n in nb {
                    if (n as usize) < self.count && !seen[n as usize] {
                        seen[n as usize] = true;
                        discovered.push(n);
                    }
                }
            }
            for id in discovered {
                let score = match &self.codes {
                    // Resident: scoring costs nothing over the network.
                    Some(c) => {
                        table.score(&c[id as usize * self.pq.m..(id as usize + 1) * self.pq.m])
                    }
                    // On disk: the score is in the record, so it must be fetched.
                    None => {
                        let rec = read(id)?;
                        nodes_read += 1;
                        table.score(&rec[..self.pq.m])
                    }
                };
                list.push((id, score, false));
            }
            list.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            list.truncate(l);
        }

        let _ = nodes_read;
        Ok(list.into_iter().take(k).map(|e| e.0).collect())
    }
}

/// A search paused between round-trips.
///
/// The synchronous [`Index::search`] needs a callback that can block, which only
/// works inside a Web Worker. This drives the same traversal from the outside
/// instead: it hands out the ids it needs, waits to be given the bytes, and steps
/// forward. Three things fall out of that shape, and they are why it is the API the
/// demo uses.
///
/// * It works with any asynchronous source — `sql.js-httpvfs` over a worker RPC,
///   `fetch()` with a `Range` header, IndexedDB — without the traversal knowing.
/// * Each request is a *batch*. A browser issues those in parallel up to its
///   per-origin limit, so one round of the loop costs about one round-trip no
///   matter how wide the frontier. That is exactly the distinction
///   `tools/analyze/netcost.py` draws between dependent hops and parallel fetches.
/// * The round-trip count stops being an estimate. [`SearchSession::hops`] is the
///   number of times the caller had to go to the network, measured rather than
///   modelled.
#[wasm_bindgen]
pub struct SearchSession {
    table: ScoreTable,
    m: usize,
    r: usize,
    count: usize,
    codes: Option<Vec<u8>>,
    k: usize,
    l: usize,
    beam: usize,
    /// `(id, score, expanded)`, best first.
    list: Vec<(u32, f32, bool)>,
    seen: Vec<bool>,
    /// Ids handed out by the last `next_request`, awaiting `supply`.
    outstanding: Vec<u32>,
    /// Whether the outstanding batch is being expanded or merely scored. A node
    /// fetched only to read its code must not contribute its neighbours, or the
    /// traversal explores a wider graph than the reference implementation and
    /// returns different results.
    outstanding_expands: bool,
    /// Ids discovered but not yet scored, for the on-disk path.
    unscored: Vec<u32>,
    hops: usize,
    nodes_read: usize,
    started: bool,
    done: bool,
}

#[wasm_bindgen]
impl SearchSession {
    /// Ids whose records the caller must supply next. Empty means finished.
    pub fn next_request(&mut self) -> Vec<u32> {
        if self.done {
            return Vec::new();
        }
        if !self.started {
            self.outstanding = vec![self.medoid_seed()];
            self.outstanding_expands = true;
            return self.outstanding.clone();
        }
        // On-disk mode scores by reading, so unscored discoveries come first. These
        // are score-only fetches: their neighbours are not explored until and unless
        // the node is later chosen for expansion.
        if self.codes.is_none() && !self.unscored.is_empty() {
            self.outstanding = std::mem::take(&mut self.unscored);
            self.outstanding_expands = false;
            return self.outstanding.clone();
        }
        let frontier: Vec<u32> = self
            .list
            .iter_mut()
            .filter(|e| !e.2)
            .take(self.beam)
            .map(|e| {
                e.2 = true;
                e.0
            })
            .collect();
        if frontier.is_empty() {
            self.done = true;
            return Vec::new();
        }
        self.outstanding = frontier.clone();
        self.outstanding_expands = true;
        frontier
    }

    fn medoid_seed(&self) -> u32 {
        self.list.first().map(|e| e.0).unwrap_or(0)
    }

    /// Supply the records for the ids from the last [`SearchSession::next_request`],
    /// concatenated in the same order.
    pub fn supply(&mut self, records: &[u8]) -> Result<(), JsValue> {
        let rec_len = self.m + 2 + self.r * 4;
        let ids = std::mem::take(&mut self.outstanding);
        if records.len() != ids.len() * rec_len {
            return Err(JsValue::from_str(&format!(
                "supplied {} bytes for {} records of {rec_len} bytes",
                records.len(),
                ids.len()
            )));
        }
        self.hops += 1;
        self.nodes_read += ids.len();

        let expands = self.outstanding_expands;
        for (slot, &id) in ids.iter().enumerate() {
            let rec = &records[slot * rec_len..(slot + 1) * rec_len];
            if !self.started {
                self.started = true;
                let score = self.score(id, rec);
                self.list = vec![(id, score, false)];
                self.seen[id as usize] = true;
                continue;
            }
            if !self.list.iter().any(|e| e.0 == id) {
                let score = self.score(id, rec);
                self.list.push((id, score, false));
            }
            if !expands {
                continue;
            }
            let deg = u16::from_le_bytes([rec[self.m], rec[self.m + 1]]) as usize;
            for i in 0..deg.min(self.r) {
                let o = self.m + 2 + i * 4;
                let nb = u32::from_le_bytes([rec[o], rec[o + 1], rec[o + 2], rec[o + 3]]);
                if (nb as usize) >= self.count || self.seen[nb as usize] {
                    continue;
                }
                self.seen[nb as usize] = true;
                match &self.codes {
                    // Resident: score immediately, no further fetch needed.
                    Some(c) => {
                        let s = self
                            .table
                            .score(&c[nb as usize * self.m..(nb as usize + 1) * self.m]);
                        self.list.push((nb, s, false));
                    }
                    // On disk: the score is in the record, so queue it for fetching.
                    None => self.unscored.push(nb),
                }
            }
        }
        self.list.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        self.list.truncate(self.l);
        Ok(())
    }

    fn score(&self, id: u32, rec: &[u8]) -> f32 {
        match &self.codes {
            Some(c) => self.table.score(&c[id as usize * self.m..(id as usize + 1) * self.m]),
            None => self.table.score(&rec[..self.m]),
        }
    }

    /// Best `k` ids so far, best first.
    pub fn results(&self) -> Vec<u32> {
        self.list.iter().take(self.k).map(|e| e.0).collect()
    }

    #[wasm_bindgen(getter)]
    pub fn finished(&self) -> bool {
        self.done
    }

    /// Network round-trips taken. Measured, not modelled.
    #[wasm_bindgen(getter)]
    pub fn hops(&self) -> usize {
        self.hops
    }

    #[wasm_bindgen(getter)]
    pub fn nodes_read(&self) -> usize {
        self.nodes_read
    }
}

#[wasm_bindgen]
impl Index {
    /// Begin a search driven from JavaScript, one round-trip at a time.
    pub fn begin(
        &self,
        query: &[f32],
        k: usize,
        l: usize,
        beam: usize,
    ) -> Result<SearchSession, JsValue> {
        if query.len() != self.pq.dim {
            return Err(JsValue::from_str("query dimension does not match the index"));
        }
        let mut list = Vec::new();
        list.push((self.medoid, 0.0, false));
        Ok(SearchSession {
            table: self.pq.score_table(query),
            m: self.pq.m,
            r: self.r,
            count: self.count,
            codes: self.codes.clone(),
            k: k.max(1),
            l: l.max(k).max(1),
            beam: beam.max(1),
            list,
            seen: vec![false; self.count],
            outstanding: Vec::new(),
            outstanding_expands: false,
            unscored: Vec::new(),
            hops: 0,
            nodes_read: 0,
            started: false,
            done: false,
        })
    }
}
