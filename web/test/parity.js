// Parity between the browser build and the native one.
//
// The fixture carries a real 500-document index and the top-5 that the native
// SQLite-backed PQ beam search produced from exactly those bytes. Comparing
// against `Vamana::search` instead would be meaningless: that scores with full
// float32 vectors while the browser scores with PQ codes, so the two rank
// differently by design.
//
// Run with: node web/test/parity.js   (after `make wasm`)
const path = require('path');
const ROOT = path.resolve(__dirname, '..', '..');
const a = require(path.join(ROOT, 'web', 'pkg', 'annlite_wasm.js'));
const fs = require('fs');
let failures = 0;
const check = (name, ok) => { console.log((ok ? 'ok   ' : 'FAIL ') + name); if (!ok) failures++; };

// --- tokenizer parity against the Rust/Python reference vectors ---
const vocab = fs.readFileSync(path.join(ROOT,'models','tokenizers','bert-base-uncased-vocab.txt'),'utf8');
const tok = new a.Tokenizer(vocab);
check('vocabulary loaded at the checkpoint size', tok.vocab_size === 30522);
const ids = Array.from(tok.encode('a man is playing a guitar on stage', 256));
console.log('encode ->', ids.join(','));
const expected = '101,1037,2158,2003,2652,1037,2858,2006,2754,102';
check('tokenizer reproduces the reference ids', ids.join(',') === expected);

// --- pooling ---
const seq = 3, dim = 4;
const hidden = Float32Array.from([1,0,0,0,  0,1,0,0,  99,99,99,99]);
const mask = Uint32Array.from([1,1,0]);   // third token is padding
const pooled = a.pool_and_normalize(hidden, mask, seq, dim);
check('masked pooling ignores padding', Math.abs(pooled[0]-0.7071)<1e-3 && Math.abs(pooled[2])<1e-6);

// --- index search driven by a synchronous fetch callback, as sql.js-httpvfs does ---
const db = require(path.join(ROOT,'web','testfixture.json'));
const idx = new a.Index(Uint8Array.from(db.codebook), db.dim, db.m, db.dsub, db.r, db.count, db.medoid);
console.log('record_bytes', idx.record_bytes, 'resident', idx.resident);
const recs = db.records.map(r => Uint8Array.from(r));
let fetches = 0;
const fetch = (id) => { fetches++; return recs[id]; };
const q = Float32Array.from(db.query);
let got = Array.from(idx.search(q, 5, 32, 4, fetch));
console.log('ondisk  search ->', got.join(','), 'fetches', fetches);
check('on-disk search matches the native result', got.join(',') === db.expected.join(','));

idx.load_codes(Uint8Array.from(db.codes));
fetches = 0;
const got2 = Array.from(idx.search(q, 5, 32, 4, fetch));
console.log('resident search ->', got2.join(','), 'fetches', fetches);
check('resident search matches the native result', got2.join(',') === db.expected.join(','));
check('resident mode reads strictly fewer records', fetches < 299);


// --- the resumable session must agree with the synchronous path ---
// This is the API the browser demo uses, because sql.js-httpvfs is asynchronous
// from the main thread. Each round of the loop is one network round-trip.
function drive(index) {
  const s = index.begin(q, 5, 32, 4);
  let rounds = 0;
  for (;;) {
    const need = Array.from(s.next_request());
    if (need.length === 0) break;
    // Concatenate the requested records, as a batched range fetch would.
    const buf = new Uint8Array(need.length * index.record_bytes);
    need.forEach((id, i) => buf.set(recs[id], i * index.record_bytes));
    s.supply(buf);
    if (++rounds > 10000) throw new Error('session failed to terminate');
  }
  return { ids: Array.from(s.results()), hops: s.hops, nodes: s.nodes_read };
}

const sessionResident = drive(idx);
check('resumable session matches the native result',
      sessionResident.ids.join(',') === db.expected.join(','));
console.log(`     resident session: ${sessionResident.hops} round-trips, ${sessionResident.nodes} records`);

const idx2 = new a.Index(Uint8Array.from(db.codebook), db.dim, db.m, db.dsub, db.r, db.count, db.medoid);
const sessionDisk = drive(idx2);
check('resumable session matches in on-disk mode too',
      sessionDisk.ids.join(',') === db.expected.join(','));
console.log(`     on-disk session:  ${sessionDisk.hops} round-trips, ${sessionDisk.nodes} records`);
check('resident mode needs no more round-trips than on-disk',
      sessionResident.hops <= sessionDisk.hops);
check('resident mode reads far fewer records',
      sessionResident.nodes * 3 < sessionDisk.nodes);

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
