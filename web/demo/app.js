// annlite browser demo.
//
// The whole point is visible in the counters: this page holds no copy of the
// database. Every byte it uses arrives as an HTTP Range request against a static
// file, which is what a CDN-hosted SQLite index looks like from a phone. The
// numbers reported after each search are measured, not modelled -- `hops` is the
// number of times the traversal actually had to go to the network.

import init, { Index } from './vendor/annlite/annlite_wasm.js';

// sql.js-httpvfs ships a UMD bundle, not an ES module, so index.html loads it with
// a classic script tag and it lands on the global object.
const { createDbWorker } = window;

const $ = (id) => document.getElementById(id);
const log = (msg) => { $('log').textContent += msg + '\n'; };

// HTTP cost is read back from the server rather than counted in the page.
//
// The obvious approach -- patching window.fetch -- silently reports zero, because
// sql.js-httpvfs does its reads with synchronous XMLHttpRequest inside a Web
// Worker, where neither the page's fetch nor its Performance entries can see them.
// annlite-netsim exposes a control plane for exactly this: bracket a query with a
// reset and a stats read and the numbers are the server's own, not an estimate.
// When the page is served by anything else these calls fail and the counters are
// reported as unavailable instead of as zero.
let netsim = true;
async function netsimReset(label) {
  if (!netsim) return;
  try {
    await fetch(`/__netsim/reset?label=${encodeURIComponent(label)}`, { cache: 'no-store' });
  } catch { netsim = false; }
}
async function netsimStats() {
  if (!netsim) return null;
  try {
    const r = await fetch('/__netsim/stats', { cache: 'no-store' });
    const j = r.ok ? await r.json() : null;
    // The counters live under `stats`; the reset endpoint returns the closed
    // span under `previous` instead.
    return j ? (j.stats || j.previous || null) : null;
  } catch { netsim = false; return null; }
}

let db = null, index = null, meta = {}, docCount = 0;
let transport = 'range';
const GRAPH_URL = new URL('./annlite-graph.bin', window.location.href).toString();
const CODES_URL = new URL('./annlite-codes.bin', window.location.href).toString();

async function boot() {
  await init({ module_or_path: new URL('./vendor/annlite/annlite_wasm_bg.wasm', import.meta.url) });
  log('annlite wasm loaded');

  // sql.js-httpvfs prefetches speculatively: `maxReadHeads` virtual read heads grow
  // their request size up to `maxReadSpeed`, so `requestChunkSize` is a floor, not a
  // cap. Left at the defaults the first query on this 5.9 MB database pulls about
  // 5.2 MB in six requests -- effectively the whole file -- which makes any amount
  // of page-locality work in the index irrelevant. `?readahead=off` turns it off so
  // the difference is measurable rather than asserted.
  const readahead = new URLSearchParams(location.search).get('readahead') !== 'off';
  const worker = await createDbWorker(
    [{
      from: 'inline',
      config: {
        serverMode: 'full',
        // One SQLite page per request.
        requestChunkSize: 4096,
        // Absolute: the worker resolves relative urls against its own location,
        // which is the vendor directory, not the page.
        url: new URL('./annlite-demo.db', window.location.href).toString(),
        ...(readahead ? {} : { maxReadHeads: 0, maxReadSpeed: 4096 }),
      },
    }],
    new URL('./vendor/sqlite.worker.js', import.meta.url).toString(),
    new URL('./vendor/sql-wasm.wasm', import.meta.url).toString(),
  );
  db = worker.db;
  log(`opened remote database over HTTP ranges (read-ahead ${readahead ? 'on' : 'off'})`);

  const rows = await db.query('SELECT key, value FROM annlite_meta');
  for (const row of rows) {
    const v = row.value;
    meta[row.key] = (v instanceof Uint8Array) ? new TextDecoder().decode(v) : v;
  }
  const codebook = (await db.query(
    "SELECT value AS v FROM annlite_meta WHERE key='pq_centroids'"))[0].v;

  docCount = parseInt(meta.count, 10);
  index = new Index(
    new Uint8Array(codebook),
    parseInt(meta.dim, 10), parseInt(meta.m, 10), parseInt(meta.dsub, 10),
    parseInt(meta.r, 10), docCount, parseInt(meta.medoid, 10),
  );
  log(`index: ${docCount} docs, dim ${meta.dim}, ${meta.m} B/code, R=${meta.r}, ` +
      `record ${index.record_bytes} B, ordering ${meta.ordering}`);
  $('stats').textContent =
    `${docCount} documents · ${meta.m} bytes/code · ordering ${meta.ordering}`;
  $('go').disabled = false;
  $('resident').disabled = false;
}

// Fetch the whole code blob so traversal never fetches a record just to score it.
// One sequential range request in exchange for roughly an order of magnitude fewer
// page reads per query -- see RESEARCH_LOG.md section 11.5.
async function loadResident() {
  const t0 = performance.now();
  await netsimReset('preload');
  const blob = transport === 'sqlite'
    ? new Uint8Array((await db.query('SELECT codes AS c FROM annlite_codeblob WHERE id=0'))[0].c)
    : new Uint8Array(await (await fetch(CODES_URL)).arrayBuffer());
  index.load_codes(blob);
  const st = await netsimStats();
  const wire = st ? `${st.requests} requests, ${(st.bytes / 1024).toFixed(0)} KB over the wire`
                  : 'wire cost unavailable';
  log(`resident codes: ${blob.byteLength} B in ${(performance.now() - t0).toFixed(0)} ms (${wire})`);
  $('resident').disabled = true;
  $('mode').textContent = 'resident';
}

// Two ways to get node records, kept side by side because the difference is the
// project's whole argument.
//
//   'sqlite' -- ask the database, letting sql.js-httpvfs decide what to fetch.
//   'range'  -- compute the byte offset and issue a bounded Range request against a
//               flat sidecar file. Record i begins at i * record_bytes, which is
//               exactly what the fixed-size format exists to make true.
//
// Measured on this 5.9 MB demo database, the sqlite path pulls about 5.2 MB on the
// first query because the library's read-ahead escalates to megabyte requests; the
// range path fetches only the records the traversal asks for.
async function fetchRecords(ids) {
  if (ids.length === 0) return new Uint8Array(0);
  const rec = index.record_bytes;
  if (transport === 'sqlite') {
    const rows = await db.query(
      `SELECT id, rec FROM annlite_nodes WHERE id IN (${ids.map(() => '?').join(',')})`, ids);
    const byId = new Map(rows.map((r) => [Number(r.id), new Uint8Array(r.rec)]));
    const out = new Uint8Array(ids.length * rec);
    ids.forEach((id, i) => {
      const r = byId.get(Number(id));
      if (r) out.set(r, i * rec);
    });
    return out;
  }

  // Coalesce runs of adjacent records into single requests, which is what a real
  // client does and what node ordering is meant to make profitable.
  const sorted = [...new Set(ids.map(Number))].sort((a, b) => a - b);
  const runs = [];
  for (const id of sorted) {
    const last = runs[runs.length - 1];
    if (last && id === last.end + 1) last.end = id;
    else runs.push({ start: id, end: id });
  }
  const byId = new Map();
  await Promise.all(runs.map(async (run) => {
    const from = run.start * rec;
    const to = (run.end + 1) * rec - 1;
    const resp = await fetch(GRAPH_URL, { headers: { Range: `bytes=${from}-${to}` } });
    const buf = new Uint8Array(await resp.arrayBuffer());
    for (let id = run.start; id <= run.end; id++) {
      byId.set(id, buf.subarray((id - run.start) * rec, (id - run.start + 1) * rec));
    }
  }));
  const out = new Uint8Array(ids.length * rec);
  ids.forEach((id, i) => {
    const r = byId.get(Number(id));
    if (r) out.set(r, i * rec);
  });
  return out;
}

async function search(queryVec, k, l, beam, rerank) {
  // Two measurement spans. The traversal is what the index design controls; result
  // display and exact reranking always go through SQLite here, and their read-ahead
  // would otherwise swamp the traversal's bytes and make the transports look alike.
  await netsimReset('traversal');
  const t0 = performance.now();
  const session = index.begin(queryVec, Math.max(k, rerank || k), l, beam);
  for (;;) {
    const need = Array.from(session.next_request());
    if (need.length === 0) break;
    session.supply(await fetchRecords(need));
  }
  const traversal = await netsimStats();
  await netsimReset('fetch-results');
  let ids = Array.from(session.results());

  // PQ recovers the right documents into a shallow pool but orders it only
  // approximately, so the pool is rescored against full vectors.
  if (rerank > 0) {
    const pool = ids.slice(0, rerank);
    const rows = await db.query(
      `SELECT id, v FROM annlite_vectors WHERE id IN (${pool.map(() => '?').join(',')})`, pool);
    const scored = rows.map((row) => {
      const v = new Float32Array(new Uint8Array(row.v).buffer);
      let s = 0;
      for (let i = 0; i < v.length; i++) s += v[i] * queryVec[i];
      return [Number(row.id), s];
    });
    scored.sort((a, b) => b[1] - a[1]);
    ids = scored.map((x) => x[0]);
  }
  ids = ids.slice(0, k);

  const rows = ids.length
    ? await db.query(`SELECT id, body FROM annlite_docs WHERE id IN (${ids.map(() => '?').join(',')})`, ids)
    : [];
  const bodies = new Map(rows.map((r) => [Number(r.id), r.body]));
  const ms = performance.now() - t0;
  return {
    ids, bodies, ms,
    hops: session.hops,
    nodes: session.nodes_read,
    traversal,
    display: await netsimStats(),
  };
}

async function embedQuery(text) {
  // Live embedding needs onnxruntime-web plus the 23 MB encoder. When those are not
  // present the demo falls back to precomputed vectors for a fixed query list, which
  // exercises the identical retrieval path -- only the source of the query vector
  // differs.
  const canned = window.__ANNLITE_QUERIES__ || {};
  if (canned[text]) return Float32Array.from(canned[text]);
  const keys = Object.keys(canned);
  if (keys.length === 0) throw new Error('no query vectors available');
  throw new Error(`no precomputed vector for "${text}". Try one of: ${keys.slice(0, 5).join(' / ')}`);
}

$('go').addEventListener('click', async () => {
  const text = $('q').value.trim();
  $('results').innerHTML = '';
  try {
    const vec = await embedQuery(text);
    const r = await search(
      vec, 10, parseInt($('l').value, 10), parseInt($('beam').value, 10),
      $('rerank').checked ? 50 : 0);
    const fmt = (st) => st
      ? `${st.requests} req / ${(st.bytes / 1024).toFixed(1)} KB`
      : 'unavailable';
    $('cost').textContent =
      `${r.ms.toFixed(0)} ms · ${r.hops} round-trips · ${r.nodes} records · ` +
      `traversal ${fmt(r.traversal)} · results+rerank ${fmt(r.display)}`;
    $('results').innerHTML = r.ids.map((id, i) =>
      `<li><span class="rank">${i + 1}</span><span class="did">#${id}</span>` +
      `<span class="body">${(r.bodies.get(id) || '').slice(0, 220)}</span></li>`).join('');
  } catch (e) {
    $('results').innerHTML = `<li class="err">${e.message}</li>`;
  }
});

$('resident').addEventListener('click', loadResident);
$('transport').addEventListener('change', (e) => {
  transport = e.target.value;
  log(`transport: ${transport}`);
});
boot().catch((e) => log('ERROR: ' + (e && e.message ? e.message : e)));
