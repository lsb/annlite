// Run a query against a remote SQLite file through the bounded-range VFS, and
// report what crossed the network.
//
// The comparison this exists for is RESEARCH_LOG.md section 13.2: sql.js-httpvfs
// escalates its read-ahead until it has fetched the whole database, so the index's
// access pattern never reaches the wire. This client fetches exactly what SQLite
// asks for, so the numbers it reports are the index's own.
//
// Two independent counts are printed for every run: the VFS's own (what SQLite
// asked for) and the server's request log (what actually arrived). They are
// produced by different processes and must agree -- the same discipline section 9.3
// used when it checked a pass-through VFS against SQLITE_DBSTATUS_CACHE_MISS.
//
//   node measure.js --url http://127.0.0.1:8080/annlite-demo.db --sql "SELECT ..."

const path = require("path");
const createAnnliteSqlite = require("./annlite-sqlite.js");

function arg(name, dflt) {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] !== undefined ? process.argv[i + 1] : dflt;
}

async function main() {
  const url = arg("--url", null);
  const sql = arg("--sql", "SELECT count(*) FROM sqlite_master");
  const json = process.argv.includes("--json");
  if (!url) {
    console.error("usage: node measure.js --url URL [--sql SQL] [--json]");
    process.exit(2);
  }

  const Module = await createAnnliteSqlite();

  // --- transport ---------------------------------------------------------
  // One request per call, for exactly the bytes asked for. No coalescing and no
  // caching: a cache here would silently turn a repeat fetch into a free one and
  // the page counts would stop meaning anything.
  let httpRequests = 0;
  let httpBytes = 0;

  Module.annliteSize = async (u) => {
    const r = await fetch(u, { method: "HEAD" });
    if (!r.ok) throw new Error(`HEAD ${u} -> ${r.status}`);
    const len = r.headers.get("content-length");
    if (len === null) throw new Error(`no content-length for ${u}`);
    return Number(len);
  };

  Module.annliteRange = async (u, offset, nByte) => {
    const end = offset + nByte - 1;
    const r = await fetch(u, { headers: { Range: `bytes=${offset}-${end}` } });
    if (r.status !== 206 && r.status !== 200) {
      throw new Error(`GET ${u} bytes=${offset}-${end} -> ${r.status}`);
    }
    const buf = new Uint8Array(await r.arrayBuffer());
    httpRequests++;
    httpBytes += buf.length;
    return buf;
  };

  Module.annliteOnError = (msg) => console.error("transport:", msg);

  // --- open through the VFS ----------------------------------------------
  const rc = Module.ccall("annlite_register_vfs", "number", [], []);
  if (rc !== 0) throw new Error(`sqlite3_vfs_register -> ${rc}`);
  const vfsName = Module.UTF8ToString(Module.ccall("annlite_vfs_name", "number", [], []));

  const ppDb = Module._malloc(4);
  // SQLITE_OPEN_READONLY = 1. The file is immutable; anything else would make
  // SQLite look for a journal it cannot have.
  const openRc = await Module.ccall(
    "sqlite3_open_v2",
    "number",
    ["string", "number", "number", "string"],
    [url, ppDb, 1, vfsName],
    { async: true }
  );
  const db = Module.getValue(ppDb, "i32");
  Module._free(ppDb);
  if (openRc !== 0) {
    const msg = Module.UTF8ToString(Module.ccall("sqlite3_errmsg", "number", ["number"], [db]));
    throw new Error(`sqlite3_open_v2 -> ${openRc}: ${msg}`);
  }

  // Opening reads the header and the schema. Those pages are a real cost, but they
  // are paid once per session rather than once per query, so they are reported
  // separately instead of being folded into the query's own page count.
  const openStats = readStats(Module);
  const openHttp = { requests: httpRequests, bytes: httpBytes };
  Module.ccall("annlite_stats_reset", null, [], []);
  httpRequests = 0;
  httpBytes = 0;

  // --- run the query ------------------------------------------------------
  const ppStmt = Module._malloc(4);
  const prepRc = await Module.ccall(
    "sqlite3_prepare_v2",
    "number",
    ["number", "string", "number", "number", "number"],
    [db, sql, -1, ppStmt, 0],
    { async: true }
  );
  if (prepRc !== 0) {
    const msg = Module.UTF8ToString(Module.ccall("sqlite3_errmsg", "number", ["number"], [db]));
    throw new Error(`sqlite3_prepare_v2 -> ${prepRc}: ${msg}`);
  }
  const stmt = Module.getValue(ppStmt, "i32");
  Module._free(ppStmt);

  const rows = [];
  for (;;) {
    const step = await Module.ccall("sqlite3_step", "number", ["number"], [stmt], { async: true });
    if (step === 101 /* SQLITE_DONE */) break;
    if (step !== 100 /* SQLITE_ROW */) {
      const msg = Module.UTF8ToString(Module.ccall("sqlite3_errmsg", "number", ["number"], [db]));
      throw new Error(`sqlite3_step -> ${step}: ${msg}`);
    }
    const n = Module.ccall("sqlite3_column_count", "number", ["number"], [stmt]);
    const row = [];
    for (let i = 0; i < n; i++) {
      const p = Module.ccall("sqlite3_column_text", "number", ["number", "number"], [stmt, i]);
      row.push(p ? Module.UTF8ToString(p) : null);
    }
    rows.push(row);
  }

  const queryStats = readStats(Module);
  const queryHttp = { requests: httpRequests, bytes: httpBytes };

  await Module.ccall("sqlite3_finalize", "number", ["number"], [stmt], { async: true });
  await Module.ccall("sqlite3_close_v2", "number", ["number"], [db], { async: true });

  // The VFS counts what SQLite asked for; the transport counts what was fetched.
  // A mismatch means something between them is caching or coalescing, which would
  // invalidate every page number here.
  const agree =
    queryStats.requests === queryHttp.requests && queryStats.bytes === queryHttp.bytes;

  const out = {
    sqlite_version_source: "libsqlite3-sys 0.30.1 amalgamation (3.46.0)",
    vfs: vfsName,
    url,
    sql,
    rows: rows.length,
    open: { ...openStats, http_requests: openHttp.requests, http_bytes: openHttp.bytes },
    query: { ...queryStats, http_requests: queryHttp.requests, http_bytes: queryHttp.bytes },
    vfs_and_transport_agree: agree,
  };

  if (json) {
    console.log(JSON.stringify(out));
  } else {
    console.log(`vfs             ${vfsName}`);
    console.log(`sqlite          3.46.0 (same amalgamation as the native benchmarks)`);
    console.log(`rows            ${rows.length}`);
    console.log(
      `open            ${openStats.pages} pages, ${openStats.requests} requests, ` +
        `${openStats.bytes.toLocaleString()} bytes`
    );
    console.log(
      `query           ${queryStats.pages} pages, ${queryStats.requests} requests, ` +
        `${queryStats.bytes.toLocaleString()} bytes`
    );
    console.log(`vfs == transport ${agree ? "yes" : "NO -- something is caching"}`);
    if (rows.length && rows.length <= 5) {
      for (const r of rows) console.log("                " + r.join(" | "));
    }
  }
  if (!agree) process.exit(1);
}

function readStats(Module) {
  return {
    reads: Module.ccall("annlite_stat_reads", "number", [], []),
    requests: Module.ccall("annlite_stat_requests", "number", [], []),
    pages: Module.ccall("annlite_stat_pages", "number", [], []),
    bytes: Module.ccall("annlite_stat_bytes", "number", [], []),
  };
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
