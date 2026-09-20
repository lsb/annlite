/*
** A read-only SQLite VFS that fetches exactly the bytes SQLite asks for.
**
** RESEARCH_LOG.md section 13.2 found that sql.js-httpvfs treats `requestChunkSize`
** as a floor rather than a cap: its speculative read heads double their request size
** until they have swallowed the file, so the first query against a 5.9 MB database
** pulled about 5.2 MB. Any amount of page-locality work in the index is invisible to
** such a client, which makes it the wrong instrument for the question this project
** is about -- and it left section 13.4's crossover question unanswerable through a
** real SQLite client.
**
** This VFS is the other instrument. `xRead(offset, amt)` issues one HTTP Range
** request for exactly `[offset, offset+amt)` and nothing more: no read-ahead, no
** speculation, no growth. What SQLite asks for is what crosses the network, so the
** request count, the byte count and the page numbers the traversal touches are the
** index's own access pattern rather than a client's caching policy laid over it.
**
** It is deliberately *not* a good general-purpose client -- a real deployment wants
** read-ahead, and section 13.4 shows why. It is a measuring instrument, and its value
** is that it adds nothing to what it measures.
**
** Built against the same SQLite amalgamation the native benchmarks link (3.46.0, the
** one libsqlite3-sys bundles), so a difference between the browser and native page
** counts is a difference in the VFS and cannot be a difference in the engine.
**
** Asynchronous I/O under a synchronous API is Emscripten's ASYNCIFY: xRead suspends
** the whole WASM stack while the fetch is in flight and resumes where it left off.
** That is why this must be linked with -sASYNCIFY, and why the entry points that can
** reach xRead are listed in ASYNCIFY_EXPORTS.
*/

#include "sqlite3.h"
#include <emscripten.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define HTTPRANGE_VFS_NAME "httprange"
#define PAGE_BYTES 4096
/* Page numbers observed, for the distinct-page count. Sized for a database of
** 4 GiB at 4 KiB pages; a bigger one would still read correctly, it would simply
** stop counting distinct pages beyond the bitmap. */
#define MAX_PAGES (1u << 20)

/* ---------------------------------------------------------------------------
** Host hooks. The JS side supplies the transport so the same WASM runs in a
** browser against fetch() and in Node against the network simulator.
** ------------------------------------------------------------------------ */

EM_ASYNC_JS(int, annlite_http_read, (const char *zUrl, double iOfst, int nByte, void *pOut), {
  const url = UTF8ToString(zUrl);
  try {
    const bytes = await Module.annliteRange(url, iOfst, nByte);
    if (!bytes || bytes.length < nByte) return 1;
    HEAPU8.set(bytes.subarray(0, nByte), pOut);
    return 0;
  } catch (e) {
    if (Module.annliteOnError) Module.annliteOnError(String(e));
    return 1;
  }
});

EM_ASYNC_JS(double, annlite_http_size, (const char *zUrl), {
  const url = UTF8ToString(zUrl);
  try {
    return await Module.annliteSize(url);
  } catch (e) {
    if (Module.annliteOnError) Module.annliteOnError(String(e));
    return -1;
  }
});

/* ---------------------------------------------------------------------------
** Access accounting. Counted here rather than in JS because this is the layer
** that knows the page arithmetic, and because it must count what SQLite asked
** for even when the host coalesces or caches underneath.
** ------------------------------------------------------------------------ */

typedef struct Stats {
  unsigned int reads;        /* xRead calls */
  unsigned int requests;     /* range requests actually issued */
  double bytes;              /* bytes requested */
  unsigned int distinct;     /* distinct 4 KiB pages touched */
  unsigned char *seen;       /* page bitmap, lazily allocated */
} Stats;

static Stats g_stats;

static void stats_note(sqlite3_int64 iOfst, int nByte) {
  g_stats.reads++;
  g_stats.requests++;
  g_stats.bytes += nByte;
  if (!g_stats.seen) {
    g_stats.seen = (unsigned char *)calloc(MAX_PAGES / 8, 1);
    if (!g_stats.seen) return;
  }
  sqlite3_int64 first = iOfst / PAGE_BYTES;
  sqlite3_int64 last = (iOfst + nByte - 1) / PAGE_BYTES;
  for (sqlite3_int64 p = first; p <= last && p < MAX_PAGES; p++) {
    if (!(g_stats.seen[p >> 3] & (1 << (p & 7)))) {
      g_stats.seen[p >> 3] |= (1 << (p & 7));
      g_stats.distinct++;
    }
  }
}

EMSCRIPTEN_KEEPALIVE void annlite_stats_reset(void) {
  if (g_stats.seen) memset(g_stats.seen, 0, MAX_PAGES / 8);
  g_stats.reads = g_stats.requests = g_stats.distinct = 0;
  g_stats.bytes = 0;
}

EMSCRIPTEN_KEEPALIVE unsigned int annlite_stat_reads(void)    { return g_stats.reads; }
EMSCRIPTEN_KEEPALIVE unsigned int annlite_stat_requests(void) { return g_stats.requests; }
EMSCRIPTEN_KEEPALIVE unsigned int annlite_stat_pages(void)    { return g_stats.distinct; }
EMSCRIPTEN_KEEPALIVE double       annlite_stat_bytes(void)    { return g_stats.bytes; }

/* ---------------------------------------------------------------------------
** The file object.
** ------------------------------------------------------------------------ */

typedef struct HttpFile {
  sqlite3_file base;
  char *zUrl;
  sqlite3_int64 nSize;
} HttpFile;

static int httpClose(sqlite3_file *pFile) {
  HttpFile *p = (HttpFile *)pFile;
  sqlite3_free(p->zUrl);
  p->zUrl = 0;
  return SQLITE_OK;
}

static int httpRead(sqlite3_file *pFile, void *zBuf, int iAmt, sqlite3_int64 iOfst) {
  HttpFile *p = (HttpFile *)pFile;
  /* A read past EOF must report SHORT_READ with the tail zeroed; SQLite relies on
  ** this when it probes a file whose size it does not yet know. */
  if (iOfst >= p->nSize) {
    memset(zBuf, 0, iAmt);
    return SQLITE_IOERR_SHORT_READ;
  }
  int n = iAmt;
  if (iOfst + iAmt > p->nSize) n = (int)(p->nSize - iOfst);
  stats_note(iOfst, n);
  if (annlite_http_read(p->zUrl, (double)iOfst, n, zBuf) != 0) return SQLITE_IOERR_READ;
  if (n < iAmt) {
    memset((char *)zBuf + n, 0, iAmt - n);
    return SQLITE_IOERR_SHORT_READ;
  }
  return SQLITE_OK;
}

/* Read-only: a database served over range requests cannot be written to, and
** saying so up front is better than failing halfway through a transaction. */
static int httpWrite(sqlite3_file *f, const void *b, int n, sqlite3_int64 o) {
  (void)f; (void)b; (void)n; (void)o;
  return SQLITE_READONLY;
}
static int httpTruncate(sqlite3_file *f, sqlite3_int64 size) { (void)f; (void)size; return SQLITE_READONLY; }
static int httpSync(sqlite3_file *f, int flags) { (void)f; (void)flags; return SQLITE_OK; }

static int httpFileSize(sqlite3_file *pFile, sqlite3_int64 *pSize) {
  *pSize = ((HttpFile *)pFile)->nSize;
  return SQLITE_OK;
}

static int httpLock(sqlite3_file *f, int l) { (void)f; (void)l; return SQLITE_OK; }
static int httpUnlock(sqlite3_file *f, int l) { (void)f; (void)l; return SQLITE_OK; }
static int httpCheckReservedLock(sqlite3_file *f, int *pRes) { (void)f; *pRes = 0; return SQLITE_OK; }
static int httpFileControl(sqlite3_file *f, int op, void *pArg) { (void)f; (void)op; (void)pArg; return SQLITE_NOTFOUND; }
static int httpSectorSize(sqlite3_file *f) { (void)f; return PAGE_BYTES; }

static int httpDeviceCharacteristics(sqlite3_file *f) {
  (void)f;
  return SQLITE_IOCAP_IMMUTABLE;
}

static const sqlite3_io_methods httpIoMethods = {
  1,                          /* iVersion */
  httpClose,
  httpRead,
  httpWrite,
  httpTruncate,
  httpSync,
  httpFileSize,
  httpLock,
  httpUnlock,
  httpCheckReservedLock,
  httpFileControl,
  httpSectorSize,
  httpDeviceCharacteristics,
  0, 0, 0, 0, 0, 0
};

/* ---------------------------------------------------------------------------
** The VFS.
** ------------------------------------------------------------------------ */

static int httpOpen(sqlite3_vfs *pVfs, const char *zName, sqlite3_file *pFile,
                    int flags, int *pOutFlags) {
  (void)pVfs;
  HttpFile *p = (HttpFile *)pFile;
  memset(p, 0, sizeof(*p));
  /* Journals and temp files never exist for an immutable remote database. */
  if (!(flags & SQLITE_OPEN_MAIN_DB) || zName == 0) return SQLITE_CANTOPEN;

  double sz = annlite_http_size(zName);
  if (sz < 0) return SQLITE_CANTOPEN;

  p->zUrl = sqlite3_mprintf("%s", zName);
  if (!p->zUrl) return SQLITE_NOMEM;
  p->nSize = (sqlite3_int64)sz;
  p->base.pMethods = &httpIoMethods;
  if (pOutFlags) *pOutFlags = SQLITE_OPEN_READONLY;
  return SQLITE_OK;
}

static int httpDelete(sqlite3_vfs *v, const char *z, int s) { (void)v; (void)z; (void)s; return SQLITE_READONLY; }

static int httpAccess(sqlite3_vfs *v, const char *z, int flags, int *pResOut) {
  (void)v; (void)z; (void)flags;
  /* No journal or WAL can accompany an immutable file, and claiming otherwise
  ** makes SQLite go looking for one over the network at every open. */
  *pResOut = 0;
  return SQLITE_OK;
}

static int httpFullPathname(sqlite3_vfs *v, const char *zIn, int nOut, char *zOut) {
  (void)v;
  sqlite3_snprintf(nOut, zOut, "%s", zIn);
  return SQLITE_OK;
}

static int httpRandomness(sqlite3_vfs *v, int nByte, char *zOut) {
  (void)v;
  memset(zOut, 0, nByte);
  return nByte;
}
static int httpSleep(sqlite3_vfs *v, int micro) { (void)v; (void)micro; return 0; }
static int httpCurrentTime(sqlite3_vfs *v, double *p) { (void)v; *p = 2440587.5; return SQLITE_OK; }

static sqlite3_vfs httpVfs = {
  1,                       /* iVersion */
  sizeof(HttpFile),        /* szOsFile */
  2048,                    /* mxPathname */
  0,                       /* pNext */
  HTTPRANGE_VFS_NAME,
  0,                       /* pAppData */
  httpOpen,
  httpDelete,
  httpAccess,
  httpFullPathname,
  0, 0, 0, 0,              /* dl* : no loadable extensions in WASM */
  httpRandomness,
  httpSleep,
  httpCurrentTime,
  0
};

EMSCRIPTEN_KEEPALIVE int annlite_register_vfs(void) {
  return sqlite3_vfs_register(&httpVfs, 0);
}

EMSCRIPTEN_KEEPALIVE const char *annlite_vfs_name(void) { return HTTPRANGE_VFS_NAME; }
