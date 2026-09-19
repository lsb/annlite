//! A pass-through SQLite VFS that records every `xRead` against the main database
//! file.
//!
//! Why a VFS rather than a counter inside the query loop: the question this baseline
//! has to answer is "how many round-trips would a browser reading this file over HTTP
//! ranges pay for one query?", and only SQLite itself knows which pages its pager
//! decides to fetch. Anything we could compute from the schema would be a guess.
//! `xRead` is the exact seam a `sql.js-httpvfs`-style client hooks, so counting there
//! measures the same events the network would have to serve.
//!
//! Three deliberate choices shape what the numbers mean:
//!
//! * **`iVersion = 1`.** The shim advertises the oldest I/O-methods version, which
//!   omits `xFetch`/`xUnfetch`. SQLite therefore cannot memory-map the database and
//!   must route every page through `xRead` — the same situation a network client is
//!   in. With mmap enabled, pages served from the map are invisible to any VFS
//!   counter, so the counts would silently under-report.
//! * **Main database only.** Reads of journals, temp files and the schema of other
//!   attached files are ignored; `SQLITE_OPEN_MAIN_DB` in the open flags is the test.
//! * **Raw reads, not pages.** The recorder stores `(offset, length)` pairs exactly as
//!   the pager asked for them. Mapping to page numbers happens in [`ReadTrace`], so
//!   short reads (notably the 100-byte header read of page 1) are visible rather than
//!   rounded away.
//!
//! The recorder is global because a VFS is global; the benchmark is single-threaded
//! and takes the lock on every read, which is irrelevant next to the work SQLite does
//! per page.

use rusqlite::ffi;
use std::collections::BTreeSet;
use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::sync::{Mutex, OnceLock};

/// Name the shim registers under; pass it to `Connection::open_with_flags_and_vfs`.
pub const VFS_NAME: &str = "annlite-count";

/// One `xRead` call against the main database file.
#[derive(Clone, Copy, Debug)]
pub struct Read {
    pub offset: i64,
    pub len: i32,
}

static RECORDER: Mutex<Option<Vec<Read>>> = Mutex::new(None);

/// Start (or restart) recording, discarding anything previously captured.
pub fn record_start() {
    *RECORDER.lock().unwrap() = Some(Vec::new());
}

/// Stop recording and return what was captured since [`record_start`].
pub fn record_take() -> ReadTrace {
    ReadTrace { reads: RECORDER.lock().unwrap().take().unwrap_or_default() }
}

/// The reads one measured operation caused, and the page-level views of them.
#[derive(Debug, Default)]
pub struct ReadTrace {
    pub reads: Vec<Read>,
}

impl ReadTrace {
    /// Number of `xRead` calls, including repeat reads of the same page.
    pub fn read_calls(&self) -> usize {
        self.reads.len()
    }

    /// Total bytes handed back by the VFS.
    pub fn bytes(&self) -> u64 {
        self.reads.iter().map(|r| r.len.max(0) as u64).sum()
    }

    /// 1-based page numbers touched, deduplicated and sorted.
    ///
    /// A read is attributed to every page it overlaps, which matters only for the
    /// header read at offset 0 (100 bytes, page 1) and never spans pages otherwise:
    /// SQLite's pager reads whole aligned pages.
    pub fn pages(&self, page_size: u64) -> BTreeSet<u64> {
        let mut out = BTreeSet::new();
        for r in &self.reads {
            if r.offset < 0 || r.len <= 0 {
                continue;
            }
            let first = r.offset as u64 / page_size;
            let last = (r.offset as u64 + r.len as u64 - 1) / page_size;
            for p in first..=last {
                out.insert(p + 1);
            }
        }
        out
    }

    /// Number of maximal runs of consecutive page numbers.
    ///
    /// This is the round-trip count a client that coalesces adjacent pages into one
    /// range request would pay, so it is the optimistic bound to set against the
    /// pessimistic one page = one request count.
    pub fn contiguous_runs(&self, page_size: u64) -> usize {
        let pages = self.pages(page_size);
        let mut runs = 0usize;
        let mut prev: Option<u64> = None;
        for p in pages {
            if prev.map_or(true, |q| p != q + 1) {
                runs += 1;
            }
            prev = Some(p);
        }
        runs
    }
}

#[repr(C)]
struct CountingFile {
    /// Must be first: SQLite casts `sqlite3_file*` to and from this struct.
    base: ffi::sqlite3_file,
    is_main_db: c_int,
    // The underlying VFS's file object lives immediately after this struct; the
    // registered `szOsFile` reserves room for it.
}

unsafe fn real_file(p: *mut ffi::sqlite3_file) -> *mut ffi::sqlite3_file {
    (p as *mut u8).add(std::mem::size_of::<CountingFile>()) as *mut ffi::sqlite3_file
}

unsafe fn real_methods(p: *mut ffi::sqlite3_file) -> &'static ffi::sqlite3_io_methods {
    &*(*real_file(p)).pMethods
}

unsafe extern "C" fn x_close(p: *mut ffi::sqlite3_file) -> c_int {
    let rc = (real_methods(p).xClose.unwrap())(real_file(p));
    (*p).pMethods = ptr::null();
    rc
}

unsafe extern "C" fn x_read(p: *mut ffi::sqlite3_file, buf: *mut c_void, amt: c_int, ofst: ffi::sqlite3_int64) -> c_int {
    if (*(p as *mut CountingFile)).is_main_db != 0 {
        if let Some(log) = RECORDER.lock().unwrap().as_mut() {
            log.push(Read { offset: ofst, len: amt });
        }
    }
    (real_methods(p).xRead.unwrap())(real_file(p), buf, amt, ofst)
}

unsafe extern "C" fn x_write(p: *mut ffi::sqlite3_file, buf: *const c_void, amt: c_int, ofst: ffi::sqlite3_int64) -> c_int {
    (real_methods(p).xWrite.unwrap())(real_file(p), buf, amt, ofst)
}

unsafe extern "C" fn x_truncate(p: *mut ffi::sqlite3_file, size: ffi::sqlite3_int64) -> c_int {
    (real_methods(p).xTruncate.unwrap())(real_file(p), size)
}

unsafe extern "C" fn x_sync(p: *mut ffi::sqlite3_file, flags: c_int) -> c_int {
    (real_methods(p).xSync.unwrap())(real_file(p), flags)
}

unsafe extern "C" fn x_file_size(p: *mut ffi::sqlite3_file, size: *mut ffi::sqlite3_int64) -> c_int {
    (real_methods(p).xFileSize.unwrap())(real_file(p), size)
}

unsafe extern "C" fn x_lock(p: *mut ffi::sqlite3_file, level: c_int) -> c_int {
    (real_methods(p).xLock.unwrap())(real_file(p), level)
}

unsafe extern "C" fn x_unlock(p: *mut ffi::sqlite3_file, level: c_int) -> c_int {
    (real_methods(p).xUnlock.unwrap())(real_file(p), level)
}

unsafe extern "C" fn x_check_reserved_lock(p: *mut ffi::sqlite3_file, out: *mut c_int) -> c_int {
    (real_methods(p).xCheckReservedLock.unwrap())(real_file(p), out)
}

unsafe extern "C" fn x_file_control(p: *mut ffi::sqlite3_file, op: c_int, arg: *mut c_void) -> c_int {
    (real_methods(p).xFileControl.unwrap())(real_file(p), op, arg)
}

unsafe extern "C" fn x_sector_size(p: *mut ffi::sqlite3_file) -> c_int {
    (real_methods(p).xSectorSize.unwrap())(real_file(p))
}

unsafe extern "C" fn x_device_characteristics(p: *mut ffi::sqlite3_file) -> c_int {
    (real_methods(p).xDeviceCharacteristics.unwrap())(real_file(p))
}

static IO_METHODS: ffi::sqlite3_io_methods = ffi::sqlite3_io_methods {
    iVersion: 1,
    xClose: Some(x_close),
    xRead: Some(x_read),
    xWrite: Some(x_write),
    xTruncate: Some(x_truncate),
    xSync: Some(x_sync),
    xFileSize: Some(x_file_size),
    xLock: Some(x_lock),
    xUnlock: Some(x_unlock),
    xCheckReservedLock: Some(x_check_reserved_lock),
    xFileControl: Some(x_file_control),
    xSectorSize: Some(x_sector_size),
    xDeviceCharacteristics: Some(x_device_characteristics),
    // Unused at iVersion 1. xFetch/xUnfetch being absent is what keeps SQLite from
    // memory-mapping the file behind our back.
    xShmMap: None,
    xShmLock: None,
    xShmBarrier: None,
    xShmUnmap: None,
    xFetch: None,
    xUnfetch: None,
};

unsafe fn base_of(vfs: *mut ffi::sqlite3_vfs) -> *mut ffi::sqlite3_vfs {
    (*vfs).pAppData as *mut ffi::sqlite3_vfs
}

unsafe extern "C" fn x_open(
    vfs: *mut ffi::sqlite3_vfs,
    name: ffi::sqlite3_filename,
    file: *mut ffi::sqlite3_file,
    flags: c_int,
    out_flags: *mut c_int,
) -> c_int {
    let base = base_of(vfs);
    let cf = file as *mut CountingFile;
    (*cf).base.pMethods = ptr::null();
    (*cf).is_main_db = i32::from(flags & ffi::SQLITE_OPEN_MAIN_DB != 0);
    let rc = ((*base).xOpen.unwrap())(base, name, real_file(file), flags, out_flags);
    if rc == ffi::SQLITE_OK && !(*real_file(file)).pMethods.is_null() {
        (*cf).base.pMethods = &IO_METHODS;
    }
    rc
}

unsafe extern "C" fn x_delete(vfs: *mut ffi::sqlite3_vfs, name: *const c_char, sync_dir: c_int) -> c_int {
    let base = base_of(vfs);
    ((*base).xDelete.unwrap())(base, name, sync_dir)
}

unsafe extern "C" fn x_access(vfs: *mut ffi::sqlite3_vfs, name: *const c_char, flags: c_int, out: *mut c_int) -> c_int {
    let base = base_of(vfs);
    ((*base).xAccess.unwrap())(base, name, flags, out)
}

unsafe extern "C" fn x_full_pathname(vfs: *mut ffi::sqlite3_vfs, name: *const c_char, n: c_int, out: *mut c_char) -> c_int {
    let base = base_of(vfs);
    ((*base).xFullPathname.unwrap())(base, name, n, out)
}

unsafe extern "C" fn x_randomness(vfs: *mut ffi::sqlite3_vfs, n: c_int, out: *mut c_char) -> c_int {
    let base = base_of(vfs);
    ((*base).xRandomness.unwrap())(base, n, out)
}

unsafe extern "C" fn x_sleep(vfs: *mut ffi::sqlite3_vfs, micros: c_int) -> c_int {
    let base = base_of(vfs);
    ((*base).xSleep.unwrap())(base, micros)
}

unsafe extern "C" fn x_current_time(vfs: *mut ffi::sqlite3_vfs, out: *mut f64) -> c_int {
    let base = base_of(vfs);
    ((*base).xCurrentTime.unwrap())(base, out)
}

unsafe extern "C" fn x_get_last_error(vfs: *mut ffi::sqlite3_vfs, n: c_int, out: *mut c_char) -> c_int {
    let base = base_of(vfs);
    ((*base).xGetLastError.unwrap())(base, n, out)
}

unsafe extern "C" fn x_current_time_int64(vfs: *mut ffi::sqlite3_vfs, out: *mut ffi::sqlite3_int64) -> c_int {
    let base = base_of(vfs);
    ((*base).xCurrentTimeInt64.unwrap())(base, out)
}

static REGISTERED: OnceLock<Result<(), String>> = OnceLock::new();

/// Register the counting VFS (idempotent). Returns the name to open databases with.
///
/// The VFS is registered non-default, so only connections that ask for it by name are
/// instrumented; the build phase uses the stock VFS and pays nothing.
pub fn register() -> Result<&'static str, String> {
    REGISTERED
        .get_or_init(|| unsafe {
            let base = ffi::sqlite3_vfs_find(ptr::null());
            if base.is_null() {
                return Err("sqlite3_vfs_find returned no default VFS".into());
            }
            let name = CString::new(VFS_NAME).unwrap();
            let shim = Box::new(ffi::sqlite3_vfs {
                iVersion: 2,
                szOsFile: std::mem::size_of::<CountingFile>() as c_int + (*base).szOsFile,
                mxPathname: (*base).mxPathname,
                pNext: ptr::null_mut(),
                zName: name.into_raw(),
                pAppData: base as *mut c_void,
                xOpen: Some(x_open),
                xDelete: Some(x_delete),
                xAccess: Some(x_access),
                xFullPathname: Some(x_full_pathname),
                xDlOpen: (*base).xDlOpen,
                xDlError: (*base).xDlError,
                xDlSym: (*base).xDlSym,
                xDlClose: (*base).xDlClose,
                xRandomness: Some(x_randomness),
                xSleep: Some(x_sleep),
                xCurrentTime: Some(x_current_time),
                xGetLastError: Some(x_get_last_error),
                xCurrentTimeInt64: Some(x_current_time_int64),
                xSetSystemCall: None,
                xGetSystemCall: None,
                xNextSystemCall: None,
            });
            // Leaked on purpose: SQLite keeps the pointer for the life of the process
            // and there is no safe moment to free it.
            let rc = ffi::sqlite3_vfs_register(Box::into_raw(shim), 0);
            if rc == ffi::SQLITE_OK {
                Ok(())
            } else {
                Err(format!("sqlite3_vfs_register failed with rc={rc}"))
            }
        })
        .as_ref()
        .map(|_| VFS_NAME)
        .map_err(|e| e.clone())
}
