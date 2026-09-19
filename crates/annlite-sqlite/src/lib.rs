//! SQLite storage for annlite indexes.
//!
//! Ordinary tables, no virtual table and no loadable extension, so a stock SQLite
//! build -- including the WASM one a browser uses over HTTP ranges -- can read the
//! result. The index lives in how records are laid out, not in the engine.

pub mod format;
pub mod late_search;
pub mod late_store;
pub mod search;
pub mod store;
