//! A static file server that answers HTTP range requests over a simulated link.
//!
//! The deployment this project targets is a SQLite file on a CDN read by
//! sql.js-httpvfs in a browser. In that setting the cost of a query is not CPU but
//! the number of HTTP round-trips its page accesses generate, and a round-trip on
//! a phone is 70 ms before any bytes move. Measuring that honestly on a laptop
//! needs two things a normal static server does not give: a link whose latency and
//! bandwidth can be set to a phone's, and a record of every byte range the client
//! asked for.
//!
//! This crate is both. It serves a directory with strict RFC 7233 semantics (see
//! [`range`]), delays and paces each response according to a named profile (see
//! [`profile`]), and writes one JSON line per request (see [`log`]). The
//! simulation is a pure function of the request sequence and a seed — replaying a
//! workload replays its delays — which is what makes two indexes comparable.

pub mod log;
pub mod profile;
pub mod range;
pub mod rng;
pub mod server;

pub use profile::{Mode, Profile, Sim};
pub use server::{Config, Server};
