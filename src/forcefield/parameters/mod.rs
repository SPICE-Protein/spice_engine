//! Existing force-field parameter families and their SE adapters.
//!
//! Each existing force field gets one implementation file named after the
//! force field, for example `amber19.rs`, `charmm36.rs`, or `oplsaa.rs`.
//! The files under `data/` are copied parameter assets; `params.rs` is the
//! migrated parameter-loading baseline.

pub mod amber19;
pub mod charmm36m;
pub mod content;
pub mod martini3;

pub use amber19::{Amber19, Amber19Prepared};
pub use charmm36m::{Charmm36m, Charmm36mPrepared};
pub use content::{ComputationContent, ForceFieldSelection, ParameterDomain};
pub use martini3::{Martini3, Martini3Prepared};

// `params.rs` is intentionally kept as a migration source for now. It still
// depends on private `dynamics` internals and is not compiled until its
// imports and bundled-data paths are ported to SE types.
