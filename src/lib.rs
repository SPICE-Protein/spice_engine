//! SPICE engine — Rust MD core built on the `dynamics` crate.
//!
//! Provides environment-parameterized MD: build a system from an mmCIF structure,
//! step it (optionally injecting per-atom bias forces for RL), read back physical
//! metrics (five-dimensional `M`), and map RL actions to bias forces.

pub mod actions;
pub mod builder;
pub mod domain;
pub mod engine;
pub mod env;
pub mod equilibrate;
pub mod metrics;
pub mod mutate;
pub mod pool;
pub mod structure;
pub mod topology;

#[cfg(feature = "python")]
pub mod ffi;

pub use actions::{ActionMask, EnvDelta, ForceAction};
pub use builder::{BuildOptions, build_system};
pub use domain::{EnvGrid, StabilityConfig, StabilityPoint, is_stable, scan_stability};
pub use engine::{SpiceEngine, StepResult};
pub use env::EnvParams;
pub use equilibrate::{EquilConfig, equilibrate};
pub use metrics::{Metrics, MetricsConfig, MetricsResult};
pub use mutate::{Mutation, apply_mutations, validate_sequence};
pub use pool::{EnginePool, EngineWorker};
pub use structure::{AtomInput, StructureInput, atoms_to_mmcif, build_from_input};
pub use topology::{ProteinTopology, ResidueInfo};
