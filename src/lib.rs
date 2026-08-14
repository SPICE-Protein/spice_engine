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
pub use builder::{BuildOptions, build_system, build_mutant_by_solvent_reuse};
pub use domain::{EnvGrid, StabilityConfig, StabilityPoint, is_stable, scan_stability};
pub use engine::{SpiceEngine, StepResult};
pub use env::EnvParams;
pub use equilibrate::{EquilConfig, equilibrate};
pub use metrics::{Metrics, MetricsConfig, MetricsResult};
pub use mutate::{Mutation, apply_mutations, validate_sequence};
pub use pool::{EnginePool, EngineWorker};
pub use structure::{AtomInput, StructureInput, atoms_to_mmcif, build_from_input};
pub use topology::{ProteinTopology, ResidueInfo};

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub fn log_print(msg: String) {
    #[cfg(feature = "python")]
    {
        pyo3::Python::attach(|py| {
            if let Ok(sys) = py.import("sys") {
                if let Ok(stdout) = sys.getattr("stdout") {
                    let _ = stdout.call_method1("write", (format!("{}\n", msg),));
                    let _ = stdout.call_method1("flush", ());
                    return;
                }
            }
        });
    }
    println!("{}", msg);
}

pub fn log_eprint(msg: String) {
    #[cfg(feature = "python")]
    {
        pyo3::Python::attach(|py| {
            if let Ok(sys) = py.import("sys") {
                if let Ok(stderr) = sys.getattr("stderr") {
                    let _ = stderr.call_method1("write", (format!("{}\n", msg),));
                    let _ = stderr.call_method1("flush", ());
                    return;
                }
            }
        });
    }
    eprintln!("{}", msg);
}
