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
pub mod forcefield;
pub mod metrics;
pub mod mutate;
pub mod pocket;
pub mod pool;
pub mod progress;
pub mod rna;
pub mod structure;
pub mod topology;

#[cfg(feature = "python")]
pub mod ffi;

/// Browser extern-C API surface (v1.3.9 web port). Only ever compiled for the
/// `wasm32-unknown-unknown` target with the `web` feature; native and the pyo3
/// wheel never see it. See `src/web.rs` for the calling contract.
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub mod web;

pub use actions::{ActionMask, EnvDelta, ForceAction};
pub use builder::{BuildOptions, build_mutant_by_solvent_reuse, build_system};
pub use domain::{EnvGrid, StabilityConfig, StabilityPoint, is_stable, scan_stability};
pub use engine::{SpiceEngine, StepResult};
pub use env::EnvParams;
pub use equilibrate::{EquilConfig, equilibrate};
pub use forcefield::{
    Amber19, AmberAngle, AmberBond, AmberDihedral, AmberMass, AmberNonbonded, AmberParameterIndex,
    AtomBlock, BlockForceAccumulator, Charmm36m, CoarseGrainedBead, CoarseGrainedTopology,
    ComputationContent, CsrNeighborList, DirectedPair, ForceAtom, ForceField, ForceFieldRegion,
    ForceFieldSelection, Martini3, PairBatch8, PairBlock, PairResult8, ParameterDomain,
    PreparedForceField, Resolution, SimdBackend, atom_block_index, build_coarse_grained_topology,
    make_atom_blocks, partition_directed_pairs,
};
#[cfg(target_arch = "x86_64")]
pub use forcefield::{PairBatch16, PairResult16};
pub use metrics::{Metrics, MetricsConfig, MetricsResult};
pub use mutate::{Mutation, apply_mutations, validate_sequence};
pub use pocket::{
    AdvancedPocketFeatures, GridStatus, NativePocket, PocketDelta, analyze_pocket_trajectory,
    calculate_advanced_features, calculate_engine_pockets, calculate_pocket_delta,
    calculate_pockets_native, is_atom_hydrophobic,
};
pub use pool::{EnginePool, EngineWorker};
pub use rna::{RnaStructureInput, build_rna_from_input, validate_rna_input};
pub use structure::{AtomInput, StructureInput, atoms_to_mmcif, build_from_input};
pub use topology::{ProteinTopology, ResidueInfo};

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub fn log_print(msg: String) {
    #[cfg(all(target_arch = "wasm32", feature = "web"))]
    {
        // The browser host has no stdout; route to the console via the same
        // `env.log` import `src/web.rs` uses (declared here too because
        // `log_print` is a crate-wide helper that compiles independently).
        #[link(wasm_import_module = "env")]
        unsafe extern "C" {
            fn log(ptr: *const u8, len: usize);
        }
        // Safety: `log` is a pure JS binding over a borrowed slice we keep alive.
        unsafe { log(msg.as_ptr(), msg.len()) };
        return;
    }
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
    #[cfg(not(all(target_arch = "wasm32", feature = "web")))]
    println!("{}", msg);
}

pub fn log_eprint(msg: String) {
    #[cfg(all(target_arch = "wasm32", feature = "web"))]
    {
        #[link(wasm_import_module = "env")]
        unsafe extern "C" {
            fn log(ptr: *const u8, len: usize);
        }
        // Safety: `log` is a pure JS binding over a borrowed slice we keep alive.
        unsafe { log(msg.as_ptr(), msg.len()) };
        return;
    }
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
    #[cfg(not(all(target_arch = "wasm32", feature = "web")))]
    eprintln!("{}", msg);
}
