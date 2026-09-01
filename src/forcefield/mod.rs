//! SE-side pluggable force-field boundary.
//!
//! The current implementation is an adapter/migration layer around existing
//! force fields; SE does not yet define an original force field.
//!
//! Source is organized by physical responsibility: bonded interactions,
//! non-bonded interactions, parameter loading/data, and SIMD computation.
//!
//! CPU SIMD capability selection is portable across x86_64 and aarch64; the
//! numerical kernels are added behind that boundary incrementally.

//! Copied migration sources are kept in these semantic directories but are
//! not all compiled yet; each still has dependencies on private `dynamics`
//! internals that will be replaced incrementally.

pub mod neighbors;
pub mod nonbonded;
pub mod parameters;
pub mod simd;
pub mod traits;
pub mod types;

pub use self::neighbors::{
    AtomBlock, BlockForceAccumulator, CsrNeighborList, DirectedPair, PairBlock, atom_block_index,
    make_atom_blocks, partition_directed_pairs,
};
pub use parameters::{
    Amber19, Amber19Prepared, AmberAngle, AmberBond, AmberDihedral, AmberMass, AmberNonbonded,
    AmberParameterIndex, Charmm36m, Charmm36mPrepared, CoarseGrainedBead, CoarseGrainedTopology,
    ComputationContent, ForceFieldSelection, Martini3, Martini3Prepared, ParameterDomain,
    build_coarse_grained_topology,
};
pub use simd::{PairBatch8, PairResult8, SimdBackend};
#[cfg(target_arch = "x86_64")]
pub use simd::{PairBatch16, PairResult16};
pub use traits::{ForceField, PreparedForceField};
pub use types::{
    EnergyVirial, ForceAtom, ForceBuffer, ForceFieldError, ForceFieldRegion, ForceFieldSystem,
    Resolution,
};
