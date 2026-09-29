//! Non-bonded force evaluation and PME support.
//!
//! The implementation is split by responsibility while keeping the original
//! private namespace and numerical call graph intact.

use std::ops::AddAssign;

use ewald::{PmeRecip, force_coulomb_short_range, get_grid_n};
#[allow(unused)]
#[cfg(target_arch = "x86_64")]
use lin_alg::f32::{Vec3x8, Vec3x16, f32x8, f32x16};
use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64};
use rayon::prelude::*;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use wide::f32x8 as WideF32x8;

#[cfg(feature = "cuda")]
use crate::engine::md_core::gpu_interface::force_nonbonded_gpu;
use crate::engine::md_core::{
    AtomDynamics, ComputationDevice, MdOverrides, MdState,
    alchemical::{
        SOFT_CORE_ALPHA, SOFT_CORE_POWER, SOFT_CORE_SIGMA_MIN, staged_decoupling_schedule,
    },
    barostat::SimBox,
    forces::{force_e_lj, force_e_lj_c4},
    solvent::{ForcesOnWaterMol, O_EPS, O_H_R, O_SIGMA, WaterMolOpc, WaterSite},
    validate_mol_start_indices,
};
#[cfg(target_arch = "x86_64")]
use crate::engine::md_core::{AtomDynamicsx8, AtomDynamicsx16};

/// Architectures whose `calc_force_cpu_dispatch` evaluates the compact water
/// SIMD batches. `setup_pairs` may remove SIMD-covered pairs from the scalar
/// streams only when a consumer exists; setup and dispatch must agree here or
/// pairs get dropped (or, historically, double-counted on arm64).
///
/// aarch64 was briefly parked (water-SIMD-on lost ~7 ms/eval to the fused
/// scalar path). The loss was never the vector arithmetic: it was rayon
/// fold-state explosion (a dense 616 KB accumulator re-zeroed per
/// work-stealing split) plus a 640k-entry intermediate candidate vector on
/// every rebuild. With split granularity tuned to ~4 states/worker and
/// batches streamed at rebuild, paired 2LYZ A/B on an Apple Silicon box has
/// water SIMD ahead in every run (force eval and step wall, ordinary and
/// rebuild steps). `SPICE_DISABLE_WATER_SIMD=1` (kill) and
/// `SPICE_FORCE_WATER_SIMD=1` (opt-in on parked arches) remain for A/B,
/// plus `SPICE_WATER_SIMD_ONLY=ws|ww` and `SPICE_WATER_SIMD_MINLEN` for
/// cost attribution.
pub(crate) const WATER_SIMD_ACTIVE: bool =
    cfg!(any(target_arch = "x86_64", target_arch = "aarch64"));

mod dispatch;
mod kernels;
mod pme;
mod types;
mod water_simd;
pub(super) use dispatch::*;
pub(crate) use kernels::combine_lj_params;
pub(super) use kernels::*;
pub(crate) use types::*;
pub(crate) use water_simd::*;
