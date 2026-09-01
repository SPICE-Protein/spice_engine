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
    forces::force_e_lj,
    solvent::{ForcesOnWaterMol, O_EPS, O_H_R, O_SIGMA, WaterMolOpc, WaterSite},
    validate_mol_start_indices,
};
#[cfg(target_arch = "x86_64")]
use crate::engine::md_core::{AtomDynamicsx8, AtomDynamicsx16};

mod dispatch;
mod kernels;
mod pme;
mod types;

pub(super) use dispatch::*;
pub(super) use kernels::*;
pub(super) use pme::*;
pub(crate) use types::*;
