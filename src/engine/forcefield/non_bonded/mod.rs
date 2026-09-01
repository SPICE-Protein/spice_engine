//! For VDW and Coulomb forces

use std::ops::AddAssign;

use ewald::{PmeRecip, force_coulomb_short_range, get_grid_n};
#[allow(unused)]
#[cfg(target_arch = "x86_64")]
use lin_alg::f32::{Vec3x8, Vec3x16, f32x8, f32x16};
use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64};
use rayon::prelude::*;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use wide::f32x8 as WideF32x8;

// Shared type and kernel definitions. Kept in one namespace so private helpers retain
// their existing visibility and call graph while the file is split by responsibility.
include!("types.inc.rs");
include!("dispatch.inc.rs");
include!("pme.inc.rs");
include!("kernels.inc.rs");
