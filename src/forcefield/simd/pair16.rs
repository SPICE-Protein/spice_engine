//! x86_64 16-lane CPU pair kernel.
//!
//! This module is compiled only on x86_64, where the existing `lin_alg`
//! 16-lane types are available. The runtime dispatch must still check AVX-512
//! before calling this kernel.

#![cfg(target_arch = "x86_64")]

use lin_alg::f32::{Vec3x16, f32x16};

#[derive(Clone, Copy, Debug)]
pub struct PairBatch16 {
    pub dx: f32x16,
    pub dy: f32x16,
    pub dz: f32x16,
    pub sigma: f32x16,
    pub epsilon: f32x16,
    pub charge_product: f32x16,
}

#[derive(Clone, Copy, Debug)]
pub struct PairResult16 {
    pub force: Vec3x16,
    pub energy: f32x16,
}

impl PairBatch16 {
    #[inline]
    pub fn eval_lj_coulomb(self, coulomb_k: f32) -> PairResult16 {
        let dist_sq = self.dx * self.dx + self.dy * self.dy + self.dz * self.dz;
        let inv_dist = dist_sq.sqrt().recip();
        let sr = self.sigma * inv_dist;
        let sr2 = sr * sr;
        let sr6 = sr2 * sr2 * sr2;
        let sr12 = sr6 * sr6;
        let lj_mag =
            f32x16::splat(24.0) * self.epsilon * (f32x16::splat(2.0) * sr12 - sr6) * inv_dist;
        let lj_energy = f32x16::splat(4.0) * self.epsilon * (sr12 - sr6);
        let coul_mag = f32x16::splat(coulomb_k) * self.charge_product * inv_dist * inv_dist;
        let coul_energy = f32x16::splat(coulomb_k) * self.charge_product * inv_dist;
        let mag = lj_mag + coul_mag;
        PairResult16 {
            force: Vec3x16 {
                x: self.dx * inv_dist * mag,
                y: self.dy * inv_dist * mag,
                z: self.dz * inv_dist * mag,
            },
            energy: lj_energy + coul_energy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sixteen_lane_kernel_is_finite() {
        let b = PairBatch16 {
            dx: f32x16::splat(2.0),
            dy: f32x16::splat(0.0),
            dz: f32x16::splat(0.0),
            sigma: f32x16::splat(1.0),
            epsilon: f32x16::splat(0.2),
            charge_product: f32x16::splat(0.1),
        };
        let r = b.eval_lj_coulomb(332.0522);
        assert!(r.energy.to_array().iter().all(|v| v.is_finite()));
    }
}
