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
    /// Compatibility entry point for AVX-512 callers that already performed
    /// dispatch. Panics rather than executing ZMM instructions on an
    /// unsupported host.
    #[inline]
    pub fn eval_lj_coulomb(self, coulomb_k: f32) -> PairResult16 {
        assert!(
            is_x86_feature_detected!("avx512f"),
            "AVX-512F is required for PairBatch16"
        );
        // SAFETY: the assertion above guarantees the target feature.
        unsafe { self.eval_lj_coulomb_avx512(coulomb_k) }
    }

    /// Evaluate the AVX-512 x16 kernel when the host supports it.
    ///
    /// `f32x16` is backed by `__m512`, so this method deliberately performs
    /// the runtime check at the API boundary. Callers on machines without
    /// AVX-512 receive `None` and must use the x8/scalar path instead.
    #[inline]
    pub fn eval_lj_coulomb_runtime(self, coulomb_k: f32) -> Option<PairResult16> {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: the feature check above dominates the only call to the
            // target-feature-specialized implementation.
            Some(unsafe { self.eval_lj_coulomb_avx512(coulomb_k) })
        } else {
            None
        }
    }

    /// AVX-512 implementation of the 16 independent LJ+Coulomb pairs.
    ///
    /// This is kept separate from runtime dispatch so LLVM can emit native
    /// ZMM arithmetic, while non-AVX-512 hosts never execute the function.
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub unsafe fn eval_lj_coulomb_avx512(self, coulomb_k: f32) -> PairResult16 {
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
    fn runtime_dispatch_is_safe_without_avx512() {
        let b = PairBatch16 {
            dx: f32x16::splat(2.0),
            dy: f32x16::splat(0.0),
            dz: f32x16::splat(0.0),
            sigma: f32x16::splat(1.0),
            epsilon: f32x16::splat(0.2),
            charge_product: f32x16::splat(0.1),
        };
        if !is_x86_feature_detected!("avx512f") {
            assert!(b.eval_lj_coulomb_runtime(332.0522).is_none());
        }
    }

    #[test]
    fn sixteen_lane_kernel_is_finite() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let b = PairBatch16 {
            dx: f32x16::splat(2.0),
            dy: f32x16::splat(0.0),
            dz: f32x16::splat(0.0),
            sigma: f32x16::splat(1.0),
            epsilon: f32x16::splat(0.2),
            charge_product: f32x16::splat(0.1),
        };
        let Some(r) = b.eval_lj_coulomb_runtime(332.0522) else {
            return;
        };
        assert!(r.energy.to_array().iter().all(|v| v.is_finite()));
        assert!(r.force.x.to_array().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sixteen_lane_kernel_matches_scalar_formula() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let dx = [
            1.7, 2.0, 2.4, 3.1, 4.0, 5.5, 7.0, 9.0, 1.8, 2.1, 2.5, 3.2, 4.1, 5.6, 7.1, 9.1,
        ];
        let dy = [
            0.2, -0.3, 0.4, 0.1, -0.2, 0.5, -0.6, 0.7, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8,
        ];
        let dz = [
            0.1, 0.4, -0.2, 0.3, 0.6, -0.1, 0.2, -0.4, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9,
        ];
        let b = PairBatch16 {
            dx: f32x16::from_array(dx),
            dy: f32x16::from_array(dy),
            dz: f32x16::from_array(dz),
            sigma: f32x16::splat(1.1),
            epsilon: f32x16::splat(0.2),
            charge_product: f32x16::splat(0.15),
        };
        let r = b.eval_lj_coulomb_runtime(332.0522).unwrap();
        for i in 0..16 {
            let r2 = dx[i] * dx[i] + dy[i] * dy[i] + dz[i] * dz[i];
            let inv = r2.sqrt().recip();
            let sr = 1.1 * inv;
            let sr6 = (sr * sr) * (sr * sr) * (sr * sr);
            let sr12 = sr6 * sr6;
            let mag = 24.0 * 0.2 * (2.0 * sr12 - sr6) * inv + 332.0522 * 0.15 * inv * inv;
            let want_e = 4.0 * 0.2 * (sr12 - sr6) + 332.0522 * 0.15 * inv;
            assert!((r.energy.to_array()[i] - want_e).abs() < 2e-4);
            assert!((r.force.x.to_array()[i] - dx[i] * inv * mag).abs() < 2e-4);
        }
    }
}
