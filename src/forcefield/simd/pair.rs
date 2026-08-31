//! Portable 8-lane CPU pair kernels.
//!
//! `wide` selects the best available implementation for the target: AVX/SSE
//! on x86_64 and NEON on aarch64. The kernel is deliberately independent of
//! `dynamics::MdState` so it can be tested against the scalar reference before
//! the production force loop is switched over.

use wide::f32x8;

const FORCE_CAP: f32 = 1.0e4;
const ENERGY_CAP: f32 = 1.0e6;

#[derive(Clone, Copy, Debug)]
pub struct PairBatch8 {
    pub dx: f32x8,
    pub dy: f32x8,
    pub dz: f32x8,
    pub sigma: f32x8,
    pub epsilon: f32x8,
    pub charge_product: f32x8,
}

#[derive(Clone, Copy, Debug)]
pub struct PairResult8 {
    pub fx: f32x8,
    pub fy: f32x8,
    pub fz: f32x8,
    pub energy: f32x8,
}

impl PairBatch8 {
    /// Evaluate Lennard-Jones plus unscreened Coulomb pair terms for eight
    /// independent pairs. `coulomb_k` is the already-scaled Coulomb constant.
    /// This low-level reference kernel is intended for ordinary pair backends;
    /// screened PME real-space terms remain in the dynamics adapter for now.
    #[inline]
    pub fn eval_lj_coulomb(self, coulomb_k: f32) -> PairResult8 {
        let dist_sq = self.dx * self.dx + self.dy * self.dy + self.dz * self.dz;
        let inv_dist = dist_sq.sqrt().recip();
        let sr = self.sigma * inv_dist;
        let sr2 = sr * sr;
        let sr6 = sr2 * sr2 * sr2;
        let sr12 = sr6 * sr6;
        let lj_mag =
            f32x8::splat(24.0) * self.epsilon * (f32x8::splat(2.0) * sr12 - sr6) * inv_dist;
        let lj_energy = f32x8::splat(4.0) * self.epsilon * (sr12 - sr6);

        let coul_mag = f32x8::splat(coulomb_k) * self.charge_product * inv_dist * inv_dist;
        let coul_energy = f32x8::splat(coulomb_k) * self.charge_product * inv_dist;
        let mag = lj_mag + coul_mag;

        // The caller applies the scalar safety policy after lane extraction;
        // keeping the vector kernel branch-free is important for SIMD codegen.
        PairResult8 {
            fx: self.dx * inv_dist * mag,
            fy: self.dy * inv_dist * mag,
            fz: self.dz * inv_dist * mag,
            energy: lj_energy + coul_energy,
        }
    }

    /// Scalar-equivalent lane result with the same safety clamps used by the
    /// production scalar LJ path.
    pub fn lane_with_safety(self, lane: usize, coulomb_k: f32) -> ([f32; 3], f32) {
        assert!(lane < 8);
        let result = self.eval_lj_coulomb(coulomb_k);
        let mut force = [
            result.fx.to_array()[lane],
            result.fy.to_array()[lane],
            result.fz.to_array()[lane],
        ];
        let magnitude = (force[0] * force[0] + force[1] * force[1] + force[2] * force[2]).sqrt();
        if magnitude > FORCE_CAP {
            let scale = FORCE_CAP / magnitude;
            for component in &mut force {
                *component *= scale;
            }
        }
        (
            force,
            result.energy.to_array()[lane].clamp(-ENERGY_CAP, ENERGY_CAP),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_lane(
        dx: f32,
        dy: f32,
        dz: f32,
        sigma: f32,
        epsilon: f32,
        q: f32,
        k: f32,
    ) -> ([f32; 3], f32) {
        let r2 = dx * dx + dy * dy + dz * dz;
        let inv = r2.sqrt().recip();
        let sr = sigma * inv;
        let sr2 = sr * sr;
        let sr6 = sr2 * sr2 * sr2;
        let sr12 = sr6 * sr6;
        let mag = 24.0 * epsilon * (2.0 * sr12 - sr6) * inv + k * q * inv * inv;
        let mut f = [dx * inv * mag, dy * inv * mag, dz * inv * mag];
        let fm = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
        if fm > FORCE_CAP {
            let scale = FORCE_CAP / fm;
            for c in &mut f {
                *c *= scale;
            }
        }
        (
            f,
            (4.0 * epsilon * (sr12 - sr6) + k * q * inv).clamp(-ENERGY_CAP, ENERGY_CAP),
        )
    }

    #[test]
    fn eight_lane_matches_scalar_reference() {
        let dx = [1.7, 2.0, 2.4, 3.1, 4.0, 5.5, 7.0, 9.0];
        let dy = [0.2, -0.3, 0.4, 0.1, -0.2, 0.5, -0.6, 0.7];
        let dz = [0.1, 0.4, -0.2, 0.3, 0.6, -0.1, 0.2, -0.4];
        let sigma = [1.1; 8];
        let epsilon = [0.2; 8];
        let q = [0.15; 8];
        let batch = PairBatch8 {
            dx: f32x8::new(dx),
            dy: f32x8::new(dy),
            dz: f32x8::new(dz),
            sigma: f32x8::new(sigma),
            epsilon: f32x8::new(epsilon),
            charge_product: f32x8::new(q),
        };
        for lane in 0..8 {
            let (got_f, got_e) = batch.lane_with_safety(lane, 332.0522);
            let (want_f, want_e) = scalar_lane(
                dx[lane],
                dy[lane],
                dz[lane],
                sigma[lane],
                epsilon[lane],
                q[lane],
                332.0522,
            );
            for (g, w) in got_f.into_iter().zip(want_f) {
                assert!((g - w).abs() < 2e-5, "force {g} vs {w}");
            }
            assert!((got_e - want_e).abs() < 2e-5, "energy {got_e} vs {want_e}");
        }
    }

    #[test]
    fn eight_lane_lj_coulomb_is_finite() {
        let batch = PairBatch8 {
            dx: f32x8::splat(2.0),
            dy: f32x8::splat(0.0),
            dz: f32x8::splat(0.0),
            sigma: f32x8::splat(1.0),
            epsilon: f32x8::splat(0.2),
            charge_product: f32x8::splat(0.1),
        };
        let result = batch.eval_lj_coulomb(332.0522);
        assert!(result.energy.to_array().iter().all(|v| v.is_finite()));
        assert!(result.fx.to_array().iter().all(|v| v.is_finite()));
    }
}
