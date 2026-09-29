//! This module deals with the sim box, and barostat.
//!
//! We set up Sim box, or cell, which is a rectangular prism (cube currently) which wraps at each face,
//! indefinitely. Its purpose is to simulate an infinity of solvent molecules. This box covers the atoms of interest,
//! but atoms in the neighboring (tiled) boxes influence the system as well. We use the concept of
//! a "minimum image" to find the closest copy of an item to a given site, among all tiled boxes.
//!
//! Note: We keep most thermostat and barostat code as f64, although we use f32 in most sections.

use std::fmt::Display;

use bincode::{Decode, Encode};
use lin_alg::f32::Vec3;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

use crate::engine::md_core::{
    AtomDynamics, NATIVE_TO_KCAL, SimBoxInit, solvent::WaterMolOpc,
    thermostat::KB_A2_PS2_PER_K_PER_AMU,
};

pub(crate) const BAR_PER_KCAL_MOL_PER_ANSTROM_CUBED: f64 = 69476.95457055373;

/// Boltzmann constant in bar·Å³·K⁻¹ (= 1.380649×10⁻²³ J/K × 10⁻⁵ bar/Pa × 10³⁰ Å³/m³).
/// Used for the stochastic term in the C-rescale barostat.
const KB_BAR_A3_PER_K: f64 = 138.064_9;

pub const PRESSURE_DEFAULT: f32 = 1.; // Bar
pub const TAU_PRESSURE_DEFAULT: f32 = 5.; // ps

#[cfg_attr(feature = "encode", derive(Encode, Decode))]
#[derive(Debug, Clone, PartialEq)]
pub struct BarostatCfg {
    /// The barostat attempts to maintain this paressure. Bar (Pa/100).
    pub pressure_target: f32,
    /// The time constant for pressure coupling. ps.
    pub tau: f32,
    /// Also known as kappa_t. bar^‑1 (≈4.5×10⁻⁵ for solvent at 300K, 1bar)
    pub solvent_compressibility: f64,
}

impl Default for BarostatCfg {
    fn default() -> Self {
        // Same as GROMACS defaults.
        Self {
            pressure_target: PRESSURE_DEFAULT,
            tau: TAU_PRESSURE_DEFAULT,
            solvent_compressibility: 4.5e-5,
        }
    }
}

/// This bounds the area where atoms are wrapped. For now at least, it is only
/// used for solvent atoms. Its size and position should be such as to keep the system
/// solvated. We may move it around during the sim.
#[derive(Clone, Copy, Default, PartialEq, Debug, Encode, Decode)]
pub struct SimBox {
    pub bounds_low: Vec3,
    pub bounds_high: Vec3,
    /// Cached; bounds_high - bounds_low
    pub extent: Vec3,
}

impl SimBox {
    pub fn new(bounds_low: Vec3, bounds_high: Vec3) -> Self {
        Self {
            bounds_low,
            bounds_high,
            extent: bounds_high - bounds_low,
        }
    }

    /// Set up to surround all atoms, with a pad, or with fixed dimensions. `atoms` is whichever we
    /// use to center the bix.
    ///
    /// `atoms` is only used when initializing from a pad.
    pub fn from_solute_atoms(atom_posits: &[Vec3], box_type: &SimBoxInit) -> Self {
        match box_type {
            SimBoxInit::Pad(pad) => {
                let (mut min, mut max) =
                    (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));

                for posit in atom_posits {
                    min = min.min(*posit);
                    max = max.max(*posit);
                }

                if atom_posits.is_empty() {
                    min = Vec3::new_zero();
                    max = Vec3::new_zero();
                }

                let bounds_low = min - Vec3::splat(*pad);
                let bounds_high = max + Vec3::splat(*pad);

                Self {
                    bounds_low,
                    bounds_high,
                    extent: bounds_high - bounds_low,
                }
            }
            SimBoxInit::Fixed((bounds_low, bounds_high)) => {
                let bounds_low: Vec3 = *bounds_low;
                let bounds_high: Vec3 = *bounds_high;

                Self::new(bounds_low, bounds_high)
            }
        }
    }

    /// We periodically run this to keep the solvent surrounding the dynamic atoms, as they move.
    pub fn recenter(&mut self, atoms: &[AtomDynamics]) {
        if atoms.is_empty() {
            return;
        }

        let half_ext = self.extent / 2.;

        let mut center = Vec3::new_zero();
        let mut count = 0usize;

        for atom in atoms.iter().filter(|a| !a.static_) {
            center += atom.posit;
            count += 1;
        }

        if count == 0 {
            for atom in atoms {
                center += atom.posit;
            }
            count = atoms.len();
        }

        center /= count as f32;

        self.bounds_low = center - half_ext;
        self.bounds_high = center + half_ext;
    }

    /// Wrap an absolute coordinate back into the unit cell. (orthorhombic). We use it to
    /// keep arbitrary coordinates inside it.
    pub fn wrap(&self, p: Vec3) -> Vec3 {
        let ext = &self.extent;

        assert!(
            ext.x > 0.0 && ext.y > 0.0 && ext.z > 0.0,
            "SimBox edges must be > 0 (lo={:?}, hi={:?})",
            self.bounds_low,
            self.bounds_high
        );

        // rem_euclid keeps the value in [0, ext)
        Vec3::new(
            (p.x - self.bounds_low.x).rem_euclid(ext.x) + self.bounds_low.x,
            (p.y - self.bounds_low.y).rem_euclid(ext.y) + self.bounds_low.y,
            (p.z - self.bounds_low.z).rem_euclid(ext.z) + self.bounds_low.z,
        )
    }

    /// Minimum-image displacement vector. Find the closest copy
    /// of an item to a given site, among all tiled boxes. Maps a displacement vector to the closest
    /// periodic image. Allows distance measurements to use the shortest separation.
    pub fn min_image(&self, dv: Vec3) -> Vec3 {
        let ext = &self.extent;

        Vec3::new(
            dv.x - (dv.x / ext.x).round() * ext.x,
            dv.y - (dv.y / ext.y).round() * ext.y,
            dv.z - (dv.z / ext.z).round() * ext.z,
        )
    }

    pub fn volume(&self) -> f32 {
        (self.bounds_high.x - self.bounds_low.x).abs()
            * (self.bounds_high.y - self.bounds_low.y).abs()
            * (self.bounds_high.z - self.bounds_low.z).abs()
    }

    pub fn center(&self) -> Vec3 {
        (self.bounds_low + self.bounds_high) * 0.5
    }

    pub fn translated(&self, offset: Vec3) -> Self {
        Self::new(self.bounds_low + offset, self.bounds_high + offset)
    }

    /// For use with the barostat. It will expand or shrink the box if it determines the pressure
    /// is too high or low based on the virial pair sum.
    pub fn scale_isotropic(&mut self, lambda: f32) {
        // f32 throughout is deliberate: per-step |λ−1| is bounded at 10^-1 by
        // the caller and positions are already f32; a f64 pass would only
        // widen the cast boundary without changing reachable precision.

        // Treat non-finite or tiny λ as "no-op"
        let lam = if lambda.is_finite() && lambda.abs() > 1.0e-12 {
            lambda
        } else {
            1.0
        };

        let c = self.center();
        let lo = c + (self.bounds_low - c) * lam;
        let hi = c + (self.bounds_high - c) * lam;

        // Enforce low <= high per component
        self.bounds_low = Vec3::new(lo.x.min(hi.x), lo.y.min(hi.y), lo.z.min(hi.z));
        self.bounds_high = Vec3::new(lo.x.max(hi.x), lo.y.max(hi.y), lo.z.max(hi.z));
        self.extent = self.bounds_high - self.bounds_low;

        debug_assert!({
            let ext = &self.extent;
            ext.x > 0.0 && ext.y > 0.0 && ext.z > 0.0
        });
    }

    pub fn contains(&self, posit: Vec3) -> bool {
        !(posit.x < self.bounds_low.x
            || posit.y < self.bounds_low.y
            || posit.z < self.bounds_low.z
            || posit.x > self.bounds_high.x
            || posit.y > self.bounds_high.y
            || posit.z > self.bounds_high.z)
    }

    pub fn contains_region(&self, other: &Self) -> bool {
        !(other.bounds_low.x < self.bounds_low.x
            || other.bounds_low.y < self.bounds_low.y
            || other.bounds_low.z < self.bounds_low.z
            || other.bounds_high.x > self.bounds_high.x
            || other.bounds_high.y > self.bounds_high.y
            || other.bounds_high.z > self.bounds_high.z)
    }
}

/// The virial, in Kcal/Mol. Converted from our native units. We use a
/// separate type to help ensure we are using the correct units.
#[derive(Debug, Default)]
pub struct VirialKcalMol {
    pub bonded: f64,
    pub nonbonded_short_range: f64,
    pub nonbonded_long_range: f64,
    pub constraints: f64,
}

impl VirialKcalMol {
    pub(crate) fn total(&self) -> f64 {
        self.bonded + self.nonbonded_short_range + self.nonbonded_long_range + self.constraints
    }
}

impl Display for VirialKcalMol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Virial, kcal/mol. W_bonded: {:.3} kcal/mol  W Short range={:.3}  W long range: {:.3}  W_constraint: {:.3}",
            self.bonded, self.nonbonded_short_range, self.nonbonded_long_range, self.constraints,
        )
    }
}

/// Accumulated during force computations. Used to measure pressure.
/// Units per field:
///   - `bonded`, `nonbonded_short_range`, `nonbonded_long_range`: **kcal/mol**
///     (forces in kcal/(mol·Å) × distances in Å — no conversion needed).
///   - `constraints`: **native units** (amu·Å²/ps²), because SHAKE constraint
///     forces are computed as m·Δr/dt² which is in amu·Å/ps², and the virial
///     r·(m·Δr/dt²) is therefore in amu·Å²/ps².
/// We split this into components to make validating and debugging easier.
#[derive(Debug, Default, Clone)]
pub struct Virial {
    pub bonded: f64,
    pub nonbonded_short_range: f64,
    pub nonbonded_long_range: f64,
    pub constraints: f64,
}

impl Virial {
    /// Convert to kcal/mol.
    /// `bonded`/`nonbonded_*` are already in kcal/mol → copied as-is.
    /// `constraints` is in native units (amu·Å²/ps²) → multiplied by NATIVE_TO_KCAL.
    pub(crate) fn to_kcal_mol(&self) -> VirialKcalMol {
        VirialKcalMol {
            bonded: self.bonded,
            nonbonded_short_range: self.nonbonded_short_range,
            nonbonded_long_range: self.nonbonded_long_range,
            constraints: self.constraints * NATIVE_TO_KCAL as f64,
        }
    }
}

impl Display for Virial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Virial, native. W_bonded: {:.3} kcal/mol  W Short range={:.3}  W long range: {:.3}  W_constraint: {:.3}",
            self.bonded, self.nonbonded_short_range, self.nonbonded_long_range, self.constraints,
        )
    }
}

/// Barostat state.
/// Isotropic C-rescale (stochastic cell rescaling) barostat — GROMACS `pcoupl = C-rescale`.
///
/// Reference: Bernetti & Bussi, J. Chem. Phys. 153, 114107 (2020).
///
/// The volume update is:
///   ΔlnV = (κT/τp)(P_inst − P₀)dt  +  √(2κT·kB·T·dt / (τp·V)) · ξ,  ξ ~ N(0,1)
///   μ = exp(ΔlnV/3)   (isotropic length scale factor)
///
/// The deterministic part is identical to Berendsen; the stochastic term restores the
/// correct NpT fluctuations that Berendsen suppresses.
pub struct Barostat {
    pub virial: Virial,
    /// C-rescale uses a Gaussian; `StdRng` (rather than the thread-local RNG) so that
    /// `MdState` is `Send` and engine pools can run workers in parallel.
    pub rng: StdRng,
    /// Instantaneous pressure (bar) from the PREVIOUS force evaluation. The
    /// barostat teleport now runs between the drift and this step's force
    /// evaluation (GROMACS placement — see `step`), so its drive term is one
    /// step behind by construction; at dlnV ≲ 10⁻³/step the lag is inert.
    pub last_p_inst_bar: f64,
}

static SEED_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn generate_unique_seed() -> u64 {
    let nanos = crate::engine::md_core::clock::unix_ns();
    let count = SEED_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SplitMix64 avalanche mixer: ensures that consecutive counts are mapped to completely
    // uncorrelated seeds, even when nanos is identical across parallel threads/nodes.
    let mut x = nanos.wrapping_add(count).wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

impl Default for Barostat {
    fn default() -> Self {
        Self {
            virial: Default::default(),
            rng: StdRng::seed_from_u64(generate_unique_seed()),
            last_p_inst_bar: 0.0,
        }
    }
}

// StdRng (rand 0.10) is not `Clone`, so we hand-implement: the virial is copied and
// the clone gets a fresh time-seeded RNG, mirroring `Default`. Cloned engines are
// independent environment points, so a fresh RNG stream is exactly what we want.
impl Clone for Barostat {
    fn clone(&self) -> Self {
        Self {
            virial: self.virial.clone(),
            rng: StdRng::seed_from_u64(generate_unique_seed()),
            last_p_inst_bar: self.last_p_inst_bar,
        }
    }
}

impl Barostat {
    /// Compute the isotropic length scale factor μ using the C-rescale algorithm.
    ///
    /// `temp_k` should be the reference (target) temperature in K.
    /// `vol_a3` is the current simulation-box volume in Å³.
    pub fn scale_factor(
        &mut self,
        p_inst: f64,
        dt: f64,
        temp_k: f64,
        vol_a3: f64,
        cfg: &BarostatCfg,
    ) -> f64 {
        // Deterministic term: ΔlnV_det = (κT/τp)(P_inst − P₀)dt
        let dlnv_det = (cfg.solvent_compressibility / cfg.tau as f64)
            * (p_inst - cfg.pressure_target as f64)
            * dt;

        // Stochastic term: σ = √(2κT·kB·T·dt / (τp·V))
        let sigma_lnv = (2.0 * cfg.solvent_compressibility * KB_BAR_A3_PER_K * temp_k * dt
            / (cfg.tau as f64 * vol_a3))
            .sqrt();
        let xi: f64 = StandardNormal.sample(&mut self.rng);

        // Cap per-step volume change (≤10%) befsore computing λ
        const MAX_DLNV: f64 = 0.10;
        let dlnv = (dlnv_det + sigma_lnv * xi).clamp(-MAX_DLNV, MAX_DLNV);

        // λ = exp(ΔlnV/3) — isotropic length scale, strictly positive and well-behaved
        (dlnv / 3.0).exp()
    }

    pub(crate) fn apply_isotropic(
        &mut self,
        dt_ps: f64,
        p_inst_bar: f64,
        temp_k: f64,
        cfg: &BarostatCfg,
        simbox: &mut SimBox,
        atoms_dyn: &mut [AtomDynamics],
        waters: &mut [WaterMolOpc],
    ) -> bool {
        let vol_a3 = simbox.volume() as f64;
        let lam = self.scale_factor(p_inst_bar, dt_ps, temp_k, vol_a3, cfg); // λ for lengths (not volume)

        if !(lam.is_finite() && lam > 0.0) || (lam - 1.0).abs() < 1e-12 {
            return false; // no-op
        }

        // 1) Scale the box about its center
        simbox.scale_isotropic(lam as f32);

        // 2) Scale all coordinates about the same center (affine dilation)
        let c = simbox.center();
        let lc = lam as f32;

        fn scale_pos(p: &mut Vec3, c: Vec3, s: f32) {
            *p = c + (*p - c) * s;
        }

        for a in atoms_dyn.iter_mut() {
            if !a.static_ {
                scale_pos(&mut a.posit, c, lc);
            }
        }
        for w in waters.iter_mut() {
            scale_pos(&mut w.o.posit, c, lc);
            scale_pos(&mut w.h0.posit, c, lc);
            scale_pos(&mut w.h1.posit, c, lc);
        }

        // Stochastic cell-rescaling velocity transform (Bussi, Zykova-Timan &
        // Parrinello JCP 131, 014101 (2009); Bernetti & Bussi 2020 eq. 4/6):
        //   v → v/μ + sqrt(kB·T·(1−μ⁻²)/m)·ξ,  ξ ~ N(0,1),
        // with the noise only when μ > 1 (contraction has no real variance).
        // Peculiar velocities are DIVIDED by μ: for an ideal gas this division
        // IS the PdV work accounting (K ∝ μ⁻² exactly balances V ∝ μ³… the
        // gas cools as it expands); the noise term restores the Maxwell
        // distribution at the reference temperature. (Historical note —
        // 2026-09 forced-μ bisecting: the v × μ this replaced was wrong per
        // the paper, but correcting it, AND moving the teleport from the step
        // END to before force evaluation (see `MdState::step`), changed the
        // hot-start heating slope by only their own small shares. Neither was
        // the detonation cause: that is the teleport's real external work
        // P·ΔV̇ on a build box that is tens of kbar overpressured, which the
        // weak production thermostat (Langevin γ=0.5) cannot shed — so NPT
        // from a cold box self-heats regardless of these velocity/ordering
        // details. RL rollouts run NVT; these two are kept as ensemble/paper
        // correctness.)
        let inv_mu = 1.0 / lc;
        let var_factor = (1.0 - inv_mu * inv_mu).max(0.0) as f64;
        let kb_t = KB_A2_PS2_PER_K_PER_AMU as f64 * temp_k.max(0.0);
        let noise = var_factor > 0.0;

        for a in atoms_dyn.iter_mut() {
            if a.static_ {
                continue;
            }
            a.vel *= inv_mu;
            if noise && a.mass > 0.0 {
                let sigma = (kb_t * var_factor / a.mass as f64).sqrt() as f32;
                a.vel += Vec3::new(
                    sigma * self.sample_normal(),
                    sigma * self.sample_normal(),
                    sigma * self.sample_normal(),
                );
            }
        }
        for w in waters.iter_mut() {
            w.o.vel *= inv_mu;
            w.h0.vel *= inv_mu;
            w.h1.vel *= inv_mu;
            if noise {
                let so = (kb_t * var_factor / w.o.mass as f64).sqrt() as f32;
                let sh = (kb_t * var_factor / w.h0.mass as f64).sqrt() as f32;
                w.o.vel += Vec3::new(
                    so * self.sample_normal(),
                    so * self.sample_normal(),
                    so * self.sample_normal(),
                );
                w.h0.vel += Vec3::new(
                    sh * self.sample_normal(),
                    sh * self.sample_normal(),
                    sh * self.sample_normal(),
                );
                w.h1.vel += Vec3::new(
                    sh * self.sample_normal(),
                    sh * self.sample_normal(),
                    sh * self.sample_normal(),
                );
            }

            // Positions moved in step 2; rebuild EP (also re-interpolates M's velocity).
            w.update_virtual_site();
        }
        true
    }

    /// One N(0,1) draw from the barostat's stream. f64 then cast — f32
    /// precision on a ~0.01 Å/ps noise term is irrelevant.
    fn sample_normal(&mut self) -> f32 {
        let x: f64 = StandardNormal.sample(&mut self.rng);
        x as f32
    }
}

/// Measure instantaneous pressure, in bar. Inputs have been converted from native units
/// to kcal and kcal/mol.
/// P = (2K + W) / (3V), in kcal/mol/Å³. Every virial bucket must use the shared
/// pair convention W = Σ (r_i − r_j)·f_on_i (positive under compression),
/// which is what makes the sum valid; angle/dihedral use the equivalent
/// Σ r_i·F_i with a reference atom (Newton-III makes it origin-independent).
pub(crate) fn measure_pressure(
    kinetic_energy: f64, // kcal
    simbox: &SimBox,
    virial: &VirialKcalMol,
) -> f64 {
    let vol = simbox.volume() as f64; // Å³

    // This is in kcal/mol/Å³
    let result = (2.0 * kinetic_energy + virial.total()) / (3.0 * vol);

    // Convert from kcal/mol/Å³ to bar
    result * BAR_PER_KCAL_MOL_PER_ANSTROM_CUBED
}

#[cfg(test)]
mod tests {
    use super::*;
    use lin_alg::f32::Vec3;

    const R_KCAL: f64 = 0.001_987_204_1; // kcal mol⁻¹ K⁻¹

    fn box_100() -> SimBox {
        SimBox::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(100.0, 100.0, 100.0))
    }

    #[test]
    fn ideal_gas_pressure_matches_nkt_over_v() {
        let n = 1000.0_f64;
        let t = 300.0_f64;
        // Translational KE of an ideal monatomic gas: 3/2 NRT.
        let ke = 1.5 * n * R_KCAL * t;
        let p = measure_pressure(ke, &box_100(), &VirialKcalMol::default());
        let expected = n * R_KCAL * t / 1.0e6 * BAR_PER_KCAL_MOL_PER_ANSTROM_CUBED;
        assert!(
            (p - expected).abs() / expected < 1e-9,
            "got {p}, expected {expected}"
        );
    }

    #[test]
    fn positive_pair_virial_raises_pressure() {
        // The shared convention W = Σ (r_i − r_j)·f_on_i is positive when
        // compression dominates; pressure must respond with the same sign.
        let base = measure_pressure(500.0, &box_100(), &VirialKcalMol::default());
        let w = VirialKcalMol {
            nonbonded_short_range: 1000.0,
            ..Default::default()
        };
        let p = measure_pressure(500.0, &box_100(), &w);
        assert!(p > base);
    }

    #[test]
    fn scale_factor_direction_and_bounds() {
        let mut barostat = Barostat::default();
        let cfg = BarostatCfg::default();
        // Over-pressurized: deterministic drift expands the box...
        let lam_hot = barostat.scale_factor(60_000.0, 0.002, 310.0, 200_000.0, &cfg);
        assert!(lam_hot > 1.0 && lam_hot < 1.04, "lam_hot={lam_hot}");
        // ...while negative (tensile) pressure contracts it.
        let lam_cold = barostat.scale_factor(-60_000.0, 0.002, 310.0, 200_000.0, &cfg);
        assert!(lam_cold < 1.0 && lam_cold > 0.96, "lam_cold={lam_cold}");
        // λ is exp(±0.1/3)-bounded by the dlnV clamp even at absurd pressure.
        let lam_extreme = barostat.scale_factor(1.0e9, 100.0, 310.0, 100.0, &cfg);
        assert!(lam_extreme.is_finite() && lam_extreme <= (0.1_f64 / 3.0).exp() + 1e-12);
    }
}
