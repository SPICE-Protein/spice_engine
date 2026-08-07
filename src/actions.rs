//! Continuous actions → bias forces (SAC micro-loop).
//!
//! An action `a ∈ R^M` (default M=16) multiplies a pretrained low-rank basis
//! matrix `W ∈ R^{[L,3]×M}` (L = sequence length) to produce a 3-D force on
//! each residue's Cα. Each component is `tanh`-clamped to `±clamp` (default
//! ±0.5 kcal/(mol·Å)), then expanded into a full-atom force vector passed to
//! `engine.step(Some(external_force))`.
//!
//! `ActionMask` re-randomises which residues receive bias force every
//! `mutation_every` steps (a "mutation" cadence, so RL does not stay locked
//! onto the same local region forever).
//! `EnvDelta` handles environment offsets: ΔT hot-switches immediately, ΔpH
//! takes effect on mutation/rebuild.

use lin_alg::f32::Vec3;

use crate::engine::{SpiceEngine, StepResult};
use crate::env::sane;

/// Which residues currently receive bias force, re-randomized every `mutation_every` steps.
pub struct ActionMask {
    enabled: Vec<bool>,
    step: usize,
    pub mutation_every: usize,
    seed: u64,
}

impl ActionMask {
    pub fn new(n_res: usize, mutation_every: usize) -> Self {
        let mut m = Self {
            enabled: vec![false; n_res],
            step: 0,
            mutation_every: mutation_every.max(1),
            seed: 0x9E37_79B9_7F4A_7C15,
        };
        m.remask();
        m
    }

    /// Advance one integration step; re-randomize the mask every `mutation_every` steps.
    pub fn tick(&mut self) {
        self.step += 1;
        if self.step >= self.mutation_every {
            self.step = 0;
            self.remask();
        }
    }

    fn remask(&mut self) {
        let mut x = self.seed;
        for e in self.enabled.iter_mut() {
            // SplitMix64-ish LCG — deterministic, dependency-free.
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *e = (x >> 33) % 3 == 0; // ~1/3 of residues active
        }
        self.seed = x;
    }

    pub fn enabled(&self) -> &[bool] {
        &self.enabled
    }

    /// Fraction of enabled residues.
    pub fn fraction(&self) -> f32 {
        if self.enabled.is_empty() {
            return 0.0;
        }
        self.enabled.iter().filter(|&&e| e).count() as f32 / self.enabled.len() as f32
    }
}

/// Low-rank force-basis action: `a ∈ R^M` → per-residue Cα forces.
pub struct ForceAction {
    /// Action-space dimension M.
    pub m: usize,
    /// Basis matrix `W`, row-major `[L*3, M]`.
    pub w: Vec<f32>,
    /// Force clamp (kcal/(mol·Å)); each component is `±clamp * tanh(·)`.
    pub clamp: f32,
    pub mask: ActionMask,
}

impl ForceAction {
    /// Build with a deterministic pseudo-random, column-normalized basis `W`
    /// (so each basis vector is unit norm and `|a|≤1` gives bounded forces).
    /// Call `set_w` to load a pre-trained basis later.
    pub fn new(n_res: usize, m: usize, clamp: f32, mutation_every: usize) -> Self {
        let mut w = vec![0.0f32; n_res * 3 * m];
        let mut x: u64 = 0x1234_5678_9ABC_DEF0;
        for v in w.iter_mut() {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0; // ~U(-1,1)
        }
        // Normalize each basis column.
        for col in 0..m {
            let mut norm = 0.0f32;
            for row in 0..n_res * 3 {
                let v = w[row * m + col];
                norm += v * v;
            }
            norm = norm.sqrt();
            if norm > 1e-6 {
                for row in 0..n_res * 3 {
                    w[row * m + col] /= norm;
                }
            }
        }
        Self {
            m,
            w,
            clamp,
            mask: ActionMask::new(n_res, mutation_every),
        }
    }

    /// Load a pre-trained basis (e.g. from Python). Length must equal `L*3*M`.
    pub fn set_w(&mut self, w: Vec<f32>) {
        assert_eq!(w.len(), self.w.len(), "W length mismatch");
        self.w = w;
    }

    /// Map action coefficients `a` (`len == m`) to per-residue forces (`len == L`).
    /// Only residues enabled by the mask get non-zero forces.
    pub fn actions_to_forces(&self, a: &[f32]) -> Vec<Vec3> {
        debug_assert_eq!(a.len(), self.m, "action length != M");
        let n_res = self.mask.enabled.len();
        let mut f = vec![Vec3::new_zero(); n_res];
        for res in 0..n_res {
            if !self.mask.enabled[res] {
                continue;
            }
            let base = res * 3 * self.m;
            let mut fx = 0.0f32;
            let mut fy = 0.0f32;
            let mut fz = 0.0f32;
            for m_idx in 0..self.m {
                let a_m = a[m_idx];
                let off = base + m_idx;
                fx += self.w[off] * a_m;
                fy += self.w[off + self.m] * a_m;
                fz += self.w[off + 2 * self.m] * a_m;
            }
            f[res] = Vec3::new(
                self.clamp * fx.tanh(),
                self.clamp * fy.tanh(),
                self.clamp * fz.tanh(),
            );
        }
        f
    }

    /// One combined step: advance the mask, map the action to forces, expand to
    /// all-atom indices (force applied at each residue's Cα), and integrate.
    pub fn step(&mut self, engine: &mut SpiceEngine, a: &[f32]) -> StepResult {
        self.mask.tick();
        let f_ca = self.actions_to_forces(a);
        let mut f_full = vec![Vec3::new_zero(); engine.state.atoms.len()];
        for (res, f) in f_ca.iter().enumerate() {
            let Some(&ca) = engine.topology.ca_indices.get(res) else { continue };
            f_full[ca] = *f;
        }
        engine.step(Some(f_full))
    }
}

/// Environment offset `[ΔpH, ΔT]`. ΔT is a "hot-switch" that applies immediately;
/// ΔpH only takes effect on mutation/rebuild (protonation is discrete), so it is
/// carried here and applied by the caller when rebuilding the system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvDelta {
    pub d_t: f32,
    pub d_ph: f32,
}

impl Default for EnvDelta {
    fn default() -> Self {
        Self {
            d_t: 0.0,
            d_ph: 0.0,
        }
    }
}

impl EnvDelta {
    pub fn new(d_t: f32, d_ph: f32) -> Self {
        Self { d_t, d_ph }
    }

    /// Apply the temperature offset immediately (target-temperature hot switch),
    /// clamped into the biologically sensible range.
    pub fn apply_t(&self, engine: &mut SpiceEngine) {
        let t = (engine.env.temp_k + self.d_t).clamp(sane::TEMP_K_MIN, sane::TEMP_K_MAX);
        engine.set_temperature(t);
    }

    /// Effective pH (used at rebuild time to set protonation states), clamped.
    pub fn effective_ph(&self, base_ph: f32) -> f32 {
        (base_ph + self.d_ph).clamp(sane::PH_MIN, sane::PH_MAX)
    }
}
