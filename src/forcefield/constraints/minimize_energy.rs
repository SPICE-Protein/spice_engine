use std::time::Instant;

use lin_alg::f32::Vec3;

use crate::{AtomDynamics, ComputationDevice, MdState};

// emtol and emstep come from cfg.energy_minimization; see EnergyMinimization.
// nstcgsteep (CG) and nbfgscorr (L-BFGS) are not used: this is a steepest-descent minimizer.
const STEP_MAX: f32 = 0.2; // cap per-atom displacement per iteration (Å)
const GROW: f32 = 1.2; // expand step if energy decreased
const SHRINK: f32 = 0.5; // backtrack factor if energy increased
const ALPHA_MIN: f32 = 1.0e-8;
const ALPHA_MAX: f32 = 1.0e-2;

/// Force/E at current geometry
fn compute_forces_and_energy(
    state: &mut MdState,
    dev: &ComputationDevice,
    external_force: &Option<Vec<Vec3>>,
) {
    state.reset_f_acc_pe_virial();
    state.potential_energy = 0.0;

    state.apply_all_forces(dev, external_force);
}

fn force_stats(state: &MdState) -> (f32, f32) {
    let mut max_f_loc = 0.0f32;
    let mut sum = 0.0f32;

    let mut n = 0;
    for a in &state.atoms {
        if a.static_ {
            continue;
        }

        let m = a.force.magnitude();
        max_f_loc = max_f_loc.max(m);

        sum += m * m;
        n += 1;
    }

    let rms = if n > 0 { (sum / n as f32).sqrt() } else { 0.0 };

    (max_f_loc, rms)
}

impl MdState {
    /// Relaxes the molecules using a steepest-descent energy minimizer. Use this at the start of the simulation
    /// to control kinetic energy that
    /// arrises from differences between atom positions, and bonded parameters. It can also be called
    /// externally. It also stabilizes the solvent molecules, so that their hydrogen bond
    /// structure is correct at initialization.
    ///
    /// Uses flexible bonds to hydrogen. (Not Shake/Rattle constraints)
    ///
    /// We don't apply this to solvent molecules, as we have a pre-sim set up for them that runs
    /// prior to this.
    pub fn minimize_energy(
        &mut self,
        dev: &ComputationDevice,
        max_iters: usize,
        external_force: Option<Vec<Vec3>>,
    ) {
        let pb = indicatif::ProgressBar::new(max_iters as u64);
        pb.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message("Minimizing energy...");

        let start = Instant::now();

        let iters = self.minimize_lbfgs(dev, max_iters, &external_force, Some(&pb));

        let elapsed = start.elapsed().as_millis();
        pb.finish_with_message(format!("Complete in {elapsed} ms. Used {iters} of {max_iters} iters"));
    }

    /// L-BFGS energy minimization (ref: OpenMM `LocalEnergyMinimizer`, Nocedal & Wright).
    ///
    /// Limited-memory BFGS with a strong-Wolfe line search. Every evaluation
    /// rebuilds the neighbor list and computes the FULL forces at the current
    /// coordinates, so — unlike the steepest-descent loop — it cannot
    /// false-converge on a stale neighbor list (the root cause of "minimizer
    /// converges but production step 1 sees ~100× larger forces" blow-ups).
    ///
    /// Returns the number of outer iterations used.
    fn minimize_lbfgs(
        &mut self,
        dev: &ComputationDevice,
        max_iters: usize,
        external_force: &Option<Vec<Vec3>>,
        pb: Option<&indicatif::ProgressBar>,
    ) -> usize {
        // L-BFGS / line-search constants (same values as OpenMM).
        const NUM_VECTORS: usize = 6;
        const FTOL: f64 = 1e-4; // strong-Wolfe c1 (Armijo)
        const WOLFE: f64 = 0.9; // strong-Wolfe c2 (curvature)
        const STEP_SCALE_DOWN: f64 = 0.5;
        const STEP_SCALE_UP: f64 = 2.1;
        const MIN_STEP: f64 = 1e-20;
        const MAX_STEP: f64 = 1e20;
        const MAX_LS: usize = 40;

        let n = self.atoms.len();
        if n == 0 {
            return 0;
        }
        let nd = 3 * n;
        let tolerance = (self.cfg.energy_minimization_tolerance as f64) * (n as f64).sqrt();

        let dot = |a: &[f32], b: &[f32]| -> f64 { a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum() };
        let norm = |a: &[f32]| -> f64 { dot(a, a).sqrt() };

        let flat_pos = |atoms: &[AtomDynamics]| -> Vec<f32> {
            atoms.iter().flat_map(|a| [a.posit.x, a.posit.y, a.posit.z]).collect()
        };
        let flat_grad = |atoms: &[AtomDynamics]| -> Vec<f32> {
            atoms.iter().flat_map(|a| {
                if a.static_ {
                    [0.0, 0.0, 0.0]
                } else {
                    [-a.force.x, -a.force.y, -a.force.z]
                }
            }).collect()
        };
        let write_pos = |atoms: &mut [AtomDynamics], x: &[f32]| {
            for (i, a) in atoms.iter_mut().enumerate() {
                a.posit = Vec3::new(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
            }
        };

        // Evaluate full forces at the current coordinates (always fresh neighbors).
        self.build_all_neighbors(dev);
        compute_forces_and_energy(self, dev, external_force);

        let mut x = flat_pos(&self.atoms);
        let mut grad = flat_grad(&self.atoms);
        let mut energy = self.potential_energy;
        let mut used = 0;

        // 2026-08-17 Safeguard: Skip L-BFGS if non-finite coordinates, forces, or energy are detected to prevent SIMD/PME SIGSEGV
        if !x.iter().all(|val| val.is_finite()) || !grad.iter().all(|val| val.is_finite()) || !energy.is_finite() {
            eprintln!("[solvent_reuse] [warn] Non-finite positions, forces, or energy detected at start of L-BFGS minimization. Skipping local minimization to let simulation-level force clamps/healers handle it.");
            self.finish_minimize(dev);
            return 0;
        }

        if norm(&grad) <= tolerance {
            self.finish_minimize(dev);
            return used;
        }

        // L-BFGS history (s, y, rho), newest last.
        let mut s_hist: Vec<Vec<f32>> = Vec::with_capacity(NUM_VECTORS);
        let mut y_hist: Vec<Vec<f32>> = Vec::with_capacity(NUM_VECTORS);
        let mut rho_hist: Vec<f64> = Vec::with_capacity(NUM_VECTORS);

        for it in 0..max_iters {
            used = it + 1;

            // ---- two-loop recursion -> search direction ----
            let m = s_hist.len();
            let mut q = grad.clone();
            let mut alpha = vec![0.0f64; m];
            for i in (0..m).rev() {
                alpha[i] = rho_hist[i] * dot(&s_hist[i], &q);
                for k in 0..nd {
                    q[k] -= (alpha[i] as f32) * y_hist[i][k];
                }
            }
            let h0 = if m > 0 {
                let sy = dot(&s_hist[m - 1], &y_hist[m - 1]);
                let yy = dot(&y_hist[m - 1], &y_hist[m - 1]);
                if yy > 0.0 { sy / yy } else { 1.0 }
            } else {
                1.0
            };
            let mut r = vec![0.0f32; nd];
            for k in 0..nd {
                r[k] = (h0 as f32) * q[k];
            }
            for i in 0..m {
                let beta = rho_hist[i] * dot(&y_hist[i], &r);
                for k in 0..nd {
                    r[k] += ((alpha[i] - beta) as f32) * s_hist[i][k];
                }
            }
            let mut dir = vec![0.0f32; nd];
            for k in 0..nd {
                dir[k] = -r[k];
            }

            // ---- strong-Wolfe line search ----
            let x0 = x.clone();
            let grad0 = grad.clone();
            let e0 = energy;
            let gd0 = dot(&grad0, &dir);
            if gd0 >= 0.0 {
                break; // not a descent direction
            }

            let mut step = 1.0f64;
            let mut ls_ok = false;
            let mut x_new = x0.clone();
            let mut grad_new = grad0.clone();
            let mut e_new = e0;
            for _ in 0..MAX_LS {
                for k in 0..nd {
                    x_new[k] = x0[k] + (step as f32) * dir[k];
                }
                write_pos(&mut self.atoms, &x_new);
                self.build_all_neighbors(dev);
                compute_forces_and_energy(self, dev, external_force);
                e_new = self.potential_energy;
                for (i, a) in self.atoms.iter().enumerate() {
                    if a.static_ {
                        grad_new[3 * i] = 0.0;
                        grad_new[3 * i + 1] = 0.0;
                        grad_new[3 * i + 2] = 0.0;
                    } else {
                        grad_new[3 * i] = -a.force.x;
                        grad_new[3 * i + 1] = -a.force.y;
                        grad_new[3 * i + 2] = -a.force.z;
                    }
                }
                // Armijo condition.
                if e_new <= e0 + FTOL * step * gd0 {
                    let gd = dot(&grad_new, &dir);
                    if gd.abs() <= WOLFE * gd0.abs() {
                        ls_ok = true;
                        break;
                    }
                    step *= STEP_SCALE_UP;
                } else {
                    step *= STEP_SCALE_DOWN;
                }
                if step < MIN_STEP || step > MAX_STEP {
                    break;
                }
            }

            if !ls_ok {
                // Restore the last accepted point and stop.
                write_pos(&mut self.atoms, &x);
                break;
            }

            // ---- update L-BFGS history ----
            let s_new: Vec<f32> = (0..nd).map(|k| x_new[k] - x0[k]).collect();
            let y_new: Vec<f32> = (0..nd).map(|k| grad_new[k] - grad0[k]).collect();
            let sy = dot(&s_new, &y_new);
            if sy > 1e-10 {
                if s_hist.len() == NUM_VECTORS {
                    s_hist.remove(0);
                    y_hist.remove(0);
                    rho_hist.remove(0);
                }
                s_hist.push(s_new);
                y_hist.push(y_new);
                rho_hist.push(1.0 / sy);
            }

            x = x_new;
            grad = grad_new;
            energy = e_new;

            if let Some(pbar) = pb {
                pbar.set_position((it + 1) as u64);
                pbar.set_message(format!("E: {:.1} kcal/mol, |g|: {:.1}", energy, norm(&grad)));
            }

            if norm(&grad) <= tolerance {
                break;
            }
        }

        write_pos(&mut self.atoms, &x);
        self.finish_minimize(dev);
        used
    }

    /// Common post-minimization cleanup: zero forces, regenerate the PME grid,
    /// and drop the stale SPME cache so the next (production) step recomputes
    /// forces for the current coordinates.
    fn finish_minimize(&mut self, dev: &ComputationDevice) {
        self.reset_f_acc_pe_virial();
        self.regen_pme(dev);
        self.spme_force_prev = None;
    }

    /// Separate, so can be called `separately by an application, e.g. if it needs to
    /// apply a new external force each step.
    pub fn minimize_energy_setup(
        &mut self,
        dev: &ComputationDevice,
        external_force: &Option<Vec<Vec3>>,
    ) -> (Vec<Vec3>, f32, f64, Vec<Vec3>, bool) {
        // Minimize in the FULL force field (long-range reciprocal INCLUDED). Minimizing with
        // the reciprocal disabled converges to a state that is not a minimum of the actual
        // production potential: step 1 of production then sees much larger forces (e.g. max
        // force ~130 vs ~19 kcal/mol/Å at the end of a recip-off minimization) and the system
        // drifts into instability within a few steps.
        let prev_recip = self.cfg.overrides.long_range_recip_disabled;

        // Zero velocities; we’re minimizing, not integrating. Note that accel and force are
        // zeroed downstream.
        // Store initial velocities, and re-apply at the end.
        let mut initial_velocities = Vec::with_capacity(self.atoms.len());
        for a in &mut self.atoms {
            initial_velocities.push(a.vel);
            a.vel = Vec3::new_zero();
        }

        compute_forces_and_energy(self, dev, external_force);

        let alpha = 0.01;
        let e_prev = self.potential_energy;

        // Per-atom last step for backtracking
        let n_atoms = self.atoms.len();
        let last_step: Vec<_> = vec![Vec3::new_zero(); n_atoms];

        // // Helper to measure convergence
        // let (max_f, _rms_f) = force_stats(self);
        // if max_f <= F_TOL {
        //     // Undo our config change.
        //     self.cfg.overrides.long_range_recip_disabled = prev_long_range;
        //     return;
        // }

        (last_step, alpha, e_prev, initial_velocities, prev_recip)
    }

    /// See the note on `minimize_energy_setup`; this is broken out so it can be called separately
    /// by an application.
    pub fn minimize_energy_cleanup(
        &mut self,
        dev: &ComputationDevice,
        prev_long_range: bool,
        initial_velocities: &[Vec3],
    ) {
        // Cleanup: zero velocities, recenter, and refresh PME grid if you do this routinely elsewhere.
        for a in &mut self.atoms {
            a.vel = Vec3::new_zero();
        }

        self.reset_f_acc_pe_virial();

        // Keep consistent with the normal cadence.
        if self.cfg.recenter_sim_box {
            self.cell.recenter(&self.atoms);
        }

        // Undo our config change.
        self.cfg.overrides.long_range_recip_disabled = prev_long_range;
        self.regen_pme(dev);

        // The cached SPME forces were computed for the pre-cleanup coordinates;
        // recenter (above) shifts the system relative to the box, so reusing the
        // cache on the first production step would apply stale reciprocal forces
        // (observed: step-1 maxF ~200 vs the minimized ~2). Drop it and force a
        // fresh SPME computation on the next step.
        self.spme_force_prev = None;

        // Re-apply our initial velocities.
        for (i, a) in self.atoms.iter_mut().enumerate() {
            a.vel = initial_velocities[i];
        }
    }

    /// One iteration of energy minimization. Returns `true` if the energy is converged, indicating
    /// to abort further steps.
    pub fn step_energy_min(
        &mut self,
        dev: &ComputationDevice,
        last_step: &mut [Vec3],
        alpha: &mut f32,
        e_prev: &mut f64,
        external_force: &Option<Vec<Vec3>>,
    ) -> bool {
        let mut alpha_try = *alpha;

        loop {
            // Normalize by the global max force (GROMACS `steep` convention): the highest-force
            // atom moves exactly `step_size`, all others proportionally less. This prevents
            // low-force atoms from over-shooting when a high-force atom is capped at STEP_MAX.
            let f_max = self
                .atoms
                .iter()
                .filter(|a| !a.static_)
                .map(|a| a.force.magnitude())
                .filter(|m| m.is_finite())
                .fold(0.0_f32, f32::max);

            if f_max == 0.0 {
                return true;
            }

            let step_size = (alpha_try * f_max).min(STEP_MAX);

            for (i, a) in self.atoms.iter_mut().enumerate() {
                last_step[i] = Vec3::new_zero();
                if a.static_ {
                    continue;
                }

                let f_mag = a.force.magnitude();
                if !f_mag.is_finite() || f_mag == 0.0 {
                    continue;
                }

                let s = a.force * (step_size / f_max);
                a.posit += s;
                last_step[i] = s;
            }

            // Track cumulative displacement since the last neighbor rebuild (not just this
            // iteration's step size), so `build_neighbors_if_needed` keeps the pair list valid
            // as atoms move. Previously this reset to 0.0 each iteration and tracked only the
            // single-step displacement, so the neighbor list was built once at min start and
            // never refreshed — missing close pairs that emerged during relaxation. That made
            // the minimizer "converge" on a stale force field while production (which does
            // track cumulative displacement) suddenly saw much larger forces on step 1.
            // Always rebuild neighbors during minimization. The old thresholded
            // `build_neighbors_if_needed` let the pair list go stale: hydrogens
            // pinned by their bonds move little, so the cumulative displacement
            // never crossed the skin threshold and close pairs that emerged
            // during relaxation were missing — the minimizer then "converged" on
            // forces that omitted exactly the clashes production MD sees (maxF
            // ~2 at min end vs ~100-200 on production step 1).
            self.update_max_displacement_since_rebuild();

            self.build_all_neighbors(dev);

            compute_forces_and_energy(self, dev, external_force);
            let e_new = self.potential_energy;

            if self.cfg.overrides.snapshots_during_energy_min {
                self.handle_snapshots(0.); // Pressure: Not required here.e
            }

            if e_new <= *e_prev {
                *e_prev = e_new;

                *alpha = (alpha_try * GROW).min(ALPHA_MAX);

                let (max_f, _rms_f) = force_stats(self);
                return max_f <= self.cfg.energy_minimization_tolerance;
            }

            // Reject: revert positions
            for (i, a) in self.atoms.iter_mut().enumerate() {
                let s = last_step[i];
                if s.magnitude_squared() > 0.0 {
                    a.posit -= s;
                }
            }

            alpha_try *= SHRINK;

            if alpha_try < ALPHA_MIN {
                *alpha = alpha_try;
                self.update_max_displacement_since_rebuild();
                self.build_all_neighbors(dev);
                compute_forces_and_energy(self, dev, external_force);
                return true;
            }

            // Ensure neighbors/forces are valid at the reverted geometry, then retry with smaller alpha_try
            self.update_max_displacement_since_rebuild();
            self.build_all_neighbors(dev);
            compute_forces_and_energy(self, dev, external_force);
        }
    }

    /// Run a localized L-BFGS minimization on atoms within `radius` of `mutated_positions`.
    /// This is extremely fast (typically <5 ms) and avoids full-system minimization overhead
    /// while resolving any steric clashes from mutated sidechain packing or solvent/ion placement.
    pub fn minimize_local_region(
        &mut self,
        dev: &ComputationDevice,
        mutated_positions: &[Vec3],
        radius: f32,
        max_iters: usize,
    ) {
        if mutated_positions.is_empty() || max_iters == 0 {
            return;
        }

        let radius_sq = radius * radius;

        // 1. Backup the static_ flags of all atoms
        let original_static: Vec<bool> = self.atoms.iter().map(|a| a.static_).collect();

        // 2. Freeze all atoms (set static_ = true) EXCEPT those near mutated_positions
        for a in &mut self.atoms {
            let mut close = false;
            for &p in mutated_positions {
                let diff = self.cell.min_image(a.posit - p);
                if diff.magnitude_squared() < radius_sq {
                    close = true;
                    break;
                }
            }
            if !close {
                a.static_ = true;
            }
        }

        // 3. Run L-BFGS minimization on the active local region
        let _iters = self.minimize_lbfgs(dev, max_iters, &None, None);

        // 4. Restore original static_ flags
        for (a, orig) in self.atoms.iter_mut().zip(original_static) {
            a.static_ = orig;
        }
    }
}
