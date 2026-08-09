//! Post-minimization equilibration: NVT settle with strong friction + gentle
//! temperature ramp.
//!
//! Releases the residual strain that L-BFGS minimization cannot see — the
//! step-0 acceleration clamps that appear even after tight minimization
//! (observed at tolerance 2.0, 0.5 AND 0.1 on 2LYZ: 13-19 atoms clamped at
//! step 0, max |force| ~10⁴ kcal/mol/Å). The hotspots are almost all ADDED
//! HYDROGENS placed by ideal per-residue geometry that end up 1.5-2.0 Å from a
//! non-bonded atom of another residue — a hard LJ-wall position that a static
//! minimizer cannot fix (the H's are pinned by their bonds) and that the
//! step-0 thermal kick pushes into the repulsive wall.
//!
//! Strategy (validated empirically):
//!   • run `ramp_steps` MD steps, linearly ramping temperature from
//!     `t_start_k` up to the engine target;
//!   • use STRONG Langevin friction (`friction_gamma`, default 10) so the
//!     H-clash strain is critically damped and the H's settle into
//!     low-energy positions instead of spiking;
//!   • finish with `hold_steps` of unrestrained NVT at the target temperature.
//!
//! Measured on 2LYZ: max |force| drops from ~10⁴ (step 0) to ~107 by step 100,
//! and post-equilibration production runs have ZERO accel clamps (vs 13-19 per
//! step before).
//!
//! WHY NOT POSITIONAL RESTRAINTS: a harmonic restraint on the heavy atoms
//! freezes the protein skeleton, and since the H's (the actual hotspots) are
//! bonded to that frozen skeleton they cannot relax — the restraint stores the
//! strain instead of releasing it, and the run blows up once the restraint is
//! released (observed: crash at mid-ramp with k>0; restraint-free is stable).
//! `k_restraint` is kept as an explicit opt-in for users who want classical
//! GROMACS-style position restraints (e.g. to keep a coarsely-placed
//! backbone/Cα from drifting), but it is OFF by default.

use dynamics::Integrator;
use lin_alg::f32::Vec3;
use na_seq::Element;

use crate::engine::SpiceEngine;

/// Equilibration settings (NVT settle + strong friction; optional position
/// restraints — see module docs for why they are OFF by default).
#[derive(Debug, Clone)]
pub struct EquilConfig {
    /// Number of MD steps for the temperature ramp (`t_start_k` -> target).
    pub ramp_steps: usize,
    /// Starting temperature of the ramp, K.
    pub t_start_k: f32,
    /// OPT-IN positional restraint constant, kcal/(mol·Å²), decaying linearly
    /// to 0 over the ramp. `0.0` (default) = restraint-free — validated as the
    /// robust choice (restraints freeze the skeleton and prevent the H's that
    /// actually carry the strain from relaxing).
    pub k_restraint: f32,
    /// Unrestrained NVT steps at the target temperature after the ramp.
    pub hold_steps: usize,
    /// Restrain hydrogens too (only relevant if `k_restraint > 0`).
    pub restrain_hydrogens: bool,
    /// Langevin friction (1/ps) used DURING the ramp; the production integrator
    /// is restored afterwards. Strong friction critically damps the H-clash
    /// strain so it settles instead of spiking. 10 = well-damped; the default
    /// production gamma is 0.5.
    pub friction_gamma: f32,
}

impl Default for EquilConfig {
    fn default() -> Self {
        Self {
            ramp_steps: 300,
            t_start_k: 100.0,
            // Restraint-free: validated robust (see module docs).
            k_restraint: 0.0,
            hold_steps: 100,
            restrain_hydrogens: false,
            friction_gamma: 10.0,
        }
    }
}

/// Run the equilibration ramp. On success the engine's history (`u_history`,
/// pseudo-label accumulator) is reset so downstream metrics start from a clean
/// equilibrated state. Returns an error if the system blows up mid-ramp.
pub fn equilibrate(engine: &mut SpiceEngine, cfg: &EquilConfig) -> Result<(), String> {
    let t_end = engine.env.temp_k;
    let total = cfg.ramp_steps + cfg.hold_steps;
    if total == 0 {
        return Ok(());
    }
    let last = cfg.ramp_steps.saturating_sub(1).max(1) as f32;

    println!(
        "[equil] NVT settle: T {:.0}K -> {t_end:.0}K over {} steps{}then {} hold steps, gamma {} (k_restraint {:.1})",
        cfg.t_start_k,
        cfg.ramp_steps,
        if cfg.k_restraint > 0.0 { "(restraints), " } else { ", " },
        cfg.hold_steps,
        cfg.friction_gamma,
        cfg.k_restraint
    );

    // 1) Boost friction for the ramp and start cold & stationary, so the first
    //    steps don't combine thermal velocities with residual strain.
    let prev_integrator = engine.state.cfg.integrator.clone();
    engine.state.cfg.integrator = Integrator::LangevinMiddle {
        gamma: cfg.friction_gamma,
    };
    engine.reset_velocities();

    // 2) Reference positions: heavy atoms only by default (see module docs).
    let n = engine.state.atoms.len();
    let restrain: Vec<bool> = (0..n)
        .map(|i| cfg.restrain_hydrogens || engine.state.atoms[i].element != Element::Hydrogen)
        .collect();
    let ref_pos: Vec<Vec3> = (0..n)
        .map(|i| {
            let p = engine.state.atoms[i].posit;
            Vec3::new(p.x, p.y, p.z)
        })
        .collect();

    let mut steps = 0usize;
    let mut run_step = |engine: &mut SpiceEngine, temp: f32, k: f32, step_idx: usize| -> Result<(), String> {
        let mut force = vec![Vec3::new_zero(); n];
        if k > 0.0 {
            for i in 0..n {
                if !restrain[i] {
                    continue;
                }
                let p = engine.state.atoms[i].posit;
                force[i] = (ref_pos[i] - p) * k; // F = -k·(p − ref), toward ref
            }
        }
        let r = engine.step(if k > 0.0 { Some(force) } else { None });
        if step_idx % 100 == 0 || r.crashed || step_idx == total - 1 {
            let mf = engine
                .state
                .atoms
                .iter()
                .map(|a| a.force.magnitude())
                .fold(0.0f32, f32::max);
            println!(
                "[equil] step {step_idx:>4}/{total} T={temp:6.1}K k={k:7.1} U={:+.3e} maxF={mf:9.1}",
                r.u_t_kcal
            );
        }
        if r.crashed {
            return Err(format!(
                "system crashed during equilibration at step {step_idx}/{total} (T={temp:.0}K, U={:.3e})",
                r.u_t_kcal
            ));
        }
        Ok(())
    };

    // 3) Ramp: temperature up, restraint down (both linear).
    for step in 0..cfg.ramp_steps {
        let frac = step as f32 / last;
        let temp = cfg.t_start_k + (t_end - cfg.t_start_k) * frac;
        let k = cfg.k_restraint * (1.0 - frac);
        engine.set_temperature(temp);
        run_step(engine, temp, k, steps)?;
        steps += 1;
    }

    // 4) Unrestrained NVT hold at the target temperature (settle, dump residual).
    engine.set_temperature(t_end);
    for _ in 0..cfg.hold_steps {
        run_step(engine, t_end, 0.0, steps)?;
        steps += 1;
    }

    // 5) Restore the production integrator and reset history so downstream
    //    metrics / pseudo-labels start from a clean equilibrated state.
    engine.state.cfg.integrator = prev_integrator;
    engine.reset_history();
    println!(
        "[equil] done: T={t_end:.0}K, U={:.3e}",
        engine.state.potential_energy
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_sane() {
        let c = EquilConfig::default();
        assert!(c.ramp_steps > 0);
        assert!(c.t_start_k < 310.0);
        assert!(c.k_restraint > 0.0);
        assert!(c.friction_gamma > 0.0);
    }
}
