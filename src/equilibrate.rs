//! Post-minimization equilibration: positional restraints + NVT temperature
//! ramp.
//!
//! Releases the residual strain left by steepest-descent minimization (the
//! ~70–90 kcal/mol/Å hotspots that otherwise make production MD randomly blow
//! up within tens of steps).
//!
//! Strategy (standard practice):
//!   1. after minimization, record reference positions of the protein heavy atoms;
//!   2. run `ramp_steps` MD steps, linearly ramping temperature from `t_start_k`
//!      up to the engine's target temperature, while a harmonic positional
//!      restraint `k` on the heavy atoms decays linearly from `k_restraint` to 0.
//!
//! The harmonic force is injected through the engine's external-force hook
//! (`engine.step(Some(force))`), so it never enters the potential energy and no
//! dynamics-fork change is needed.

use lin_alg::f32::Vec3;

use crate::engine::SpiceEngine;

/// Equilibration settings (positional restraints + NVT ramp).
#[derive(Debug, Clone)]
pub struct EquilConfig {
    /// Number of MD steps for the combined ramp (temperature up, restraint down).
    pub ramp_steps: usize,
    /// Starting temperature of the ramp, K.
    pub t_start_k: f32,
    /// Initial restraint force constant, kcal/(mol·Å²). Decays linearly to 0.
    pub k_restraint: f32,
}

impl Default for EquilConfig {
    fn default() -> Self {
        Self {
            ramp_steps: 500,
            t_start_k: 100.0,
            // Strong enough to freeze the structure against the residual build
            // strain (~70–90 kcal/mol/Å) — a weak restraint cannot hold atoms in
            // place and the strain erupts as overlapping atoms / U blow-up.
            k_restraint: 100.0,
        }
    }
}

/// Run the equilibration ramp. On success the engine's history (`u_history`,
/// pseudo-label accumulator) is reset so downstream metrics start from a clean
/// equilibrated state. Returns an error if the system blows up mid-ramp.
pub fn equilibrate(engine: &mut SpiceEngine, cfg: &EquilConfig) -> Result<(), String> {
    let t_end = engine.env.temp_k;
    if cfg.ramp_steps == 0 {
        return Ok(());
    }
    let last = (cfg.ramp_steps.saturating_sub(1)).max(1) as f32;

    println!(
        "[equil] start: T {:.0}K -> {t_end:.0}K over {} steps, k_restraint {:.1} -> 0",
        cfg.t_start_k, cfg.ramp_steps, cfg.k_restraint
    );

    // Start from a cold, stationary state: minimization restores the incoming
    // (hot) velocities, which would fight the low-temperature ramp + restraint
    // and can destabilise the first steps.
    engine.reset_velocities();

    // Reference positions: ALL protein atoms (including H). Restraining only the
    // heavy atoms leaves the light hydrogens free to absorb the residual strain
    // and fly off — restraining the whole protein lets it settle as a unit.
    let n_atoms = engine.state.atoms.len();
    let ref_pos: Vec<Vec3> = (0..n_atoms)
        .map(|i| {
            let p = engine.state.atoms[i].posit;
            Vec3::new(p.x as f32, p.y as f32, p.z as f32)
        })
        .collect();

    for step in 0..cfg.ramp_steps {
        let frac = step as f32 / last;
        let temp = cfg.t_start_k + (t_end - cfg.t_start_k) * frac;
        let k = cfg.k_restraint * (1.0 - frac);

        engine.set_temperature(temp);

        let mut force = vec![Vec3::new_zero(); n_atoms];
        if k > 0.0 {
            for i in 0..n_atoms {
                let p = engine.state.atoms[i].posit;
                let d = Vec3::new(p.x as f32, p.y as f32, p.z as f32) - ref_pos[i];
                force[i] = -d * k;
            }
        }

        let r = engine.step(Some(force));
        if step % 50 == 0 || step == cfg.ramp_steps - 1 || r.crashed {
            let mf = engine
                .state
                .atoms
                .iter()
                .map(|a| a.force.magnitude())
                .fold(0.0f32, f32::max);
            println!(
                "[equil] step {step:>4}/{} T={temp:6.1}K k={k:6.2} U={:+.3e} maxF={mf:9.1}",
                cfg.ramp_steps, r.u_t_kcal
            );
        }
        if r.crashed {
            return Err(format!(
                "system crashed during equilibration ramp at step {}/{} (T={temp:.0}K, U={:.3e})",
                step + 1,
                cfg.ramp_steps,
                r.u_t_kcal
            ));
        }
    }

    engine.set_temperature(t_end);
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
    }
}
