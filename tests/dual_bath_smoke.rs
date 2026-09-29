//! Dual-bath annealing guard (v1.3.8): `set_solute_temperature(Some(k))`
//! rides the solute's Langevin noise at its own setpoint while water keeps
//! `temp_target`. Single-bath failure this fixes (measured v8b): water owns
//! ~96% of the heat capacity, so a stage setpoint never reaches the solute
//! inside an anneal window (150 ns at 360 K left solute at 434 K).
//! γ=2.0 → solite τ≈0.5 ps, so 4 ns is > 7 relaxation times: the assertion
//! is a steady-state check, not a kinetics check.

use bio_files::MmCif;
use spice_engine::engine::md_core::params::FfParamSet;
use spice_engine::engine::md_core::{ComputationDevice, Integrator};
use spice_engine::{BuildOptions, build_system};
use std::path::Path;

const R_KCAL: f64 = 0.001_987_204; // kcal/mol/K per atom-DOF

fn temps(state: &spice_engine::engine::md_core::MdState) -> (f64, f64) {
    let (mut ke_s, mut n_s) = (0.0f64, 0.0f64);
    for a in &state.atoms {
        if !a.static_ {
            ke_s += f64::from(a.mass) * f64::from(a.vel.magnitude_squared());
            n_s += 1.0;
        }
    }
    let (mut ke_w, mut n_w) = (0.0f64, 0.0f64);
    for w in &state.water {
        // ½Σm v² per molecule equals the physical COM+rotation KE exactly
        // (SETTLE kills the 3 internal DOF): count 6 DOF per molecule, not
        // 9 — the naive 3-site estimator under-reads rigid water T by ⅓.
        for atom in [&w.o, &w.h0, &w.h1] {
            ke_w += f64::from(atom.mass) * f64::from(atom.vel.magnitude_squared());
        }
        n_w += 2.0;
    }
    // <½ m v²> = ½ kT per DOF; per atom (3 DOF): m<v²> = 3kT. Native units
    // m[amu]·v²[Å²/ps²] → kcal/mol via /418.4. (SETTLE projects water
    // internals; the 3-site estimator under-rotates slightly — same
    // estimator for both phases, bounds are generous.)
    let conv = 1.0 / (418.4 * 3.0 * R_KCAL);
    (ke_s / n_s * conv, ke_w / n_w * conv)
}

/// Run with: `cargo test --release --test dual_bath_smoke -- --ignored --nocapture`
#[test]
#[ignore = "expensive: 2LYZ build + 2000 Langevin steps"]
fn solute_bath_follows_its_own_setpoint() {
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("load Amber parameters");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
    let mut engine =
        build_system(&dev, &params, protein, &BuildOptions::default()).expect("build engine");

    engine.state.cfg.integrator = Integrator::LangevinMiddle { gamma: 0.5 };
    // 2LYZ's BuildOptions default SKIPS the water Langevin (init-phase
    // remnant); production python Env does not. Dual-bath needs the bath
    // thermostat actually on.
    engine.state.cfg.overrides.skip_water_thermostat = false;
    // NVT: the default NPT barostat rescales ALL velocities on teleport
    // (Bussi), which recouples the phases and masks the dual-bath signal.
    engine.state.cfg.barostat_cfg = None;
    engine.state.cfg.temp_target = 400.0; // water bath stays hot
    engine.set_solute_temperature(Some(250.0)); // solute target cold
    engine.state.initialize_velocities(400.0, true);
    for step in 0..25_000 {
        if step % 3000 == 2999 {
            let (a, b) = temps(&engine.state);
            println!("t={}ps solT={a:.1} watT={b:.1}", (step + 1) as f64 * 0.002,);
        }
        let r = engine.step(None);
        assert!(!r.crashed, "step crashed");
    }
    let (t_sol, t_wat) = temps(&engine.state);
    println!("solT={t_sol:.1}K watT={t_wat:.1}K (targets 250 / 400)");
    assert!(t_sol.is_finite() && t_wat.is_finite(), "non-finite T");
    assert!(
        t_sol < 320.0,
        "solute must ride its 250 K setpoint, got {t_sol:.0} K"
    );
    assert!(
        t_wat > 350.0,
        "water must hold near 400 K, got {t_wat:.0} K"
    );
}
