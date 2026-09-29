//! NPT + pair-virial-sign integration checks on 2LYZ.
//!
//! `BuildOptions::default()` carries `EnvParams::default()` with
//! `pressure_bar = 1.0`, so every default engine is NPT once
//! `apply_isotropic` runs (it was previously short-circuited with
//! `return false`). These tests are the re-enablement guard:
//! - The bond/constraint pair-virial sign bug (bonded recorded
//!   +(r_1−r_0)·f_on_0, opposite the nonbonded convention) made the
//!   instantaneous pressure hugely positive; the box would then collapse
//!   at the per-step clamp. The volume-drift bound catches that class.
//! - A relaxed harmonic network has a *negative* bond pair-virial (bonds
//!   pull atoms together against collisions); the direct sign assertion
//!   is the cheap, high-margin regression (pre-fix measured ~ +30k kcal/mol).

use bio_files::MmCif;
use spice_engine::engine::md_core::ComputationDevice;
use spice_engine::engine::md_core::params::FfParamSet;
use spice_engine::{BuildOptions, EquilConfig, build_system};
use std::path::Path;

/// Run with: `cargo test --release --test npt_virial_smoke -- --ignored --nocapture`
#[test]
#[ignore = "expensive: full 2LYZ build plus 50 NPT steps"]
fn npt_bonded_virial_sign_and_box_stability() {
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("load Amber parameters");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
    let mut engine =
        build_system(&dev, &params, protein, &BuildOptions::default()).expect("build engine");

    for _ in 0..10 {
        let result = engine.step(None);
        assert!(!result.crashed, "warm-up crashed");
    }
    engine.state.initialize_velocities(310.0, true);

    fn volume(state_extent: lin_alg::f32::Vec3) -> f64 {
        f64::from(state_extent.x) * f64::from(state_extent.y) * f64::from(state_extent.z)
    }

    let v0 = volume(engine.state.cell.extent);
    let mut bonded_signs = vec![];
    let mut max_frac_dvol = 0.0_f64;
    let mut prev_v = v0;
    for step in 0..40 {
        let result = engine.step(None);
        assert!(!result.crashed, "step {step} crashed");
        assert!(result.u_t_kcal.is_finite(), "step {step} energy NaN");
        let (bonded, short, long, constraints) = engine.state.virial_components();
        for w in [bonded, short, long, constraints] {
            assert!(w.is_finite(), "step {step} virial component non-finite");
        }
        bonded_signs.push(bonded);
        let v = volume(engine.state.cell.extent);
        max_frac_dvol = max_frac_dvol.max((v - prev_v).abs() / prev_v);
        prev_v = v;
    }

    // Median over the window: bond stretching + constraints must net to a
    // negative virial for a relaxed network.
    bonded_signs.sort_by(|a, b| a.total_cmp(b));
    let median = bonded_signs[bonded_signs.len() / 2];
    assert!(
        median < 0.0,
        "bonded virial median {median} >= 0: pair-virial convention regressed"
    );

    // A correct 1-bar ensemble with tau = 1 ps and a ~10^2–10^3 bar
    // instantaneous error moves lnV by ≲ 10^-3 per step. Inverted-sign or
    // otherwise broken pressure (~10^4 bar) pins every step at the 10%
    // clamp, which these bounds catch with a wide margin.
    assert!(
        max_frac_dvol < 0.03,
        "volume jumped {max_frac_dvol} in one step; pressure/barostat feedback is unstable"
    );
    let drift = ((prev_v - v0) / v0).abs();
    assert!(
        drift < 0.05,
        "NPT drifted {drift:.3} of box volume in 40 steps"
    );
}

/// Long-window stability guard on the RECOMMENDED RL flow: build + NVT
/// equilibration ramp, then production at NVT (barostat off) with a settling
/// discard. This is the path that must never detonate. In v1.3.2 the
/// equivalent barostat-ON rollout injected up to +0.7 K/step (the position
/// teleport doing pressurization work on a tens-of-kbar overpressured build
/// box, undissipatable by the weak production thermostat) — a detonation by
/// 620–930 steps. `barostat_cfg = None` removes that pump entirely.
///
/// The one real transient left is a HANDOVER overshoot: the ramp holds T with
/// STRONG friction (γ=10), production uses WEAK γ=0.5, so residual dense-box
/// strain still converting to heat bleeds off over ~700 steps (T peaks ~410,
/// then settles to the Langevin ~320 band). RL must discard that settling
/// window — here 700 warm-up steps — exactly as one discards equilibration in
/// any MD protocol. The measured band after settling is what's asserted.
/// Run with: `cargo test --release --test npt_virial_smoke -- --ignored --nocapture`
#[test]
#[ignore = "expensive: full build + equilibrate + 700 settle + 1000 NVT steps"]
fn nvt_no_detonation_1000_steps() {
    const SETTLE: usize = 700;
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("load Amber parameters");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
    let opts = BuildOptions {
        equil: Some(EquilConfig::default()),
        ..Default::default()
    };
    let mut engine = build_system(&dev, &params, protein, &opts).expect("build engine");
    engine.state.cfg.barostat_cfg = None; // NVT rollout (the recommended path)
    let v0 = f64::from(engine.state.cell.extent.x)
        * f64::from(engine.state.cell.extent.y)
        * f64::from(engine.state.cell.extent.z);

    // Settling discard: no detonation allowed even here.
    for k in 0..SETTLE {
        let r = engine.step(None);
        let t = engine.state.last_temperature_k;
        assert!(
            !r.crashed && t.is_finite() && t < 1000.0,
            "NVT detonated during settling at step {k} (T={t:.0})"
        );
    }
    let mut temps = Vec::with_capacity(20);
    for k in 0..1000 {
        let r = engine.step(None);
        let t = engine.state.last_temperature_k;
        if r.crashed || !t.is_finite() || t > 1000.0 {
            panic!("NVT left the sane band at measure-step {k} (T={t:.0}) — runaway");
        }
        if (k + 1) % 50 == 0 {
            temps.push(t);
        }
    }
    assert!(
        temps.iter().all(|&t| t > 200.0 && t < 450.0),
        "settled T outside [200, 450] K somewhere in {temps:?}"
    );
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    let (early, late) = (mean(&temps[2..8]), mean(&temps[14..]));
    println!("[nvt] settled early-T {early:.1} K, late-T {late:.1} K, samples {temps:?}");
    assert!(
        (late - early).abs() < 50.0,
        "settled NVT temperature drifted {early:.0} → {late:.0} K — energy injection is back"
    );
    // NVT holds the build volume fixed; a live barostat re-scaling the box
    // (the old fault's signature) would show up here.
    let v1 = f64::from(engine.state.cell.extent.x)
        * f64::from(engine.state.cell.extent.y)
        * f64::from(engine.state.cell.extent.z);
    let drift = ((v1 - v0) / v0).abs();
    assert!(
        drift < 0.02,
        "NVT box volume drifted {drift:.3} over the rollout — barostat leaked into the path"
    );
}

#[test]
fn pressure_unit_chain_scale() {
    // Pure unit-composition check for the pressure chain (kcal/mol/Å³→bar):
    // Water at 1 g/cm³ is 0.0334 molecules/Å³; the molarized engine units
    // (kcal/mol per particle, R·T ≈ 0.596 kcal/mol at 300 K) give the ideal
    // NkT/V as n_particles_per_Å³ × R·T ÷ N_A... which in these units is
    // exactly n × R·T, because the /N_A and ×N_A cancel between "per mol"
    // energies and particle counts: 0.0334 × 0.596 × 69477 ≈ 1383 bar.
    // Liquid water's −1382 bar configurational virial is what brings the net
    // back to ~1 bar. (Pre-sign-fix the engine computed it the wrong way.)
    const KCAL_MOL_A3_TO_BAR: f64 = 69_476.954_570_553_73;
    const R_KCAL: f64 = 0.001_987_204_1;
    let particles_per_a3 = 0.0334;
    let p_bar = particles_per_a3 * R_KCAL * 300.0 * KCAL_MOL_A3_TO_BAR;
    assert!(
        (p_bar - 1383.4).abs() < 5.0,
        "ideal-gas reference scale changed: {p_bar} bar"
    );
}
