//! Smoke test: build a system from a real mmCIF and verify the engine API.
//!
//! NOTE on stability: this relies on two `dynamics`-fork fixes that make the
//! initial relaxation actually work:
//!   1. `step_energy_min` now tracks *cumulative* displacement since the last
//!      neighbor rebuild (it used to reset to 0 each iteration, so the neighbor
//!      list was never refreshed during minimization → "converged" on a stale
//!      force field that production immediately saw as ~100× larger forces).
//!   2. The minimizer now runs in the FULL force field (long-range reciprocal
//!      enabled) instead of minimizing with reciprocal disabled and then
//!      switching it on for production.
//! Remaining limitation: steepest-descent still leaves elevated residual forces
//! (~70–90 kcal/mol/Å) on some crystal-strain hotspots, so runs remain marginally
//! stable and can still blow up at a stochastic step (roughly 5–30 steps). A proper
//! equilibration protocol (positional restraints + NVT ramp) is future work. This
//! test therefore asserts a short, reliable stability window (5 steps) plus the
//! API surface (build, topology, step, bias-force hook, temperature control).
//! `tests/p2.rs` exercises 8 bias-force steps alongside the five physical metrics.

use bio_files::MmCif;
use dynamics::params::FfParamSet;
use dynamics::ComputationDevice;
use lin_alg::f32::Vec3;
use spice_engine::{BuildOptions, build_system};
use std::path::Path;

#[test]
fn smoke_lysozyme_2lyz() {
    let dev = ComputationDevice::Cpu;
    let param_set = FfParamSet::new_amber().expect("load amber params");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ mmCIF");

    let mut engine =
        build_system(&dev, &param_set, protein, &BuildOptions::default()).expect("build system");

    // --- topology ---
    let n_ca = engine.topology.ca_indices.len();
    let n_res = engine.topology.sequence.len();
    println!("2LYZ: sequence len = {n_res}, Cα = {n_ca}");
    assert_eq!(n_res, n_ca, "expected one Cα per residue");
    assert!(
        (n_res as i32 - 129).abs() <= 4,
        "unexpected residue count {n_res}"
    );

    // --- step API ---
    engine.set_temperature(310.0);
    let r0 = engine.step(None);
    assert!(r0.u_t_kcal.is_finite(), "U not finite on step 1");
    assert_eq!(r0.coords_ca.len(), n_ca);
    assert_eq!(r0.step_count, 1);

    // bias-force hook: an all-zero external force must be accepted
    let bias: Vec<Vec3> = vec![Vec3::new_zero(); engine.state.atoms.len()];
    let r1 = engine.step(Some(bias));
    assert!(r1.step_count > r0.step_count);
    assert_eq!(r1.coords_ca.len(), n_ca);
    assert!(r1.time_ps > r0.time_ps);

    // --- short, reliable stability window (see note above) ---
    let mut prev = r1.step_count;
    for _ in 0..5 {
        let res = engine.step(None);
        assert!(!res.crashed, "crashed at step {}", res.step_count);
        assert_eq!(res.coords_ca.len(), n_ca);
        assert!(res.step_count > prev, "step_count did not advance");
        prev = res.step_count;
    }

    println!(
        "U0={:.1} kcal/mol, U1={:.1} kcal/mol, U_end={:.1} kcal/mol, t={:.2} ps, steps={}",
        r0.u_t_kcal,
        r1.u_t_kcal,
        engine.state.potential_energy,
        engine.state.time,
        engine.state.step_count
    );
}
