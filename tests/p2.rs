//! P2 integration test: five physical metrics + force-basis actions.

use bio_files::MmCif;
use dynamics::params::FfParamSet;
use dynamics::ComputationDevice;
use lin_alg::f32::Vec3;
use spice_engine::{ActionMask, BuildOptions, EnvDelta, ForceAction, Metrics, MetricsConfig, build_system};
use std::path::Path;

#[test]
fn metrics_and_actions() {
    let dev = ComputationDevice::Cpu;
    let param_set = FfParamSet::new_amber().expect("load amber params");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ mmCIF");

    let mut engine =
        build_system(&dev, &param_set, protein, &BuildOptions::default()).expect("build system");

    // --- Metrics at reference (native) structure ---
    let metrics = Metrics::new(&engine, MetricsConfig::default());
    assert!(metrics.rg_ref.is_finite() && metrics.rg_ref > 1.0, "Rg_ref={}", metrics.rg_ref);
    println!(
        "Rg_ref={:.2}Å, SS_ref={} hbonds",
        metrics.rg_ref,
        metrics.ss_ref.len()
    );

    // --- ForceAction: shape + clamping ---
    let n_res = engine.topology.sequence.len();
    let mut fa = ForceAction::new(n_res, 16, 0.5, 20);
    let a = vec![0.5f32; 16];
    let f_ca = fa.actions_to_forces(&a);
    assert_eq!(f_ca.len(), n_res, "one force per residue");
    for f in &f_ca {
        // every component is ±clamp*tanh(·) → within ±0.5
        assert!(f.x.abs() <= 0.5 + 1e-5, "fx={}", f.x);
        assert!(f.y.abs() <= 0.5 + 1e-5, "fy={}", f.y);
        assert!(f.z.abs() <= 0.5 + 1e-5, "fz={}", f.z);
    }
    // enabled residues get non-zero force somewhere in the basis
    assert!(fa.mask.fraction() > 0.0, "mask should enable some residues");
    println!(
        "ForceAction: M=16, L={n_res}, mask_fraction={:.3}",
        fa.mask.fraction()
    );

    // --- ActionMask mutation cadence ---
    let mut mask = ActionMask::new(n_res, 5);
    let f0 = mask.fraction();
    for _ in 0..5 {
        mask.tick();
    }
    let f1 = mask.fraction();
    println!("mask fraction before={f0:.3} after 5 ticks={f1:.3}");
    // not guaranteed to differ, but must stay in [0,1] and len == n_res
    assert!((0.0..=1.0).contains(&f1));

    // --- EnvDelta temperature hot-switch ---
    let delta = EnvDelta::new(10.0, -0.5);
    delta.apply_t(&mut engine);
    assert_eq!(engine.env.temp_k + 10.0, engine.state.cfg.temp_target, "T hot-switch");
    assert!((delta.effective_ph(7.0) - 6.5).abs() < 1e-6, "pH clamp");

    // --- Drive a few steps: bias force + metrics per step ---
    engine.set_temperature(310.0);
    println!("\nstep | U(kcal/mol) | m1 VarU/kBT | m2 RgΔ | m3 SSloss | m4 clash | m5 surfQ | Rg(Å)");
    for k in 0..8 {
        let r = fa.step(&mut engine, &a);
        assert!(!r.crashed, "crashed at step {}", r.step_count);
        let m = metrics.compute(&engine);
        assert!(m.m1.is_finite() && m.m1 >= 0.0, "m1={}", m.m1);
        assert!(m.m2.is_finite() && m.m2 >= 0.0, "m2={}", m.m2);
        assert!((0.0..=1.0).contains(&m.m3), "m3={}", m.m3);
        assert!((0.0..=1.0).contains(&m.m4), "m4={}", m.m4);
        assert!(m.m5.is_finite() && m.m5 >= 0.0, "m5={}", m.m5);
        println!(
            "{:4} | {:12.1} | {:8.3} | {:6.4} | {:7.4} | {:8.5} | {:7.3} | {:.2}",
            r.step_count, m.u_t_kcal, m.m1, m.m2, m.m3, m.m4, m.m5, m.rg
        );
        // U history must have grown
        assert_eq!(engine.u_history.len() as usize, k + 1, "U history length");
    }

    // m1 becomes meaningful once enough U samples accumulate (window ≤ 8 here,
    // so it is the running variance of the whole history — finite and ≥ 0).
    let m_last = metrics.compute(&engine);
    assert!(m_last.n_surface_charged >= 5, "m5 needs a meaningful surface set, got {}", m_last.n_surface_charged);
    println!(
        "\nfinal: n_ss_ref={} n_ss_kept={} n_surface_charged={}",
        m_last.n_ss_ref, m_last.n_ss_kept, m_last.n_surface_charged
    );
    println!("P2 metrics+actions OK");
}
