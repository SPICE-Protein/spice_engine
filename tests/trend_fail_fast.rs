//! Trend-detector fail-fast acceptance (v1.3.4).
//!
//! The soft trend detector that used to live only in the stability scan is now
//! mounted on the engine (`SpiceEngine::set_trend_monitor`): every `step()`
//! feeds it U (free) + Rg/SS (O(solute), on `check_every` gates), and once a
//! sliding window shows ≥2 of 3 signals trending the wrong way the step returns
//! `crashed = true` with a `trend_alarm` — the RL loop stops BEFORE the hard
//! NaN blow-up. Two things must hold, checked on one real 2LYZ build:
//!
//! 1. NO FALSE POSITIVE — a healthy equilibrated NVT rollout (the recommended
//!    RL path) must not trip the monitor over a long stretch.
//! 2. EARLY KILL — a hot-start NPT rollout (barostat on the ~+38 kbar cold box,
//!    the v1.3.3 detonation) must be terminated by `trend_alarm` strictly
//!    before it reaches numerical crash, so a doomed episode costs ~300 steps
//!    of wall time instead of ~900 + a hard NaN.
//!
//! Run: `cargo test --release --test trend_fail_fast -- --ignored --nocapture --test-threads=1`

use bio_files::MmCif;
use spice_engine::engine::md_core::ComputationDevice;
use spice_engine::engine::md_core::params::FfParamSet;
use spice_engine::engine::{SpiceEngine, TrendConfig};
use spice_engine::env::EnvParams;
use spice_engine::{BuildOptions, EquilConfig, build_system};
use std::path::Path;

fn build_2lyz() -> SpiceEngine {
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("load Amber parameters");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
    // Run the NVT ramp during build (barostat frozen inside equilibrate).
    let opts = BuildOptions {
        env: EnvParams::new(7.0, 300.0, 1.0, 0.15),
        equil: Some(EquilConfig::default()),
        ..Default::default()
    };
    build_system(&dev, &params, protein, &opts).expect("build engine")
}

/// 1. Healthy NVT rollout must not trip the monitor (false-positive guard).
#[test]
#[ignore = "expensive: 2LYZ build + 1000 NVT steps with the monitor armed"]
fn healthy_nvt_does_not_trip_trend() {
    let mut engine = build_2lyz();
    engine.state.cfg.barostat_cfg = None; // the RL-recommended fixed-density path
    // Discard the strong-ramp → weak-γ production handover transient (T peaks
    // ~410 then settles ~320) exactly like the budget test, THEN arm so the
    // window only ever sees steady production.
    for _ in 0..650 {
        engine.step(None);
    }
    engine.set_trend_monitor(TrendConfig::rl_fail_fast());
    let mut tripped = None;
    for k in 0..1000 {
        let r = engine.step(None);
        assert!(
            !r.crashed,
            "[healthy] crashed at step {k}: {:?}",
            r.crash_reason
        );
        if let Some(sig) = r.trend_alarm {
            tripped = Some((k, sig));
            break;
        }
    }
    assert!(
        tripped.is_none(),
        "[healthy] false positive: trend tripped at step {:?} on a stable NVT run",
        tripped
    );
    println!("[healthy] 1000 NVT steps, trend monitor armed, never tripped ✓");
}

/// 2. Hot-start NPT detonation must be killed by trend_alarm BEFORE hard crash.
#[test]
#[ignore = "expensive: 2LYZ build + hot-start NPT until trend kill / crash"]
fn npt_hot_start_is_killed_early() {
    let mut engine = build_2lyz();
    // Barostat ON the still-overpressured cold box — the v1.3.3 detonation.
    assert!(
        engine.state.cfg.barostat_cfg.is_some(),
        "expected NPT build"
    );
    engine.set_trend_monitor(TrendConfig::rl_fail_fast());

    let max_steps = 1500;
    let mut trend_step: Option<usize> = None;
    let mut hard_crash_step: Option<usize> = None;
    for k in 0..max_steps {
        let r = engine.step(None);
        if r.trend_alarm.is_some() {
            trend_step = Some(k);
            break;
        }
        if r.crashed {
            hard_crash_step = Some(k);
            break;
        }
    }
    match (trend_step, hard_crash_step) {
        (Some(ts), crash) => {
            println!(
                "[npt] trend ALARM at step {ts} (crash {:?}) ✓ fail-fast",
                crash
            );
            assert!(
                crash.is_none_or(|cs| ts < cs),
                "[npt] trend alarm (step {ts}) must precede hard crash ({crash:?})"
            );
            // Must fire well before the ~620–930 NaN window: it filled the
            // 300-step window then needed the ≥2/3 slopes to go significant.
            assert!(
                ts < 600,
                "[npt] alarm too late to save the episode: step {ts}"
            );
        }
        (None, Some(cs)) => {
            panic!(
                "[npt] system HARD-CRASHED at step {cs} without a trend kill — fail-fast missed it"
            );
        }
        (None, None) => {
            panic!(
                "[npt] neither trend nor crash in {max_steps} steps — the hot-start \
                 detonation repro silently broke (was this box's build density fixed?)"
            );
        }
    }
}
