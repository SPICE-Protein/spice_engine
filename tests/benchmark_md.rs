//! Detailed MD step benchmark.
//!
//! Run with:
//!   cargo test --release --test benchmark_md -- --nocapture
//!
//! This intentionally separates build/solvent initialization from production
//! steps. It reports per-step wall time and the engine's accumulated internal
//! phase counters.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use bio_files::MmCif;
use dynamics::ComputationDevice;
use dynamics::params::FfParamSet;
use spice_engine::{BuildOptions, build_system};

#[test]
fn benchmark_production_steps_detailed() {
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("load Amber parameters");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");

    let build_start = Instant::now();
    let mut engine =
        build_system(&dev, &params, protein, &BuildOptions::default()).expect("build engine");
    let build_ms = build_start.elapsed().as_secs_f64() * 1_000.0;

    // Warm-up steps are excluded from the reported production statistics.
    for _ in 0..10 {
        let result = engine.step(None);
        assert!(
            !result.crashed,
            "warm-up crashed at step {}",
            result.step_count
        );
    }
    engine.state.computation_time = Default::default();
    engine.state.initialize_velocities(310.0, true);
    println!(
        "layout atoms={} water={} pairs={} simd_pairs={} scalar_pairs={}",
        engine.state.atoms.len(),
        engine.state.water.len(),
        engine.state.nb_pair_count(),
        engine.state.simd_pair_count(),
        engine.state.scalar_pair_count(),
    );

    const STEPS: usize = 100;
    let mut wall_ms = Vec::with_capacity(STEPS);
    let mut kind_times = BTreeMap::<&'static str, Vec<f64>>::new();
    let mut virial_nonfinite = 0usize;
    let mut ordinary_steps = 0usize;
    let mut pme_steps = 0usize;
    let mut rebuild_steps = 0usize;
    let mut pme_rebuild_steps = 0usize;
    let mut kind_virial = BTreeMap::<&'static str, [f64; 4]>::new();
    let mut kind_virial_nonfinite = BTreeMap::<&'static str, usize>::new();
    let mut first_virial: Option<[f64; 4]> = None;
    let mut last_virial: Option<[f64; 4]> = None;
    for _ in 0..STEPS {
        let start = Instant::now();
        let result = engine.step(None);
        let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
        assert!(
            !result.crashed,
            "production crashed at step {}",
            result.step_count
        );
        wall_ms.push(elapsed);
        let kind = engine.state.step_kind();
        kind_times.entry(kind).or_default().push(elapsed);
        match kind {
            "ordinary" => ordinary_steps += 1,
            "pme" => pme_steps += 1,
            "rebuild" => rebuild_steps += 1,
            "pme+rebuild" => pme_rebuild_steps += 1,
            _ => unreachable!("unknown step kind"),
        }
        let (v_bonded, v_short, v_long, v_constraints) = engine.state.virial_components();
        let virials = [v_bonded, v_short, v_long, v_constraints];
        if !virials.iter().all(|v| v.is_finite()) {
            virial_nonfinite += 1;
            *kind_virial_nonfinite.entry(kind).or_default() += 1;
        }
        if first_virial.is_none() {
            first_virial = Some(virials);
        }
        last_virial = Some(virials);
        let sums = kind_virial.entry(kind).or_default();
        for (sum, value) in sums.iter_mut().zip(virials) {
            if value.is_finite() {
                *sum += value;
            }
        }
        println!(
            "step={} kind={} wall_ms={elapsed:.3} u_kcal={:.3} virial={{bonded:{v_bonded:.3},short:{v_short:.3},long:{v_long:.3},constraints:{v_constraints:.3}}}",
            result.step_count, kind, result.u_t_kcal
        );
    }

    wall_ms.sort_by(f64::total_cmp);
    let sum: f64 = wall_ms.iter().sum();
    let pct = |p: f64| -> f64 {
        let index = ((STEPS - 1) as f64 * p).round() as usize;
        wall_ms[index]
    };
    let phase = engine
        .state
        .computation_time
        .time_per_step(STEPS)
        .expect("phase timing");

    println!("=== md benchmark ===");
    println!("build_ms={build_ms:.3}");
    println!(
        "steps={STEPS} wall_mean_ms={:.3} wall_p50_ms={:.3} wall_p95_ms={:.3} wall_p99_ms={:.3}",
        sum / STEPS as f64,
        pct(0.50),
        pct(0.95),
        pct(0.99)
    );
    println!(
        "phase_mean_us bonded={} nonbonded_short={} ewald_long={} neighbor_all={} neighbor_rebuild={} integration={} ambient={} snapshot={} total={}",
        phase.bonded,
        phase.non_bonded_short_range,
        phase.ewald_long_range,
        phase.neighbor_all,
        phase.neighbor_rebuild,
        phase.integration,
        phase.ambient,
        phase.snapshots,
        phase.total
    );
    for (kind, samples) in &kind_times {
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let max = samples.iter().copied().fold(0.0, f64::max);
        println!(
            "class kind={kind} count={} mean_ms={mean:.3} max_ms={max:.3}",
            samples.len()
        );
    }
    println!("virial_nonfinite_steps={virial_nonfinite}");
    println!(
        "virial_first={:?} virial_last={:?}",
        first_virial, last_virial
    );
    for (kind, sums) in &kind_virial {
        let count = kind_times.get(kind).map_or(0, Vec::len) as f64;
        println!(
            "class_virial kind={kind} mean={{bonded:{:.3},short:{:.3},long:{:.3},constraints:{:.3}}} nonfinite={}",
            sums[0] / count.max(1.0),
            sums[1] / count.max(1.0),
            sums[2] / count.max(1.0),
            sums[3] / count.max(1.0),
            kind_virial_nonfinite.get(kind).copied().unwrap_or(0)
        );
    }
    println!(
        "step_class_counts ordinary={ordinary_steps} pme={pme_steps} rebuild={rebuild_steps} pme_rebuild={pme_rebuild_steps}"
    );
    println!(
        "neighbor_rebuild_count={} pme_ratio={:.3} simd_pairs={} scalar_pairs={} total_pairs={}",
        engine.state.computation_time.neighbor_rebuild_count,
        engine.state.computation_time.ewald_long_range_sum as f64
            / engine.state.computation_time.total.max(1) as f64,
        engine.state.simd_pair_count(),
        engine.state.scalar_pair_count(),
        engine.state.nb_pair_count(),
    );
}
