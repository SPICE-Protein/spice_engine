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
use spice_engine::engine::dynamics::ComputationDevice;
use spice_engine::engine::dynamics::params::FfParamSet;
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
        "phase_mean_us bonded={} nonbonded_short={} ewald_long={} neighbor_all={} neighbor_rebuild={} integration={} ambient={} kinetic={} water_settle={} thermostat={} barostat={} snapshot={} total={}",
        phase.bonded,
        phase.non_bonded_short_range,
        phase.ewald_long_range,
        phase.neighbor_all,
        phase.neighbor_rebuild,
        phase.integration,
        phase.ambient,
        phase.kinetic,
        phase.water_settle,
        phase.thermostat,
        phase.barostat,
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

/// Neighbor skin trade-off sweep. Run with:
/// `cargo test --release --test benchmark_md benchmark_neighbor_skin_sweep -- --ignored --nocapture`
#[test]
#[ignore = "expensive solvated neighbor skin sweep"]
fn benchmark_neighbor_skin_sweep() {
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("load Amber parameters");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
    for skin in [1.0_f32, 2.0, 3.0, 4.0] {
        let mut opts = BuildOptions::default();
        opts.neighbor_skin = skin;
        let mut engine = build_system(&dev, &params, protein.clone(), &opts)
            .unwrap_or_else(|e| panic!("build failed for skin={skin}: {e}"));
        for _ in 0..3 {
            assert!(!engine.step(None).crashed);
        }
        engine.state.computation_time = Default::default();
        let start = Instant::now();
        for _ in 0..20 {
            assert!(!engine.step(None).crashed);
        }
        let wall_ms = start.elapsed().as_secs_f64() * 1_000.0 / 20.0;
        let phase = engine
            .state
            .computation_time
            .time_per_step(20)
            .expect("phase timing");
        println!(
            "skin_A={skin:.2} wall_mean_ms={wall_ms:.3} neighbor_all_us={} rebuild_us={} rebuild_count={}",
            phase.neighbor_all,
            phase.neighbor_rebuild,
            engine.state.computation_time.neighbor_rebuild_count
        );
    }
}

/// Regression guard for reciprocal-space cache policy. SPME_RATIO is intentionally
/// one in dynamics; this confirms every measured step has fresh PME forces and
/// finite energy/virial instead of silently consuming stale cached forces.
/// Run with: `cargo test --release --test benchmark_md benchmark_pme_cache_regression -- --ignored --nocapture`
/// Compare the optimized CPU nonbonded dispatcher with the scalar reference
/// on identical coordinates. The environment switch is process-global, so this
/// test is intentionally ignored and should be run with one test thread.
/// Run with: `cargo test --release --test benchmark_md nonbonded_reference_vs_optimized -- --ignored --nocapture --test-threads=1`
#[test]
#[ignore = "expensive reference-vs-optimized force comparison"]
fn nonbonded_reference_vs_optimized() {
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("load Amber parameters");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
    let engine =
        build_system(&dev, &params, protein, &BuildOptions::default()).expect("build engine");

    let mut reference = engine.state.clone();
    for a in &mut reference.atoms {
        a.force = lin_alg::f32::Vec3::new_zero();
    }
    for w in &mut reference.water {
        w.o.force = lin_alg::f32::Vec3::new_zero();
        w.m.force = lin_alg::f32::Vec3::new_zero();
        w.h0.force = lin_alg::f32::Vec3::new_zero();
        w.h1.force = lin_alg::f32::Vec3::new_zero();
    }
    reference.potential_energy = 0.0;
    reference.potential_energy_nonbonded = 0.0;
    unsafe { std::env::set_var("SPICE_NONBONDED_REFERENCE", "1") };
    let reference_start = Instant::now();
    reference.apply_nonbonded_forces(&dev);
    let reference_us = reference_start.elapsed().as_micros();
    let reference_forces: Vec<_> = reference.atoms.iter().map(|a| a.force).collect();
    let reference_water: Vec<_> = reference
        .water
        .iter()
        .map(|w| [w.o.force, w.m.force, w.h0.force, w.h1.force])
        .collect();
    let reference_energy = reference.potential_energy_nonbonded;
    let reference_virial = reference.virial_components().1;
    let reference_pressure = f64::NAN;

    let mut optimized = engine.state;
    for a in &mut optimized.atoms {
        a.force = lin_alg::f32::Vec3::new_zero();
    }
    for w in &mut optimized.water {
        w.o.force = lin_alg::f32::Vec3::new_zero();
        w.m.force = lin_alg::f32::Vec3::new_zero();
        w.h0.force = lin_alg::f32::Vec3::new_zero();
        w.h1.force = lin_alg::f32::Vec3::new_zero();
    }
    optimized.potential_energy = 0.0;
    optimized.potential_energy_nonbonded = 0.0;
    unsafe { std::env::remove_var("SPICE_NONBONDED_REFERENCE") };
    let optimized_start = Instant::now();
    optimized.apply_nonbonded_forces(&dev);
    let optimized_us = optimized_start.elapsed().as_micros();
    let optimized_forces: Vec<_> = optimized.atoms.iter().map(|a| a.force).collect();
    let optimized_water: Vec<_> = optimized
        .water
        .iter()
        .map(|w| [w.o.force, w.m.force, w.h0.force, w.h1.force])
        .collect();

    assert_eq!(reference_forces.len(), optimized_forces.len());
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (a, b) in reference_forces.iter().zip(&optimized_forces) {
        for (x, y) in [(a.x, b.x), (a.y, b.y), (a.z, b.z)] {
            max_abs = max_abs.max((x - y).abs());
            max_rel = max_rel.max((x - y).abs() / x.abs().max(y.abs()).max(1.0));
        }
    }
    for (wa, wb) in reference_water.iter().zip(&optimized_water) {
        for (a, b) in wa.iter().zip(wb) {
            for (x, y) in [(a.x, b.x), (a.y, b.y), (a.z, b.z)] {
                max_abs = max_abs.max((x - y).abs());
                max_rel = max_rel.max((x - y).abs() / x.abs().max(y.abs()).max(1.0));
            }
        }
    }
    let optimized_energy = optimized.potential_energy_nonbonded;
    let optimized_virial = optimized.virial_components().1;
    let energy_abs = (reference_energy - optimized_energy).abs();
    let virial_abs = (reference_virial - optimized_virial).abs();
    let optimized_pressure = f64::NAN;
    let pressure_abs = (reference_pressure - optimized_pressure).abs();
    println!(
        "reference_vs_optimized reference_us={reference_us} optimized_us={optimized_us} max_force_abs={max_abs:.6e} max_force_rel={max_rel:.6e} energy_abs={energy_abs:.6e} virial_abs={virial_abs:.6e} pressure_abs={pressure_abs:.6e} reference_energy={reference_energy:.6e} optimized_energy={optimized_energy:.6e}"
    );
    assert!(
        max_rel < 5.0e-4,
        "force mismatch: abs={max_abs} rel={max_rel}"
    );
    assert!(energy_abs < 1.0e-3, "energy mismatch: {energy_abs}");
    assert!(virial_abs < 1.0e-2, "virial mismatch: {virial_abs}");
    if reference_pressure.is_finite() && optimized_pressure.is_finite() {
        assert!(pressure_abs < 1.0e-3, "pressure mismatch: {pressure_abs}");
    }
}

#[test]
#[ignore = "expensive solvated PME cache regression"]
fn benchmark_pme_cache_regression() {
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("load Amber parameters");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
    let mut engine =
        build_system(&dev, &params, protein, &BuildOptions::default()).expect("build engine");
    for _ in 0..3 {
        assert!(!engine.step(None).crashed);
    }
    engine.state.computation_time = Default::default();
    for _ in 0..10 {
        let result = engine.step(None);
        assert!(!result.crashed);
        assert!(result.u_t_kcal.is_finite());
        assert!(engine.state.virial_components().2.is_finite());
        assert!(engine.state.last_step_used_pme());
    }
    let rebuilds = engine.state.computation_time.neighbor_rebuild_count;
    assert!(rebuilds <= 10, "unexpected rebuild count: {rebuilds}");
    println!(
        "pme_cache_policy=refresh_every_step pme_steps=10 ewald_us={} long_virial={:.6} rebuilds={rebuilds}",
        engine.state.computation_time.ewald_long_range_sum,
        engine.state.virial_components().2
    );
}

/// Explicit reciprocal-space tuning sweep. Run with:
/// `cargo test --release --test benchmark_md benchmark_pme_parameter_sweep -- --ignored --nocapture`
#[test]
#[ignore = "expensive solvated PME parameter sweep"]
fn benchmark_pme_parameter_sweep() {
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("load Amber parameters");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
    let cases = [(1.25_f32, 0.22_f32), (1.00, 0.26), (0.80, 0.30)];
    println!("=== PME parameter sweep (mesh_spacing_A, alpha_A^-1) ===");
    for (mesh_spacing, alpha) in cases {
        let mut opts = BuildOptions::default();
        opts.spme_mesh_spacing = mesh_spacing;
        opts.spme_alpha = alpha;
        let mut engine = build_system(&dev, &params, protein.clone(), &opts).unwrap_or_else(|e| {
            panic!("build failed for spacing={mesh_spacing}, alpha={alpha}: {e}")
        });
        for _ in 0..3 {
            let result = engine.step(None);
            assert!(
                !result.crashed,
                "warm-up crashed for spacing={mesh_spacing}, alpha={alpha}"
            );
        }
        engine.state.computation_time = Default::default();
        let mut wall = Vec::with_capacity(10);
        let mut energy_sum = 0.0;
        let mut long_virial_sum = 0.0;
        let mut pme_steps = 0usize;
        for _ in 0..10 {
            let start = Instant::now();
            let result = engine.step(None);
            assert!(
                !result.crashed,
                "production crashed for spacing={mesh_spacing}, alpha={alpha}"
            );
            wall.push(start.elapsed().as_secs_f64() * 1_000.0);
            energy_sum += result.u_t_kcal;
            long_virial_sum += engine.state.virial_components().2;
            if engine.state.step_kind().contains("pme") {
                pme_steps += 1;
            }
        }
        let mean_ms = wall.iter().sum::<f64>() / wall.len() as f64;
        let phase = engine
            .state
            .computation_time
            .time_per_step(10)
            .expect("phase timing");
        let e_mean = energy_sum / 10.0;
        let v_mean = long_virial_sum / 10.0;
        assert!(e_mean.is_finite() && v_mean.is_finite());
        println!(
            "spacing_A={mesh_spacing:.2} alpha_A^-1={alpha:.3} mean_ms={mean_ms:.3} ewald_mean_us={} energy_mean_kcal={e_mean:.3} long_virial_mean={v_mean:.3} pme_steps={pme_steps} rebuilds={}",
            phase.ewald_long_range, engine.state.neighbor_rebuild_count,
        );
    }
}
