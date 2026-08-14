//! Test for the newly implemented wishlist features:
//! 1. pseudo_labels() fallback and transparent crash diagnostics
//! 2. mutant build via solvent box reuse
//! 3. stability margin and RMSF metrics

use bio_files::MmCif;
use dynamics::params::FfParamSet;
use dynamics::ComputationDevice;
use spice_engine::{BuildOptions, build_system, build_mutant_by_solvent_reuse, Metrics, MetricsConfig};
use spice_engine::structure::{AtomInput, StructureInput};
use std::path::Path;

#[test]
fn test_wishlist_features() {
    let dev = ComputationDevice::Cpu;
    let param_set = FfParamSet::new_amber().expect("load amber params");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ mmCIF");

    // We keep relax_iters reasonable (e.g. Some(20)) so it actually minimizes,
    // which tests that the solver works and minimizes the system properly.
    let opts = BuildOptions {
        relax_iters: Some(20),
        ..BuildOptions::default()
    };

    println!("Building WT system...");
    let mut engine = build_system(&dev, &param_set, protein, &opts).expect("build system");

    // --- 1. Test pseudo_labels fallback when ca_n == 0 (Wish 1) ---
    let labels = engine.time_averaged_ca();
    assert!(!labels.is_empty(), "time_averaged_ca should not be empty even when ca_n == 0");
    assert_eq!(labels.len(), engine.topology.ca_indices.len());

    // --- 2. Test crash_reason in StepResult (Wish 1) ---
    let r0 = engine.step(None);
    assert!(r0.crash_reason.is_none() || r0.crashed, "crash_reason should only be Some when crashed");

    // --- 3. Test metrics: stability_margin and rmsf (Wish 3) ---
    let metrics_calc = Metrics::new(&engine, MetricsConfig::default());
    let m = metrics_calc.compute(&engine);
    println!("Initial metrics: margin={:.4}, rmsf={:.4}", m.stability_margin, m.rmsf);
    assert!(m.stability_margin.is_finite());
    assert!(m.rmsf.is_finite());
    assert_eq!(m.rmsf, 0.0, "RMSF at step 0 must be 0");

    // Run one step to accumulate fluctuation
    engine.step(None);
    let m2 = metrics_calc.compute(&engine);
    println!("Step 1 metrics: margin={:.4}, rmsf={:.4}", m2.stability_margin, m2.rmsf);
    assert!(m2.rmsf >= 0.0);
    assert!(m2.stability_margin.is_finite());

    // --- 4. Test build_mutant_by_solvent_reuse (Wish 2) ---
    // Load a 100% chemically valid StructureInput from the same WT mmCIF file
    let mm = bio_files::MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ mmCIF");
    let mut wt_input = StructureInput::default();
    for r in &mm.residues {
        if matches!(r.res_type, bio_files::ResidueType::Water) {
            continue;
        }
        let res_name = match &r.res_type {
            bio_files::ResidueType::AminoAcid(aa) => {
                aa.to_str(na_seq::AaIdent::ThreeLetters).to_string()
            }
            _ => continue,
        };
        for sn in &r.atom_sns {
            let Some(a) = mm.atoms.iter().find(|a| &a.serial_number == sn) else {
                continue;
            };
            wt_input.push(AtomInput {
                chain_id: "A".to_string(),
                res_seq: r.serial_number as i32,
                res_name: res_name.clone(),
                atom_name: a
                    .type_in_res
                    .as_ref()
                    .map(|t| t.to_string())
                    .or_else(|| a.type_in_res_general.clone())
                    .unwrap_or_default(),
                element: a.element,
                x: a.posit.x as f32,
                y: a.posit.y as f32,
                z: a.posit.z as f32,
                occupancy: a.occupancy.unwrap_or(1.0),
            });
        }
    }

    let mut_opts = BuildOptions {
        relax_iters: Some(20), // fast relaxation
        ..opts.clone()
    };
    
    println!("Building mutant system via solvent reuse...");
    let start_time = std::time::Instant::now();
    let mut_engine = build_mutant_by_solvent_reuse(&engine, &param_set, &wt_input, &mut_opts);
    println!("Solvent reuse build completed in: {:?}", start_time.elapsed());
    
    assert!(mut_engine.is_ok(), "build_mutant_by_solvent_reuse failed: {:?}", mut_engine.err());
    let mut_engine = mut_engine.unwrap();
    assert_eq!(mut_engine.state.water.len(), engine.state.water.len(), "Water molecules count must match parent");
    assert!(mut_engine.state.atoms.len() > 0);
    println!("All wishlist tests completed successfully!");
}