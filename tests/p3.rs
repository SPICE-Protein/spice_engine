//! P3 integration test: external structure input, mutation tools, engine pool,
//! and time-averaged Cα pseudo-labels.

use bio_files::MmCif;
use dynamics::params::FfParamSet;
use dynamics::ComputationDevice;
use spice_engine::{
    AtomInput, BuildOptions, EnginePool, Mutation, StructureInput, apply_mutations, build_from_input,
};
use std::path::Path;

/// Convert an MmCif (used as a *test fixture* for building a `StructureInput`) into
/// Convert an MmCif (used as a *test fixture* for building a `StructureInput`) into
/// an in-memory structure. In production, the Python side supplies `StructureInput`
/// directly via FFI.
fn mmcif_to_structure(mm: &MmCif) -> StructureInput {
    let mut input = StructureInput::default();
    for r in &mm.residues {
        // Test fixture: only amino-acid residues (skip crystal waters etc.).
        if matches!(r.res_type, bio_files::ResidueType::Water) {
            continue;
        }
        let res_name = match &r.res_type {
            bio_files::ResidueType::AminoAcid(aa) => {
                aa.to_str(na_seq::AaIdent::ThreeLetters).to_string()
            }
            bio_files::ResidueType::Water => "HOH".to_string(),
            bio_files::ResidueType::Other(n) => n.clone(),
        };
        for sn in &r.atom_sns {
            let Some(a) = mm.atoms.iter().find(|a| &a.serial_number == sn) else {
                continue;
            };
            input.push(AtomInput {
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
            });
        }
    }
    input
}

#[test]
fn p3_structure_mutate_pool() {
    let dev = ComputationDevice::Cpu;
    let param_set = FfParamSet::new_amber().expect("load amber params");
    let mm = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");

    // --- StructureInput round-trip: 129 residues, sequence inference ---
    let input = mmcif_to_structure(&mm);
    let n_res_in = input.residue_count();
    let seq = input.sequence().expect("infer sequence");
    println!("StructureInput: {n_res_in} residues, sequence len {}", seq.len());
    assert_eq!(n_res_in, 129);
    assert_eq!(seq.len(), 129);

    // --- build engine from in-memory structure (not from the mmCIF file) ---
    let mut engine =
        build_from_input(&dev, &param_set, &input, &BuildOptions::default()).expect("build from input");
    assert_eq!(engine.topology.sequence.len(), 129);
    assert_eq!(engine.topology.ca_indices.len(), 129);

    // --- pseudo-labels: time-averaged Cα ---
    engine.reset_pseudo_labels();
    assert!(engine.time_averaged_ca().is_empty(), "no frames yet");
    for _ in 0..3 {
        let r = engine.step(None);
        assert!(!r.crashed, "crashed at step {}", r.step_count);
    }
    let avg = engine.time_averaged_ca();
    assert_eq!(avg.len(), 129);
    // average must be finite; each frame's Cα is finite too
    let cur_ca: Vec<[f32; 3]> = engine
        .topology
        .ca_indices
        .iter()
        .map(|&i| {
            let p = engine.state.atoms[i].posit;
            [p.x, p.y, p.z]
        })
        .collect();
    for (a, c) in avg.iter().zip(cur_ca.iter()) {
        assert!(a[0].is_finite() && a[1].is_finite() && a[2].is_finite());
        let _ = c;
    }

    // --- mutation tools ---
    let m1 = apply_mutations(&engine.topology.sequence, &[Mutation::new(0, 'A')]).unwrap();
    assert_eq!(m1.as_bytes()[0] as char, 'A');
    assert!(apply_mutations(&engine.topology.sequence, &[Mutation::new(0, 'X')]).is_err());

    // --- EnginePool (1 worker here; MdState is now Send so rayon can parallelize) ---
    let mut pool = EnginePool::new(vec![engine]);
    assert_eq!(pool.len(), 1);
    let action = vec![0.3f32; 16];
    let results = pool.step_all(&[action]).expect("step_all");
    assert_eq!(results.len(), 1);
    assert!(!results[0].crashed);
    let metrics = pool.metrics_all();
    assert_eq!(metrics.len(), 1);
    assert!(metrics[0].m4 >= 0.0 && metrics[0].m4 <= 1.0);
    pool.set_temperature_all(300.0);
    assert_eq!(pool.worker(0).engine.state.cfg.temp_target, 300.0);

    println!("P3 structure/mutate/pool OK");
}
