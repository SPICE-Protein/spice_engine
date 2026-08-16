//! Smoke test for custom dynamic protonation overrides and geometry-based His tautomer selection.

use std::collections::HashMap;
use std::path::Path;

use bio_files::MmCif;
use dynamics::params::FfParamSet;
use dynamics::ComputationDevice;
use na_seq::AminoAcidProtenationVariant;
use spice_engine::{BuildOptions, build_system};

#[test]
fn test_dynamic_protonation_and_his_geometry() {
    let dev = ComputationDevice::Cpu;
    let param_set = FfParamSet::new_amber().expect("load amber params");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ mmCIF");

    // Let's first check our geometry-based His selection by building without custom overrides.
    let opts_geom = BuildOptions::default();
    let engine_geom = build_system(&dev, &param_set, protein.clone(), &opts_geom)
        .expect("build system with geometry-based His");

    // Find the first Histidine residue dynamically
    let his_res_idx = engine_geom
        .topology
        .residues
        .iter()
        .position(|r| r.one_letter == 'H')
        .expect("should find at least one Histidine in Lysozyme");

    println!("Found Histidine at index: {}", his_res_idx);

    // Let's build another engine with an EXPLICIT custom protonation override.
    // We will force this Histidine to be HIP (doubly protonated) instead of its geometry-derived neutral tautomer.
    let mut custom_map = HashMap::new();
    custom_map.insert(his_res_idx, AminoAcidProtenationVariant::Hip);

    let opts_custom = BuildOptions {
        custom_protonation: Some(custom_map),
        ..BuildOptions::default()
    };

    let mut engine_custom = build_system(&dev, &param_set, protein, &opts_custom)
        .expect("build system with custom protonation override");

    // The two engines should have different charge distributions or atom counts on this His residue
    // because HIP has more hydrogen atoms and a different net charge than HID or HIE.
    let count_atoms_geom = engine_geom.topology.residues[his_res_idx].atom_indices.len();
    let count_atoms_custom = engine_custom.topology.residues[his_res_idx].atom_indices.len();

    println!(
        "His {} - Geom-based atoms: {}, Custom (HIP) atoms: {}",
        his_res_idx, count_atoms_geom, count_atoms_custom
    );

    // HIP (doubly protonated) must have more hydrogens (and thus more total atoms) than neutral His (HID or HIE).
    assert!(
        count_atoms_custom > count_atoms_geom,
        "HIP should have more atoms than neutral His due to the extra proton!"
    );

    // Let's run a few MD steps to verify physical stability under the custom protonation state.
    let r0 = engine_custom.step(None);
    assert!(r0.u_t_kcal.is_finite(), "U not finite with custom protonation");
    println!("Step 1 Potential Energy with HIP: {:.2} kcal/mol", r0.u_t_kcal);
}
