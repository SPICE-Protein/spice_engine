use std::path::Path;
use bio_files::MmCif;
use dynamics::params::FfParamSet;
use dynamics::ComputationDevice;
use spice_engine::{
    build_system, calculate_engine_pockets, calculate_pockets_native, is_atom_hydrophobic,
    calculate_advanced_features, calculate_pocket_delta, analyze_pocket_trajectory,
    BuildOptions,
};

#[test]
fn test_pocket_calculation_on_cif() {
    // 1. Load protein CIF structure
    let protein_path = Path::new("data/test/2LYZ.cif");
    let protein = MmCif::load(protein_path).expect("failed to load 2LYZ mmCIF");

    // Extract atom coordinates, residue indices, and determine hydrophobicity
    let mut coords = Vec::new();
    let mut residue_indices = Vec::new();
    let mut is_hydrophobic = Vec::new();

    // Map serial numbers to index to associate with residues
    let mut atom_to_res = std::collections::HashMap::new();
    for (res_idx, res) in protein.residues.iter().enumerate() {
        for &sn in &res.atom_sns {
            atom_to_res.insert(sn, (res_idx, res.res_type.clone()));
        }
    }

    for a in &protein.atoms {
        if a.hetero {
            continue;
        }
        if a.element == na_seq::Element::Hydrogen {
            continue;
        }
        coords.push([a.posit.x as f32, a.posit.y as f32, a.posit.z as f32]);

        let (res_idx, res_type) = atom_to_res
            .get(&a.serial_number)
            .cloned()
            .unwrap_or((0, bio_files::ResidueType::Water));

        residue_indices.push(res_idx);

        let one_letter = match &res_type {
            bio_files::ResidueType::AminoAcid(aa) => {
                use na_seq::AminoAcid::*;
                match aa {
                    Arg => 'R', His => 'H', Lys => 'K', Asp => 'D', Glu => 'E',
                    Ser => 'S', Thr => 'T', Asn => 'N', Gln => 'Q', Cys => 'C',
                    Sec => 'U', Gly => 'G', Pro => 'P', Ala => 'A', Val => 'V',
                    Ile => 'I', Leu => 'L', Met => 'M', Phe => 'F', Tyr => 'Y',
                    Trp => 'W',
                }
            }
            _ => 'X',
        };

        is_hydrophobic.push(is_atom_hydrophobic(a.element, one_letter));
    }

    // 2. Compute pockets using the raw native API
    println!("Calculating pockets with raw native API on {} heavy atoms...", coords.len());
    let pockets = calculate_pockets_native(&coords, &residue_indices, &is_hydrophobic, 1.0);

    println!("Detected {} pockets using native API.", pockets.len());
    for p in &pockets {
        println!(
            "Pocket ID {}: Volume = {:.2} Å³, Druggability = {:.4}, Center = {:?}",
            p.id, p.volume, p.druggability, p.center
        );
        assert!(p.volume >= 200.0, "pockets should be filtered by minimum volume");
        assert!(p.druggability >= 0.0 && p.druggability <= 1.0, "druggability score must be in [0, 1]");
        assert!(!p.surface_residues.is_empty(), "pocket should touch some residues");
    }

    // Verify sorting order: descending by volume
    if pockets.len() > 1 {
        for i in 0..pockets.len() - 1 {
            assert!(
                pockets[i].volume >= pockets[i + 1].volume,
                "pockets should be sorted in descending order of volume"
            );
        }
    }
}

#[test]
fn test_pocket_calculation_on_engine() {
    let dev = ComputationDevice::Cpu;
    let param_set = FfParamSet::new_amber().expect("failed to load amber params");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("failed to load 2LYZ mmCIF");

    let opts = BuildOptions {
        relax_iters: Some(5), // Keep it low for fast integration tests
        ..BuildOptions::default()
    };

    println!("Building WT system for 2LYZ...");
    let engine = build_system(&dev, &param_set, protein, &opts).expect("failed to build system");

    println!("Calculating pockets via High-Level Engine API...");
    let pockets = calculate_engine_pockets(&engine, 1.0);

    println!("Engine API detected {} pockets.", pockets.len());
    for p in &pockets {
        println!(
            "Engine Pocket ID {}: Volume = {:.2} Å³, Druggability = {:.4}, Center = {:?}",
            p.id, p.volume, p.druggability, p.center
        );
        assert!(p.volume >= 200.0);
        assert!(p.druggability >= 0.0 && p.druggability <= 1.0);
        assert!(!p.surface_residues.is_empty());
    }

    if !pockets.is_empty() {
        // Test Advanced Features
        println!("Testing Advanced Features on first pocket...");
        let adv = calculate_advanced_features(&engine, &pockets[0]);
        println!("Advanced features: net_charge_at_ph = {:.4} e", adv.net_charge_at_ph);
        println!("Advanced features: hydrophobic_ratio = {:.4}", adv.hydrophobic_ratio);
        println!("Advanced features: hydrophobic_centroids len = {}", adv.hydrophobic_centroids.len());
        println!("Advanced features: hbond_acceptors len = {}", adv.hbond_acceptors.len());
        println!("Advanced features: hbond_donors len = {}", adv.hbond_donors.len());

        assert!(adv.net_charge_at_ph.is_finite());
        assert!(adv.hydrophobic_ratio >= 0.0 && adv.hydrophobic_ratio <= 1.0);

        // Test Pocket Delta
        println!("Testing Pocket Delta...");
        // Mutate the same pocket slightly in our coordinates to simulate mutation effect
        let mut mutant_pocket = pockets[0].clone();
        mutant_pocket.volume += 15.0; // simulated change
        if !mutant_pocket.voxels.is_empty() {
            mutant_pocket.voxels.remove(0); // slightly change voxels
        }
        let delta = calculate_pocket_delta(&pockets[0], &mutant_pocket, 1.0);
        println!("Pocket Delta: volume_delta = {:.2} Å³, Jaccard overlap = {:.4}", delta.volume_delta, delta.jaccard_overlap);
        assert_eq!(delta.volume_delta, 15.0);
        assert!(delta.jaccard_overlap >= 0.0 && delta.jaccard_overlap <= 1.0);

        // Test Pocket Trajectory analysis
        println!("Testing Pocket Trajectory analysis (2 samples, step size 1)...");
        let mut mut_engine = engine.clone();
        let (mad, open_prob) = analyze_pocket_trajectory(&mut mut_engine, 2, 1, 1.0, 200.0);
        println!("Trajectory Analysis: Volume MAD = {:.4} Å³, Open Probability = {:.4}", mad, open_prob);
        assert!(mad >= 0.0);
        assert!(open_prob >= 0.0 && open_prob <= 1.0);
    }
}

#[test]
fn test_pockets_on_all_repository_cifs() {
    let test_dir = Path::new("data/test");
    if !test_dir.exists() {
        println!("test directory does not exist, skipping multi-CIF test");
        return;
    }

    let files = vec![
        "1R2I.cif", "1T19.cif", "1XJ3.cif", "2LYZ.cif", "4LPX.cif", "5H3G.cif", "7G0M.cif", "8CWC.cif"
    ];

    for file_name in files {
        let file_path = test_dir.join(file_name);
        if !file_path.exists() {
            continue;
        }

        println!("------------------------------------------------------------");
        println!("Processing file: {}", file_name);
        let protein = MmCif::load(&file_path);
        if let Err(e) = protein {
            println!("Failed to load CIF {}: {:?}", file_name, e);
            continue;
        }
        let protein = protein.unwrap();

        let mut coords = Vec::new();
        let mut residue_indices = Vec::new();
        let mut is_hydrophobic = Vec::new();

        let mut atom_to_res = std::collections::HashMap::new();
        for (res_idx, res) in protein.residues.iter().enumerate() {
            for &sn in &res.atom_sns {
                atom_to_res.insert(sn, (res_idx, res.res_type.clone()));
            }
        }

        for a in &protein.atoms {
            if a.hetero {
                continue;
            }
            if a.element == na_seq::Element::Hydrogen {
                continue;
            }
            coords.push([a.posit.x as f32, a.posit.y as f32, a.posit.z as f32]);

            let (res_idx, res_type) = atom_to_res
                .get(&a.serial_number)
                .cloned()
                .unwrap_or((0, bio_files::ResidueType::Water));

            residue_indices.push(res_idx);

            let one_letter = match &res_type {
                bio_files::ResidueType::AminoAcid(aa) => {
                    use na_seq::AminoAcid::*;
                    match aa {
                        Arg => 'R', His => 'H', Lys => 'K', Asp => 'D', Glu => 'E',
                        Ser => 'S', Thr => 'T', Asn => 'N', Gln => 'Q', Cys => 'C',
                        Sec => 'U', Gly => 'G', Pro => 'P', Ala => 'A', Val => 'V',
                        Ile => 'I', Leu => 'L', Met => 'M', Phe => 'F', Tyr => 'Y',
                        Trp => 'W',
                    }
                }
                _ => 'X',
            };

            is_hydrophobic.push(is_atom_hydrophobic(a.element, one_letter));
        }

        let pockets = calculate_pockets_native(&coords, &residue_indices, &is_hydrophobic, 1.0);
        println!("File {}: Found {} pockets >= 200.0 Å³", file_name, pockets.len());
        if let Some(p) = pockets.first() {
            println!(
                "  Largest Pocket ID {}: Volume = {:.2} Å³, Druggability = {:.4}, Center = {:?}",
                p.id, p.volume, p.druggability, p.center
            );
        }
    }
}
