//! Temporary diagnostic: inspect serial-number/bond consistency after preparation.
//! Kept as #[ignore] — run explicitly with `cargo test --release -- --ignored prep_debug`.

use bio_files::MmCif;
use dynamics::params::{FfParamSet, prepare_peptide_mmcif};
use dynamics::ComputationDevice;
use na_seq::{AtomTypeInRes, Element};
use std::collections::HashSet;
use std::path::Path;

#[test]
#[ignore]
fn debug_prep() {
    let param_set = FfParamSet::new_amber().unwrap();
    let mut protein = MmCif::load(Path::new("data/test/2LYZ.cif")).unwrap();
    let h0 = protein
        .atoms
        .iter()
        .filter(|a| a.element == Element::Hydrogen)
        .count();
    let het0 = protein.atoms.iter().filter(|a| a.hetero).count();
    let waters0 = protein
        .residues
        .iter()
        .filter(|r| matches!(r.res_type, bio_files::ResidueType::Water))
        .count();
    println!(
        "H before prep: {h0}, atoms: {}, hetero: {het0}, residues: {}, water-residues: {waters0}",
        protein.atoms.len(),
        protein.residues.len()
    );

    let (bonds, _dih) =
        prepare_peptide_mmcif(&mut protein, &param_set.peptide_ff_q_map.as_ref().unwrap(), 7.0, None, true)
            .unwrap();
    let n = protein.atoms.len();
    println!("atoms after prep: {n}, bonds: {}", bonds.len());

    let serials: Vec<u32> = protein.atoms.iter().map(|a| a.serial_number).collect();
    let uniq: HashSet<u32> = serials.iter().copied().collect();
    println!(
        "serial range {}..={}  len={}  uniq={}",
        serials.iter().min().unwrap(),
        serials.iter().max().unwrap(),
        serials.len(),
        uniq.len()
    );

    let mut missing = 0;
    let mut dup = 0;
    for b in &bonds {
        if !serials.contains(&b.atom_0_sn) || !serials.contains(&b.atom_1_sn) {
            missing += 1;
        }
        if serials.iter().filter(|&&s| s == b.atom_0_sn).count() > 1 {
            dup += 1;
        }
    }
    println!("bonds referencing a missing serial: {missing} / {}", bonds.len());
    println!("bonds where atom_0_sn is duplicated: {dup}");
    for b in bonds.iter().take(6) {
        println!("  bond {} - {}", b.atom_0_sn, b.atom_1_sn);
    }

    // Inspect assigned ff types on CA atoms.
    let mut ca_types: std::collections::HashMap<String, u32> = Default::default();
    for a in &protein.atoms {
        if matches!(a.type_in_res, Some(AtomTypeInRes::CA)) {
            *ca_types
                .entry(a.force_field_type.clone().unwrap_or_default())
                .or_insert(0) += 1;
        }
    }
    println!(
        "CA atoms by ff_type: {ca_types:?}"
    );

    // Inspect the atoms that blow up on step 1 (backbone serials).
    let target: std::collections::HashSet<u32> = [40u32, 41, 47, 86, 87, 143, 145, 208, 210, 237, 251].into_iter().collect();
    for a in protein.atoms.iter().filter(|a| target.contains(&a.serial_number)) {
        let res = protein
            .residues
            .iter()
            .find(|r| r.atom_sns.contains(&a.serial_number))
            .map(|r| format!("{:?} #{}", r.res_type, r.serial_number));
        println!(
            "  sn={} elm={} t_in_res={:?} ff={:?} residue={res:?}",
            a.serial_number, a.element, a.type_in_res, a.force_field_type
        );
    }
    // bonds involving the step-1 exploding atoms CZ(40)/NH1(41) of Arg #5
    let mut b: Vec<(u32, u32)> = bonds
        .iter()
        .filter(|b| b.atom_0_sn == 40 || b.atom_1_sn == 40 || b.atom_0_sn == 41 || b.atom_1_sn == 41)
        .map(|b| (b.atom_0_sn, b.atom_1_sn))
        .collect();
    b.sort();
    println!("bonds around CZ(40)/NH1(41): {b:?}");
    // Also: any bonds within Arg #5 that aren't the expected guanidinium ones
    let arg5: std::collections::HashSet<u32> = protein
        .residues
        .iter()
        .filter(|r| r.serial_number == 5)
        .flat_map(|r| r.atom_sns.iter().copied())
        .collect();
    let mut intra: Vec<(u32, u32)> = bonds
        .iter()
        .filter(|b| arg5.contains(&b.atom_0_sn) && arg5.contains(&b.atom_1_sn))
        .map(|b| (b.atom_0_sn, b.atom_1_sn))
        .collect();
    intra.sort();
    println!("Arg#5 intra bonds: {intra:?}");

    // Inspect the persistent high-force hotspot atoms (indices 86, 87, 200, 201, 825, 1189,
    // 1205, 1703) — nearest-neighbor distances. State atoms[i] ↔ prepared serial i+1.
    {
        let hotspot = [86usize, 87, 200, 201, 825, 1189, 1205, 1703];
        let all: Vec<(u32, &bio_files::AtomGeneric)> =
            protein.atoms.iter().map(|a| (a.serial_number, a)).collect();
        for &idx in &hotspot {
            let sn = idx as u32 + 1;
            let Some((_, a)) = all.iter().find(|(s, _)| *s == sn) else {
                continue;
            };
            let (px, py, pz) = (a.posit.x, a.posit.y, a.posit.z);
            let res = protein
                .residues
                .iter()
                .find(|r| r.atom_sns.contains(&sn))
                .map(|r| format!("{:?} #{}", r.res_type, r.serial_number));
            let mut near: Vec<(u32, &str, f64)> = protein
                .atoms
                .iter()
                .filter(|b| b.serial_number != sn)
                .map(|b| {
                    let d = ((b.posit.x - px).powi(2) + (b.posit.y - py).powi(2) + (b.posit.z - pz).powi(2)).sqrt();
                    (b.serial_number, b.type_in_res_general.as_deref().unwrap_or("?"), d)
                })
                .filter(|(_, _, d)| *d < 2.0)
                .collect();
            near.sort_by(|x, y| x.2.partial_cmp(&y.2).unwrap());
            let near_s: Vec<String> = near
                .iter()
                .map(|(s, n, d)| format!("{n}({s})={d:.2}"))
                .collect();
            println!(
                "HOT i={idx} sn={sn} name={} ff={:?} res={res:?}  near(<2Å): {}",
                a.type_in_res_general.as_deref().unwrap_or("?"), a.force_field_type, near_s.join(", ")
            );
        }
    }

    // ── Energy / acceleration convergence diagnostic ────────────────────────
    // NOTE: must build from a FRESH raw MmCif — build_system prepares it once.
    // Reusing the already-prepared `protein` double-prepares and corrupts the system.
    let opts = spice_engine::builder::BuildOptions {
        relax_iters: Some(2_000),
        energy_minimization_tolerance: 2.0,
        ..Default::default()
    };
    let dev = ComputationDevice::default();
    let raw = MmCif::load(Path::new("data/test/2LYZ.cif")).unwrap();
    let mut engine = spice_engine::builder::build_system(&dev, &param_set, raw, &opts).unwrap();
    {
        let st = &engine.state;
        // Verify partial charges (internal = q_e * 18.2223; C≈0.60e, O≈-0.57e).
        let scaler = 18.2223f32;
        for i in [0usize, 1, 2, 3, 4, 5] {
            let a = &st.atoms[i];
            println!(
                "  atom {i}: ff={} elm={:?} q_e={:.4}",
                a.force_field_type, a.element, a.partial_charge / scaler
            );
        }
        let n = st.atoms.len();
        let (mut max_f, mut sum_sq, mut n_moved) = (0.0f64, 0.0f64, 0usize);
        let mut worst_i = 0usize;
        for (i, a) in st.atoms.iter().enumerate() {
            if a.static_ {
                continue;
            }
            let m = ((a.force.x as f64).powi(2) + (a.force.y as f64).powi(2) + (a.force.z as f64).powi(2)).sqrt();
            n_moved += 1;
            sum_sq += m * m;
            if m > max_f {
                max_f = m;
                worst_i = i;
            }
        }
        let rms = (sum_sq / n_moved as f64).sqrt();
        let w = &st.atoms[worst_i];
        println!(
            "\nPOST-MIN: U={:.1} kcal/mol  max|force|={:.3} kcal/mol/Å  rms={:.3}  (worst i={worst_i} ff={} elm={:?})",
            st.potential_energy, max_f, rms, w.force_field_type, w.element
        );
        println!(
            "  n_atoms={n} n_waters={}  U_bonded={:.1} U_nonbonded={:.1}",
            st.water.len(), st.potential_energy_bonded, st.potential_energy_nonbonded
        );
    }

    // Run 6 production steps at 310K.
    println!("\n[production 310K] running 6 steps");
    for _ in 0..6 {
        let r = engine.step(None);
        println!("  step {}: U={:.1} crashed={}", r.step_count, r.u_t_kcal, r.crashed);
        if r.crashed {
            break;
        }
    }
    // bonds involving 992 (the NH2)
    let mut b992: Vec<(u32, u32)> = bonds
        .iter()
        .filter(|b| b.atom_0_sn == 992 || b.atom_1_sn == 992)
        .map(|b| (b.atom_0_sn, b.atom_1_sn))
        .collect();
    println!("  bonds involving NH2(992): {b992:?}");
    for a in protein.atoms.iter().filter(|a| {
        matches!(a.type_in_res, Some(AtomTypeInRes::CA))
            && (200..235).contains(&a.serial_number)
    }) {
        println!(
            "  sn={} t_in_res={:?} ff={:?}",
            a.serial_number, a.type_in_res, a.force_field_type
        );
    }
}
