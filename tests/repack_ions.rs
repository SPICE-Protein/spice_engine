//! v1.3.8 OPTIONAL ion repack on the solvent-reuse path (mutate `repack_ions=True`).
//!
//! Proves the CONTRACT the feature advertises: with a changed salt request the
//! repacked mutant carries the SAME ion counts and effective ionic strength as
//! a FRESH build at that salt (integer rounding identical), the box stays
//! neutral, no site hard-clashes, and a short NVT run is stable - all at reuse
//! cost, NOT a cold rebuild. Positions legitimately differ from the fresh
//! build (accepted density drift): this test never compares coordinates.
//!
//! Run: `cargo test --release --test repack_ions -- --ignored --nocapture --test-threads=1`

use bio_files::MmCif;
use spice_engine::engine::md_core::ComputationDevice;
use spice_engine::engine::md_core::params::FfParamSet;
use spice_engine::env::EnvParams;
use spice_engine::structure::{AtomInput, StructureInput};
use spice_engine::{BuildOptions, SpiceEngine, build_mutant_by_solvent_reuse, build_system};
use std::path::Path;
use std::time::Instant;

const IONS: [&str; 7] = ["Na+", "K+", "Cl-", "Mg2+", "Ca2+", "Sr2+", "Ba2+"];

fn ion_counts(e: &SpiceEngine) -> (usize, usize) {
    let mut na = 0;
    let mut cl = 0;
    for a in &e.state.atoms {
        match a.force_field_type.as_str() {
            "Na+" => na += 1,
            "Cl-" => cl += 1,
            _ => {}
        }
    }
    (na, cl)
}

fn wt_input_from_cif(mm: &MmCif) -> StructureInput {
    let mut input = StructureInput::default();
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
                occupancy: a.occupancy.unwrap_or(1.0),
            });
        }
    }
    input
}

fn build_2lyz(ionic: f32) -> SpiceEngine {
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("amber");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
    let opts = BuildOptions {
        env: EnvParams::new(7.0, 300.0, 0.0, ionic),
        equil: None,
        ..Default::default()
    };
    build_system(&dev, &params, protein, &opts).expect("build")
}

#[test]
#[ignore = "expensive: two cold 2LYZ builds + one repack reuse + 50 MD steps"]
fn repack_matches_fresh_salt_counts_and_is_stable() {
    let parent = build_2lyz(0.15);
    let fresh = build_2lyz(0.30);
    let (pna, pcl) = ion_counts(&parent);
    let (fna, fcl) = ion_counts(&fresh);
    assert!(fna > pna && fcl > pcl, "sanity: 0.30 M > 0.15 M counts");
    assert!((fresh.state.net_charge_e()).abs() < 0.05);

    let wt = {
        let mm = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
        wt_input_from_cif(&mm)
    };
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("amber");
    let opts = BuildOptions {
        env: EnvParams::new(7.0, 300.0, 0.0, 0.30),
        equil: None,
        relax_iters: Some(600),
        repack_ions: true,
        ..Default::default()
    };
    let t0 = Instant::now();
    let mut child =
        build_mutant_by_solvent_reuse(&parent, &params, &wt, &opts).expect("repack reuse");
    let dt = t0.elapsed();
    println!("[repack] WT->WT 0.15->0.30 M reuse: {dt:?}");

    // Contract 1: ion counts and effective I identical to a fresh build.
    let (cna, ccl) = ion_counts(&child);
    assert_eq!(cna, fna, "Na+ count repack({cna}) != fresh({fna})");
    assert_eq!(ccl, fcl, "Cl- count repack({ccl}) != fresh({fcl})");
    let di =
        (child.state.effective_ionic_strength_m() - fresh.state.effective_ionic_strength_m()).abs();
    assert!(di < 1e-3, "effective I mismatch {di} M");
    assert!(
        (child.state.net_charge_e()).abs() < 0.05,
        "repacked box not neutral"
    );

    // Contract 2: solvent box unchanged in SHAPE (same cell) but the ion
    // insertion displaced waters (never healed back - the accepted drift).
    assert_eq!(child.state.cell.extent, parent.state.cell.extent);
    assert!(
        child.state.water.len() <= parent.state.water.len(),
        "repack must not add waters"
    );

    // Contract 3: no ion sits hard-clashing inside another site.
    let mut min_d = f64::MAX;
    for (i, a) in child.state.atoms.iter().enumerate() {
        if !IONS.contains(&a.force_field_type.as_str()) {
            continue;
        }
        let ext = child.state.cell.extent;
        let (lx, ly, lz) = (ext.x as f64, ext.y as f64, ext.z as f64);
        let mut near = |j: Option<usize>, pj: [f64; 3]| {
            if let Some(jj) = j {
                if i == jj {
                    return;
                }
            }
            let mut dx = a.posit.x as f64 - pj[0];
            let mut dy = a.posit.y as f64 - pj[1];
            let mut dz = a.posit.z as f64 - pj[2];
            dx -= lx * (dx / lx).round();
            dy -= ly * (dy / ly).round();
            dz -= lz * (dz / lz).round();
            min_d = min_d.min((dx * dx + dy * dy + dz * dz).sqrt());
        };
        for (j, b) in child.state.atoms.iter().enumerate() {
            if b.element != na_seq::Element::Hydrogen {
                near(
                    Some(j),
                    [b.posit.x as f64, b.posit.y as f64, b.posit.z as f64],
                );
            }
        }
        for w in &child.state.water {
            near(
                None,
                [w.o.posit.x as f64, w.o.posit.y as f64, w.o.posit.z as f64],
            );
            near(
                None,
                [
                    w.h0.posit.x as f64,
                    w.h0.posit.y as f64,
                    w.h0.posit.z as f64,
                ],
            );
            near(
                None,
                [
                    w.h1.posit.x as f64,
                    w.h1.posit.y as f64,
                    w.h1.posit.z as f64,
                ],
            );
        }
    }
    assert!(min_d > 1.5, "ion hard clash at {min_d:.2} A after repack");
    println!("[repack] min ion-site distance {min_d:.2} A");

    // Contract 4: short NVT is stable (no detonation from the drift).
    for _ in 0..50 {
        let r = child.step(None);
        assert!(!r.crashed, "repacked box crashed: {:?}", r.crash_reason);
        assert!(
            r.u_t_kcal.is_finite() && r.u_t_kcal.abs() < 1e7,
            "U {}",
            r.u_t_kcal
        );
    }
    println!(
        "[repack] U after 50 steps {:.0} kcal/mol; I_eff {:.4} M (fresh {:.4})",
        child.state.potential_energy,
        child.state.effective_ionic_strength_m(),
        fresh.state.effective_ionic_strength_m(),
    );
}
