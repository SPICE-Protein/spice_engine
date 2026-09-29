//! Bit-identity guard for the v1.3.2 "insertable species" refactor.
//!
//! Post-relaxation coordinates are deliberately NOT reproducible across
//! builds (thermo/baro noise streams reseed per build), but everything the
//! *insertion layer* decides is deterministic before any dynamics runs:
//! the seeded water slot choice, the append order of ion/cosolute atoms,
//! their per-atom attributes (charge, LJ, C4), molecule grouping, and the
//! distance-restraint indices urea's bonds create. This test snapshots
//! exactly that stable tail — so a migration step that changes *what gets
//! inserted where* fails even though coordinates may legitimately drift.
//!
//! Baseline lives in `tests/ion_golden.txt` (captured on v1.3.1, pre-refactor).
//! REGENERATION HISTORY (why the current header reads what it reads): the
//! v1.3.6 re-capture, after the Rough-1 charge fixes, moved the header from
//! `atoms=2013 water=5867 solute=1915 mols=57 restraints=42` (v1.3.1 era) to
//! `atoms=2061 water=5861 solute=1960 mols=60 restraints=42` — +45 solute
//! hydrogens restored (the clash-deleted and count-guessed H of Rough-1; see
//! tests/charge_probe.rs sweep), and Cl⁻ 29→32 because neutralization now
//! targets the TRUE integer net charge (+8, not the leaky +4.8). Water count
//! falls exactly as ion count rises: insertion displaces, never adds.
//! Run:
//! `GOLDEN_UPDATE=1 cargo test --release --test ion_layout_golden -- --ignored --nocapture --test-threads=1`
//! regenerates it (review the diff!). Plain run compares and must stay green.

use bio_files::MmCif;
use spice_engine::engine::md_core::ComputationDevice;
use spice_engine::engine::md_core::cosolvent_presets::cosolvent_preset;
use spice_engine::engine::md_core::params::FfParamSet;
use spice_engine::env::EnvParams;
use spice_engine::{BuildOptions, build_system};
use std::path::Path;

const GOLDEN: &str = "tests/ion_golden.txt";

fn snapshot() -> String {
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("load Amber parameters");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
    let env = EnvParams::new(7.0, 300.0, 1.0, 0.15).with_divalent(0.025, 0.0, 0.0, 0.0);
    let mut urea = cosolvent_preset("UREA").expect("urea preset");
    urea.molarity = 0.05;
    let opts = BuildOptions {
        env,
        cosolvents: vec![urea],
        ..Default::default()
    };
    let engine = build_system(&dev, &params, protein, &opts).expect("build engine");
    let st = &engine.state;

    let mut s = String::new();
    s.push_str(&format!(
        "atoms={} water={} solute={} mols={} restraints={}\n",
        st.atoms.len(),
        st.water.len(),
        st.solute_atom_count,
        st.mol_start_indices.len(),
        st.distance_restraints.len(),
    ));
    for a in &st.atoms[st.solute_atom_count..] {
        s.push_str(&format!(
            "TAIL {}|{}|{}|{}|{}|{}\n",
            a.serial_number,
            a.force_field_type,
            a.partial_charge.to_bits(),
            a.lj_sigma.to_bits(),
            a.lj_eps.to_bits(),
            a.lj_c4.to_bits(),
        ));
    }
    s.push_str(&format!("MOLS {:?}\n", st.mol_start_indices));
    for r in &st.distance_restraints {
        s.push_str(&format!(
            "REST {}|{}|{}|{}\n",
            r.atom_0_idx,
            r.atom_1_idx,
            r.r0.to_bits(),
            r.k.to_bits(),
        ));
    }
    s
}

/// Diagnostic companion: build the same system and identify the atoms the
/// solvent-init MD reported as hitting the accel clamp (indices are stable
/// post-build — no reordering), with their nearest-foreign contacts.
#[test]
#[ignore = "identify hot atoms after the build-time solvent MD"]
fn identify_clamped_atoms() {
    let idxs: Vec<usize> = vec![2047, 2052];
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("params");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load");
    let env = EnvParams::new(7.0, 300.0, 1.0, 0.15).with_divalent(0.025, 0.0, 0.0, 0.0);
    let mut urea = cosolvent_preset("UREA").expect("urea preset");
    urea.molarity = 0.05;
    let opts = BuildOptions {
        env,
        cosolvents: vec![urea],
        ..Default::default()
    };
    let engine = build_system(&dev, &params, protein, &opts).expect("build");
    let st = &engine.state;
    let n = st.atoms.len();
    println!("solute_atom_count={}", st.solute_atom_count);
    // Full contents of the molecule containing 2047.
    for (i, a) in st.atoms.iter().enumerate().take(2053).skip(2045) {
        println!(
            "  mol-at {i}: serial={} ff={} name={:?} q_e={:.3}",
            a.serial_number,
            a.force_field_type,
            a.element,
            f64::from(a.partial_charge) / 18.2223
        );
    }
    for &i in &idxs {
        if i >= n {
            println!("atom {i}: out of range (n={n})");
            continue;
        }
        let a = &st.atoms[i];
        // nearest contact excluding own molecule (via mol_start_indices)
        let mol_start = st
            .mol_start_indices
            .iter()
            .copied()
            .filter(|&m| m <= i)
            .max()
            .unwrap_or(0);
        let mol_end = *st
            .mol_start_indices
            .iter()
            .filter(|&&m| m > mol_start)
            .min()
            .unwrap_or(&n);
        let mut best = (f32::INFINITY, usize::MAX);
        for (j, b) in st.atoms.iter().enumerate() {
            if j == i || (j >= mol_start && j < mol_end) {
                continue;
            }
            let d = st.cell.min_image(b.posit - a.posit).magnitude();
            if d < best.0 {
                best = (d, j);
            }
        }
        let bj = &st.atoms[best.1];
        println!(
            "atom {i}: serial={} ff={} q={:.3} posit=({:.2},{:.2},{:.2}) \
             mol=[{mol_start},{mol_end}) | nearest foreign {}: {}: {:.3} A",
            a.serial_number,
            a.force_field_type,
            a.partial_charge,
            a.posit.x,
            a.posit.y,
            a.posit.z,
            best.1,
            bj.force_field_type,
            best.0
        );
    }
}

#[test]
#[ignore = "full 2LYZ build with salt + divalent + urea"]
fn ion_layout_golden() {
    let got = snapshot();
    if std::env::var("GOLDEN_UPDATE").is_ok() {
        std::fs::write(GOLDEN, &got).expect("write golden");
        println!("golden baseline written to {GOLDEN} — review the diff!");
        return;
    }
    let want = std::fs::read_to_string(GOLDEN)
        .unwrap_or_else(|_| panic!("missing {GOLDEN}: run with GOLDEN_UPDATE=1"));
    assert_eq!(want.trim_end(), got.trim_end(), "insertion layout drifted");
}
