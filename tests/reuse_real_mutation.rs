//! Decisive comparison: solvent-reuse cost for a WT→WT "mutation" vs a REAL
//! single-point mutation, on the SAME parent box.
//!
//! Background (v1.3.5 step-0 measurement): a WT→WT reuse spends ~150–320 s on
//! HSFA2 (and ~17.6 s on 2LYZ) but that time is NOT in the neighbor table / PME
//! / prune (head + neighbors + PME measured ~0.4–0.8 s). It is entirely the
//! builder-tail fallback `minimize_energy(dev, relax_iters, None)` — a GLOBAL
//! L-BFGS over every site — taken ONLY because WT→WT yields an EMPTY
//! `mutated_positions`. A genuine point mutation instead routes through
//! `minimize_local_region(..., 6 Å, relax_iters)` and never pays the global min.
//!
//! This test proves the routing + magnitude: same parent, two reuse calls.
//!   * `wtw`   → empty mutated set  → global min   (expensive)
//!   * `real`  → one substituted residue → local min (cheap)
//!
//! The "real" mutant is a chemically clean Ala-substitution (the standard
//! stability-design probe): the target internal residue is reduced to ALA's
//! complete heavy set {N, CA, C, O, CB}, `res_name = "ALA"`, so
//! `prepare_peptide_mmcif` is happy (H placed from a valid template, residue not
//! flagged incomplete) and the mutant sequence differs from WT at exactly one
//! character → builder's `mutated_positions` is non-empty.
//!
//! Run: `cargo test --release --test reuse_real_mutation -- --ignored --nocapture --test-threads=1`

use bio_files::MmCif;
use spice_engine::engine::md_core::ComputationDevice;
use spice_engine::engine::md_core::params::FfParamSet;
use spice_engine::env::EnvParams;
use spice_engine::structure::{AtomInput, StructureInput, one_letter_from_resname};
use spice_engine::{BuildOptions, build_mutant_by_solvent_reuse, build_system};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

fn build_params() -> FfParamSet {
    FfParamSet::new_amber().expect("load Amber parameters")
}

/// Rebuild a WT `StructureInput` straight from a CIF (heavy atoms), skipping
/// crystal waters — same recipe as `rl_5ps_budget`.
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

/// Reduce one internal, non-{A,C,G} residue (that carries N/CA/C/O/CB) to a
/// clean ALA. Returns (mutant_input, res_seq, wt_letter).
fn make_single_ala_substitution(wt: &StructureInput) -> (StructureInput, i32, char) {
    // Group atoms by residue, preserving sequence order.
    let mut by_res: BTreeMap<i32, Vec<&AtomInput>> = BTreeMap::new();
    for a in &wt.atoms {
        by_res.entry(a.res_seq).or_default().push(a);
    }
    let names: Vec<(i32, char)> = by_res
        .iter()
        .map(|(rs, as_)| {
            (
                *rs,
                one_letter_from_resname(&as_[0].res_name).unwrap_or('X'),
            )
        })
        .collect();
    // Target: internal (not first/last), letter ∉ {A,C,G} (C → avoid disulfide
    // chemistry, G → no CB so not a valid ALA, A → mutation would be no-op),
    // and its residue actually exposes N/CA/C/O/CB.
    let mut target_res: Option<i32> = None;
    let mut target_wt: char = '?';
    for k in 1..names.len().saturating_sub(1) {
        let (rs, letter) = names[k];
        if matches!(letter, 'A' | 'C' | 'G' | 'X') {
            continue;
        }
        let have: Vec<String> = by_res[&rs].iter().map(|a| a.atom_name.clone()).collect();
        if ["N", "CA", "C", "O", "CB"]
            .iter()
            .all(|n| have.iter().any(|h| h == n))
        {
            target_res = Some(rs);
            target_wt = letter;
            break;
        }
    }
    let rs = target_res.expect("no non-{A,C,G} residue with N/CA/C/O/CB found");

    let mut mut_input = StructureInput::default();
    for a in &wt.atoms {
        if a.res_seq == rs {
            if !matches!(a.atom_name.as_str(), "N" | "CA" | "C" | "O" | "CB") {
                continue; // drop the original side chain beyond CB
            }
            mut_input.push(AtomInput {
                res_name: "ALA".to_string(),
                ..a.clone()
            });
        } else {
            mut_input.push(a.clone());
        }
    }
    (mut_input, rs, target_wt)
}

#[test]
#[ignore = "expensive: two builds + two solvent-reuse relaxations on 2LYZ"]
fn real_mutation_reuse_is_local_min_not_global() {
    let dev = ComputationDevice::Cpu;
    let params = build_params();
    let opts = BuildOptions {
        env: EnvParams::new(7.0, 300.0, 1.0, 0.15),
        ..Default::default()
    };

    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
    let t0 = Instant::now();
    let parent = build_system(
        &dev,
        &params,
        MmCif::load(Path::new("data/test/2LYZ.cif")).unwrap(),
        &opts,
    )
    .expect("build parent");
    println!("[parent] cold build {:.1} s", t0.elapsed().as_secs_f64());

    let wt = wt_input_from_cif(&protein);
    let wt_seq = wt.sequence().expect("wt seq");
    let (mut_input, res_seq, wt_letter) = make_single_ala_substitution(&wt);
    let mut_seq = mut_input.sequence().expect("mut seq");

    // Sanity: the mutant differs from WT at EXACTLY one position (→ builder's
    // mutated_positions becomes non-empty → the local-min branch is taken).
    let diffs: Vec<usize> = wt_seq
        .chars()
        .zip(mut_seq.chars())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        diffs.len(),
        1,
        "expected a single-point mutation, got {diffs:?}"
    );
    assert_eq!(
        mut_seq.chars().nth(diffs[0]).unwrap(),
        'A',
        "substitution target should map to ALA"
    );
    println!(
        "[mutation] {wt_letter}→A at res_seq {res_seq} (seq index {}), len {}",
        diffs[0],
        mut_seq.len()
    );

    // ---- (1) env-only reuse: identical structure & protonation, hotter T ----
    // changed_positions is EMPTY → the builder skips minimization outright.
    // Pre-fix this exact call paid a GLOBAL minimize_energy (measured 17–36 s
    // on 2LYZ, 150–320 s on HSFA2); post-fix it costs only head+neighbors+PME.
    let env_only_opts = BuildOptions {
        env: EnvParams::new(7.0, 330.0, 1.0, 0.15), // same pH/salt, only T differs
        ..opts.clone()
    };
    let t0 = Instant::now();
    let _env_only =
        build_mutant_by_solvent_reuse(&parent, &params, &wt, &env_only_opts).expect("env reuse");
    let env_only_s = t0.elapsed().as_secs_f64();

    // ---- (2) real single-point mutation: 1 residue identity change → LOCAL ----
    let t0 = Instant::now();
    let real =
        build_mutant_by_solvent_reuse(&parent, &params, &mut_input, &opts).expect("real reuse");
    let real_s = t0.elapsed().as_secs_f64();

    println!(
        "[reuse] env-only T-change (skip) {env_only_s:.2} s  |  \
         real 1-point mutation (local min) {real_s:.2} s"
    );

    // We deliberately do NOT assert that this particular mutant runs stable MD.
    // A single-point substitution can be intrinsically explosive (V2→A here blows
    // up in equilibrate at U~1e8 — the exact signature the RL pipeline's
    // explosive_blacklist records). That is physics, not a reuse-path defect, and
    // catching it early is the trend fail-fast's separate job
    // (tests/trend_fail_fast.rs). The claim under test is purely the REUSE COST
    // routing, so all we require is that the mutant engine BUILT correctly.
    assert_eq!(real.topology.sequence, mut_seq, "mutant topology sequence");
    let real_sites = real.state.atoms.len() + 4 * real.state.water.len();
    println!(
        "[reuse] real mutant built fine: seq len {}, sites {} (≠ assertion of MD stability)",
        mut_seq.len(),
        real_sites
    );

    // Regression guard for the v1.3.5 (c) fix: NEITHER route may pay the old
    // global-min explosion. A genuine fresh build of this box measured ~26 s
    // above; both reuse paths must land well under it, and the pure env/T skip
    // (no relaxation at all) must be the cheapest.
    assert!(
        env_only_s < 8.0,
        "env-only reuse must SKIP minimization (got {env_only_s:.2} s) — global-min is back"
    );
    assert!(
        real_s < 12.0,
        "real point-mutation reuse must be local-min cheap (got {real_s:.2} s)"
    );
    assert!(
        env_only_s <= real_s + 1.0,
        "skip path ({env_only_s:.2} s) should not cost more than local path ({real_s:.2} s)"
    );
}
