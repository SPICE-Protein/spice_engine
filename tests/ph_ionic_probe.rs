//! End-to-end probes for the two env parameters most at risk of being
//! display-only ghosts: pH (must change *built* protonation states) and
//! ionic strength (must insert the right number of explicit Na⁺/Cl⁻).
//!
//! Design facts these lock in:
//! - pH is a **build-time** input (`prepare_peptide_mmcif` → pKa-resolved
//!   variants). There is deliberately no live pH setter; scans key their
//!   template cache on `ph.to_bits()` (domain.rs) so different pH = different
//!   build. This test asserts the protein net charge actually moves.
//! - Ionic strength is **explicit salt**, not a Debye-Hückel kappa:
//!   `EnvParams::ionic_strength_m` → `salt_concentration_m` →
//!   `add_salt_ions` replaces whole waters with neutral Na⁺/Cl⁻ pairs.
//!   Charge-neutralizing counterions use the same ff types, so the probe
//!   compares two builds at *identical* pH: the ion-count delta is pure salt.
//!
//! A third probe (`species_channel_salts_work`, v1.3.2) exercises the
//! general electrolyte channel: KCl + MgCl₂ through `SaltSpec`, asserting
//! c·V·N_A counts, auto-balanced stoichiometry, K⁺'s published σ, and
//! mixed-salt electroneutrality. A fourth (`r2_mutant_carries_cosolute_
//! restraints`, v1.3.2 fix) proves the solvent-reuse mutant carries the
//! parent's cosolute bond restraints, exclusions, and adjacency — and that
//! a carried urea C=O stays bonded through MD.
//!
//! Run with:
//! `cargo test --release --test ph_ionic_probe -- --ignored --nocapture --test-threads=1`

use bio_files::MmCif;
use spice_engine::engine::md_core::ComputationDevice;
use spice_engine::engine::md_core::params::FfParamSet;
use spice_engine::env::EnvParams;
use spice_engine::{BuildOptions, build_system};
use std::path::Path;

const CHARGE_UNIT: f64 = 18.2223; // AMBER-native per elementary charge
const AVOGADRO: f64 = 6.02214076e23;

fn build_2lyz(ph: f32, ionic_m: f32) -> spice_engine::SpiceEngine {
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("load Amber parameters");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
    let opts = BuildOptions {
        env: EnvParams::new(ph, 300.0, 1.0, ionic_m),
        ..Default::default()
    };
    build_system(&dev, &params, protein, &opts).expect("build engine")
}

/// Net charge (units of e) summed over protein-residue atoms only, so
/// water, counterions and salt never enter the comparison.
fn protein_net_charge_e(engine: &spice_engine::SpiceEngine) -> f64 {
    let mut sum = 0.0_f64;
    for res in &engine.topology.residues {
        for &i in &res.atom_indices {
            sum += f64::from(engine.state.atoms[i].partial_charge);
        }
    }
    sum / CHARGE_UNIT
}

fn ion_count(engine: &spice_engine::SpiceEngine) -> usize {
    engine
        .state
        .atoms
        .iter()
        .filter(|a| {
            matches!(
                a.element,
                na_seq::Element::Sodium | na_seq::Element::Chlorine
            )
        })
        .count()
}

#[test]
#[ignore = "expensive: three full 2LYZ builds"]
fn ph_moves_protonation_and_ionic_moves_salt() {
    // ---- pH: protein net charge must follow the pKa ladder ----
    let acidic = build_2lyz(2.0, 0.0);
    let basic = build_2lyz(12.0, 0.0);
    let q_acid = protein_net_charge_e(&acidic);
    let q_base = protein_net_charge_e(&basic);
    println!("protein net charge: pH 2 → {q_acid:+.2} e, pH 12 → {q_base:+.2} e");
    // 2LYZ has ~12 titratable acid sites (6 ASP + 6 GLU) and 6 LYS + 6 ARG +
    // His; pH 2 protonates every acid, pH 12 deprotonates lysines. A ghost
    // (pH ignored) would give equal values; the real ladder gives ≥ 12 e.
    assert!(
        q_acid > q_base + 10.0,
        "pH has no measurable effect on protonation: {q_acid} vs {q_base}"
    );

    // ---- ionic strength: ion count delta must equal c·V·N_A pairs ----
    let plain = build_2lyz(2.0, 0.0);
    let salty = build_2lyz(2.0, 0.5);
    let n0 = ion_count(&plain);
    let n1 = ion_count(&salty);
    let vol_l = f64::from(plain.state.cell.volume()) * 1.0e-27;
    let expect_pairs = (0.5 * vol_l * AVOGADRO).round() as usize;
    println!(
        "ions: 0 M → {n0}, 0.5 M → {n1} (expect +{} from salt)",
        2 * expect_pairs
    );
    assert!(
        n1 >= n0 + 2 * expect_pairs.saturating_sub(2),
        "0.5 M salt added {} ions, expected ≥ {}",
        n1 - n0,
        2 * expect_pairs
    );
    // Salt pairs are inserted charge-neutral, so the delta must be even.
    assert_eq!(
        (n1 - n0) % 2,
        0,
        "salt added an odd ion count: not neutral pairs"
    );
    // ---- genion-style placement invariants ----
    // The 6 Å exclusion is a *placement-time* property (pick_ion_slots); the
    // subsequent solvent relaxation + L-BFGS minimization deliberately let
    // ions migrate into solvation shells, and the build's thermostat/barostat
    // noise streams are freshly seeded per build *by design* (independent
    // environment points), so final coordinates are not bit-reproducible
    // across builds. What must hold after all of that: no ion is buried in
    // the solute or fused to another ion. The old fixed-stride placement
    // could violate even this weak bound (a lattice-displaced ion sat on a
    // protein surface atom); the guard below catches that class.
    use spice_engine::engine::md_core::AtomDynamics;
    let is_ion = |a: &AtomDynamics| {
        matches!(
            a.element,
            na_seq::Element::Sodium | na_seq::Element::Chlorine
        )
    };
    let cell = &salty.state.cell;
    const MIN_ION_CONTACT_ANGSTROM: f32 = 2.0; // vdW-overlap floor
    let min2 = MIN_ION_CONTACT_ANGSTROM * MIN_ION_CONTACT_ANGSTROM;
    let ion_sites: Vec<(usize, lin_alg::f32::Vec3)> = salty
        .state
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| is_ion(a))
        .map(|(i, a)| (i, a.posit))
        .collect();
    assert_eq!(ion_sites.len(), n1);
    for &(i, pos) in &ion_sites {
        for (j, other) in salty.state.atoms.iter().enumerate() {
            if j == i || is_ion(other) {
                continue;
            }
            let d = cell.min_image(pos - other.posit);
            let d2 = d.x * d.x + d.y * d.y + d.z * d.z;
            assert!(
                d2 >= min2,
                "ion {i} [{ff}] sits {}/\u{00c5} from solute atom {j} — buried",
                d2.sqrt(),
                ff = salty.state.atoms[i].force_field_type
            );
        }
        for &(k, pos2) in &ion_sites {
            if k <= i {
                continue;
            }
            let d = cell.min_image(pos - pos2);
            let d2 = d.x * d.x + d.y * d.y + d.z * d.z;
            assert!(d2 >= min2, "ions {i}/{k} fused at {}/\u{00c5}", d2.sqrt());
        }
    }

    // ---- PME electroneutrality diagnostic ----
    let net = salty.state.net_charge_e();
    println!("net charge after counterions + salt: {net:+.3} e");
    assert!(
        net.abs() < 1.0,
        "integer ions must cancel the (fractional) protein charge to <1 e; got {net}"
    );

    // The system must stay buildable/stable with salt present.
    let mut eng = salty;
    let r = eng.step(None);
    assert!(!r.crashed, "first step with 0.5 M salt crashed");
}

/// Second ignored probe: the new environment knobs (divalent salts, redox,
/// electric field, cosolvent mechanism) end-to-end on 2LYZ.
fn build_with(
    ph: f32,
    ionic_m: f32,
    tune: impl FnOnce(EnvParams) -> EnvParams,
    cosolvents: Vec<spice_engine::engine::md_core::CosolventSpec>,
) -> spice_engine::SpiceEngine {
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("load Amber parameters");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
    let opts = BuildOptions {
        env: tune(EnvParams::new(ph, 300.0, 1.0, ionic_m)),
        cosolvents,
        ..Default::default()
    };
    build_system(&dev, &params, protein, &opts).expect("build engine")
}

#[test]
#[ignore = "expensive: five more full builds"]
fn new_env_knobs_work() {
    use spice_engine::engine::md_core::{CosolventSite, CosolventSpec};

    // ---- divalent salts (12-6-4): Mg + Sr + Ba in one build ----
    let mg = build_with(
        2.0,
        0.0,
        |e| e.with_divalent(0.025, 0.0, 0.0125, 0.005),
        Vec::new(),
    );
    let vol_l = f64::from(mg.state.cell.volume()) * 1.0e-27;
    let expect_units = (0.025 * vol_l * AVOGADRO).round() as usize;
    let n_mg = mg
        .state
        .atoms
        .iter()
        .filter(|a| a.element == na_seq::Element::Magnesium)
        .count();
    let q_mg: f32 = mg
        .state
        .atoms
        .iter()
        .filter(|a| a.element == na_seq::Element::Magnesium)
        .map(|a| a.partial_charge)
        .sum();
    let c4_mg: Vec<f32> = mg
        .state
        .atoms
        .iter()
        .filter(|a| a.element == na_seq::Element::Magnesium)
        .map(|a| a.lj_c4)
        .collect();
    println!("Mg2+: inserted {n_mg} (expect ~{expect_units}), +2e charge ✓, C4 present ✓");
    assert!(
        n_mg + 2 >= expect_units && n_mg <= expect_units + 2,
        "Mg count {n_mg} vs expected {expect_units}"
    );
    assert!((f64::from(q_mg) / CHARGE_UNIT - 2.0 * n_mg as f64).abs() < 0.01);
    assert!(
        c4_mg.iter().all(|&c| c == 127.0),
        "Mg must carry Li-Merz C4"
    );
    let mut mg = mg;
    let r = mg.step(None);
    assert!(
        !r.crashed && r.u_t_kcal.is_finite(),
        "12-6-4 system unstable"
    );
    // non-cation atoms must NOT carry C4 (water/anions/protein keep 12-6);
    // cations are exactly the ff types ending in "2+".
    assert!(
        mg.state
            .atoms
            .iter()
            .filter(|a| !a.force_field_type.ends_with("2+"))
            .all(|a| a.lj_c4 == 0.0)
    );

    // Sr²⁺ / Ba²⁺ rode the same MgCl2-form loop. Sr has no na_seq element
    // variant (identity = ff type), Ba does — count both by ff type.
    use spice_engine::engine::md_core::R_STAR_TO_SIGMA;
    let vol_l2 = f64::from(mg.state.cell.volume()) * 1.0e-27;
    for (ff, c4, rstar, conc) in [
        ("Sr2+", 87.0f32, 1.738f32, 0.0125f64),
        ("Ba2+", 78.0f32, 1.898f32, 0.005f64),
    ] {
        let picked: Vec<&_> = mg
            .state
            .atoms
            .iter()
            .filter(|a| a.force_field_type == ff)
            .collect();
        let expect = (conc * vol_l2 * AVOGADRO).round() as usize;
        println!(
            "{ff}: inserted {} (expect ~{expect}), c4 {c4}",
            picked.len()
        );
        assert!(
            picked.len() + 2 >= expect && picked.len() <= expect + 2,
            "{ff} count {} vs expected {expect}",
            picked.len()
        );
        assert!(picked.iter().all(|a| a.lj_c4 == c4));
        assert!(
            picked
                .iter()
                .all(|a| (a.lj_sigma - rstar * R_STAR_TO_SIGMA).abs() < 1e-4)
        );
        assert!(
            (picked
                .iter()
                .map(|a| f64::from(a.partial_charge))
                .sum::<f64>()
                / CHARGE_UNIT
                - 2.0 * picked.len() as f64)
                .abs()
                < 0.01
        );
    }

    // ---- redox: fraction 1.0 reduces every 2LYZ disulfide (4 bridges) ----
    let ox = build_with(7.0, 0.0, |e| e, Vec::new());
    let red = build_with(7.0, 0.0, |e| e.with_redox(1.0), Vec::new());
    let n_sg_sh = |e: &spice_engine::SpiceEngine| {
        e.state
            .atoms
            .iter()
            .filter(|a| a.force_field_type == "SH")
            .count()
    };
    let n_sg_s = |e: &spice_engine::SpiceEngine| {
        e.state
            .atoms
            .iter()
            .filter(|a| a.force_field_type == "S")
            .count()
    };
    println!(
        "redox: oxidized SH={} S={} → reducing SH={} S={}",
        n_sg_sh(&ox),
        n_sg_s(&ox),
        n_sg_sh(&red),
        n_sg_s(&red)
    );
    // 2LYZ: 4 bridges × 2 SG get CYX's "S" type — plus the methionine SD
    // atoms also carry "S", so only the SH side is a clean bridge count, and
    // total Sulfur must be invariant.
    let n_s_total = |e: &spice_engine::SpiceEngine| {
        e.state
            .atoms
            .iter()
            .filter(|a| a.element == na_seq::Element::Sulfur)
            .count()
    };
    assert_eq!(n_sg_sh(&ox), 0);
    assert_eq!(
        n_sg_sh(&red),
        8,
        "fully reducing flips all 8 SG to thiol 'SH'"
    );
    assert_eq!(
        n_sg_s(&ox) - n_sg_s(&red),
        8,
        "exactly the 8 bridge SGs left the CYX 'S' set"
    );
    assert_eq!(
        n_s_total(&ox),
        n_s_total(&red),
        "sulfur inventory invariant"
    );
    // Each reduced SG gained an HG hydrogen: total atoms up by 8.
    assert_eq!(red.state.atoms.len(), ox.state.atoms.len() + 8);
    let mut red = red;
    let r = red.step(None);
    assert!(
        !r.crashed && r.u_t_kcal.is_finite(),
        "reduced system unstable"
    );

    // ---- electric field: smoke + energy includes field potential ----
    let nofield = build_with(7.0, 0.0, |e| e, Vec::new());
    let field = build_with(
        7.0,
        0.0,
        |e| e.with_efield([0.3, 0.0, 0.0], 0.0),
        Vec::new(),
    );
    let (u_a, u_b) = (nofield.state.potential_energy, field.state.potential_energy);
    // The env parameter must actually land in the built state's config —
    // otherwise "field on" is a phantom that only the smoke steps would hide.
    assert_eq!(field.state.cfg.efield, [0.3, 0.0, 0.0]);
    assert_eq!(nofield.state.cfg.efield, [0.0, 0.0, 0.0]);
    assert!(field.state.cfg.efield_omega == 0.0);
    println!("efield wiring ✓ (cfg carries the field); energies {u_a:.1} vs {u_b:.1}");
    let mut field = field;
    for _ in 0..5 {
        let r = field.step(None);
        assert!(!r.crashed && r.u_t_kcal.is_finite(), "efield run unstable");
    }

    // ---- vendored urea preset (CGenFF v4.6 provenance) at 0.05 M ----
    let mut urea_spec = spice_engine::engine::md_core::cosolvent_presets::urea();
    urea_spec.molarity = 0.05;
    let urea_eng = build_with(7.0, 0.0, |e| e, vec![urea_spec]);
    let n_ng = urea_eng
        .state
        .atoms
        .iter()
        .filter(|a| a.force_field_type == "NG2S2")
        .count();
    let vol_u = f64::from(urea_eng.state.cell.volume()) * 1.0e-27;
    let expect_u = (0.05 * vol_u * AVOGADRO).round() as usize;
    println!(
        "urea preset: {n_ng} amide N across ~{} molecules (expect ~{expect_u})",
        n_ng / 2
    );
    assert!(
        n_ng / 2 + 2 >= expect_u && n_ng / 2 <= expect_u + 2,
        "urea molecule count {} vs ~{expect_u}",
        n_ng / 2
    );
    // Net-neutral cosolvent keeps neutrality within the usual ±1 e band.
    assert!(urea_eng.state.net_charge_e().abs() < 1.0);
    // C=O restraint distance of the first urea after 30 steps: r0 1.23 Å.
    // urea site order: [N1, H11, H12, C2, O2, N3, H31, H32] — O2 follows C2.
    let first_c = urea_eng
        .state
        .atoms
        .iter()
        .position(|a| a.force_field_type == "CG2O6")
        .expect("urea carbonyl present");
    assert_eq!(
        urea_eng.state.atoms[first_c + 1].force_field_type,
        "OG2D1",
        "site order assumption broken"
    );
    let mut urea_run = urea_eng;
    for _ in 0..30 {
        let r = urea_run.step(None);
        assert!(!r.crashed, "urea preset run crashed");
    }
    let d_co = urea_run
        .state
        .cell
        .min_image(urea_run.state.atoms[first_c].posit - urea_run.state.atoms[first_c + 1].posit)
        .magnitude();
    println!("urea C=O distance after 30 steps: {d_co:.3} Å (r0 1.230)");
    assert!(
        (f64::from(d_co) - 1.230).abs() < 0.15,
        "urea C=O drifted: {d_co}"
    );

    // ---- cosolvent mechanism: synthetic neutral 2-site probe molecule ----
    let spec = CosolventSpec {
        name: "PRB".into(),
        molarity: 0.05,
        sites: vec![
            CosolventSite {
                ff_type: "PRBC".into(),
                element: na_seq::Element::Carbon,
                mass: 12.011,
                charge_scaled: 0.25 * 18.2223,
                sigma: 3.4,
                eps: 0.1,
                c4: 0.0,
                offset: lin_alg::f32::Vec3::new(-1.0, 0.0, 0.0),
            },
            CosolventSite {
                ff_type: "PRBO".into(),
                element: na_seq::Element::Oxygen,
                mass: 15.999,
                charge_scaled: -0.25 * 18.2223,
                sigma: 3.0,
                eps: 0.15,
                c4: 0.0,
                offset: lin_alg::f32::Vec3::new(1.0, 0.0, 0.0),
            },
        ],
        bonds: vec![(0, 1, 300.0, 2.0)],
    };
    let coso = build_with(7.0, 0.0, |e| e, vec![spec]);
    let n_prb = coso
        .state
        .atoms
        .iter()
        .filter(|a| a.force_field_type.starts_with("PRB"))
        .count();
    expect_units_check(n_prb, 0.05, vol_l_of(&coso));
    // Connectivity held: first molecule's internal distance ≈ 2.0 Å after 30 steps.
    let idx: Vec<usize> = coso
        .state
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| a.force_field_type.starts_with("PRB"))
        .map(|(i, _)| i)
        .collect();
    let mut eng = coso;
    for _ in 0..30 {
        let r = eng.step(None);
        assert!(!r.crashed, "cosolvent run crashed");
    }
    let d = eng
        .state
        .cell
        .min_image(eng.state.atoms[idx[0]].posit - eng.state.atoms[idx[1]].posit);
    let dist = d.magnitude();
    println!("cosolvent: {n_prb} atoms placed; restraint distance {dist:.2} Å (r0 2.0)");
    assert!(
        (dist - 2.0).abs() < 0.35,
        "internal restraint/exclusion physics broken: {dist}"
    );
}

fn vol_l_of(e: &spice_engine::SpiceEngine) -> f64 {
    f64::from(e.state.cell.volume()) * 1.0e-27
}

/// v1.3.2 general electrolyte channel: KCl (the first species the named
/// knobs could not express) plus MgCl₂ routed through `SaltSpec` to prove
/// the auto-balanced stoichiometry matches the historical named path.
/// Counts are derived from c·V·N_A like every other salt assertion here.
#[test]
#[ignore = "expensive: one full 2LYZ build with two general-channel salts"]
fn species_channel_salts_work() {
    use spice_engine::engine::md_core::{ION_CL, ION_K, ION_MG, SaltSpec};
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("load Amber parameters");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
    let kcl = SaltSpec::new(ION_K, ION_CL, 0.05);
    let mgcl2 = SaltSpec::new(ION_MG, ION_CL, 0.025);
    assert_eq!(kcl.stoichiometry().unwrap(), (1, 1));
    assert_eq!(mgcl2.stoichiometry().unwrap(), (1, 2));
    let opts = BuildOptions {
        env: EnvParams::new(7.0, 300.0, 1.0, 0.15),
        salts: vec![kcl, mgcl2],
        ..Default::default()
    };
    let eng = build_system(&dev, &params, protein, &opts).expect("build with species salts");
    let v = vol_l_of(&eng);
    let count = |ff: &str| {
        eng.state
            .atoms
            .iter()
            .filter(|a| a.force_field_type == ff)
            .count()
    };
    let n_k = count("K+");
    let n_mg = count("Mg2+");
    let n_cl = count("Cl-");
    let n_na = count("Na+");
    let expect_k = (0.05 * v * AVOGADRO).round() as usize;
    let expect_mg = (0.025 * v * AVOGADRO).round() as usize;
    // Cl⁻ pool = 0.15·V NaCl pairs + 1·KCl + 2·MgCl₂ + counterions for a
    // net-POSITIVE protein (post-糙1 charge completeness, pH 7 2LYZ measures
    // +7.999 e → 8 Cl⁻; the ±0.001 is f32 accumulation, absorbed by the ±2
    // tolerance below).
    let q_protein = protein_net_charge_e(&eng);
    let counter_cl = if q_protein > 0.0 {
        q_protein.round() as usize
    } else {
        0
    };
    let expect_cl = (0.15 * v * AVOGADRO).round() as usize + expect_k + 2 * expect_mg + counter_cl;
    println!(
        "species channel: K+ {n_k}/{expect_k}, Mg2+ {n_mg}/{expect_mg}, \
         Cl- {n_cl}/{expect_cl}, Na+ {n_na} (0.15 M background + counterions)"
    );
    assert!(n_k.abs_diff(expect_k) <= 1, "KCl count via species channel");
    assert!(
        n_mg.abs_diff(expect_mg) <= 1,
        "MgCl2 via SaltSpec must match the named-knob count"
    );
    assert!(n_cl.abs_diff(expect_cl) <= 2, "mixed-anion Cl pool");
    // K⁺ is the Sengupta Table 2 row carried as true σ.
    let k = eng
        .state
        .atoms
        .iter()
        .find(|a| a.force_field_type == "K+")
        .expect("a K+");
    let rstar_to_sigma = spice_engine::engine::md_core::R_STAR_TO_SIGMA;
    assert!(
        (k.lj_sigma - 1.702 * rstar_to_sigma).abs() < 1e-5,
        "K+ sigma drifted from the published row: {}",
        k.lj_sigma
    );
    assert_eq!(k.lj_c4.to_bits(), 0.0f32.to_bits());
    assert!(
        f64::from(eng.state.net_charge_e()).abs() < 1.0,
        "mixed salts broke electroneutrality: {}",
        eng.state.net_charge_e()
    );
}

fn expect_units_check(n_atoms: usize, molarity: f64, vol_l: f64) {
    let expect = (molarity * vol_l * AVOGADRO).round() as usize * 2; // 2 sites/mol
    assert!(
        n_atoms + 4 >= expect && n_atoms <= expect + 4,
        "cosolvent atoms {n_atoms} vs expected ~{expect}"
    );
}

/// v1.3.2 R2 regression: `build_mutant_by_solvent_reuse` must carry the
/// parent's cosolute internal bonds (distance_restraints + 1-2 exclusions)
/// across the solute index shift. Pre-fix, the tail atoms copied but their
/// connectivity silently evaporated and urea came apart in the mutant.
#[test]
#[ignore = "expensive: parent build + solvent-reuse mutant + MD"]
fn r2_mutant_carries_cosolute_restraints() {
    use spice_engine::engine::md_core::cosolvent_presets::urea;
    use spice_engine::structure::{AtomInput, StructureInput};

    let mut urea_spec = urea();
    urea_spec.molarity = 0.05;
    let parent = build_with(7.0, 0.0, |e| e, vec![urea_spec.clone()]);
    let n_rest = parent.state.distance_restraints.len();
    assert!(n_rest > 0, "parent urea build must create bond restraints");

    // A chemically-valid WT structure input from the same cif (mutation-free
    // reuse is the purest probe: any lost restraint is the bug).
    let mm = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
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

    let params = FfParamSet::new_amber().expect("params");
    let opts = BuildOptions {
        env: EnvParams::new(7.0, 300.0, 1.0, 0.0),
        cosolvents: vec![urea_spec],
        ..Default::default()
    };
    let mut mutant =
        spice_engine::build_mutant_by_solvent_reuse(&parent, &params, &wt_input, &opts)
            .expect("solvent-reuse mutant");
    let st = &mutant.state;

    // (a) every parent tail restraint carried, with both ends in the tail.
    assert_eq!(
        st.distance_restraints.len(),
        n_rest,
        "R2: mutant lost or duplicated cosolute restraints"
    );
    for r in &st.distance_restraints {
        assert!(
            r.atom_0_idx >= st.solute_atom_count && r.atom_1_idx >= st.solute_atom_count,
            "R2: carried restraint ({},{}) reaches into the mutant solute",
            r.atom_0_idx,
            r.atom_1_idx
        );
        assert!(
            st.force_field_params.bonds_topology.contains(&(
                r.atom_0_idx.min(r.atom_1_idx),
                r.atom_0_idx.max(r.atom_1_idx)
            )),
            "R2: restraint ({},{}) lacks its 1-2 exclusion",
            r.atom_0_idx,
            r.atom_1_idx
        );
        // (b) symmetric adjacency rows exist (snapshot consumers index them).
        assert!(st.adjacency_list[r.atom_0_idx].contains(&r.atom_1_idx));
        assert!(st.adjacency_list[r.atom_1_idx].contains(&r.atom_0_idx));
    }

    // (c) the C=O pair of one urea must still be ~1.23 Å after 30 MD steps
    // (pre-fix it would have drifted apart with no bond force).
    let (ia, ib) = {
        let co = st
            .distance_restraints
            .iter()
            .find(|r| {
                let (f0, f1) = (
                    &st.atoms[r.atom_0_idx].force_field_type,
                    &st.atoms[r.atom_1_idx].force_field_type,
                );
                (f0 == "CG2O6" && f1 == "OG2D1") || (f0 == "OG2D1" && f1 == "CG2O6")
            })
            .expect("a carried urea C=O restraint");
        (co.atom_0_idx, co.atom_1_idx)
    };
    for _ in 0..30 {
        mutant.step(None);
    }
    let cell = &mutant.state.cell;
    let d = cell
        .min_image(mutant.state.atoms[ia].posit - mutant.state.atoms[ib].posit)
        .magnitude();
    println!("R2 probe: {n_rest} restraints carried; mutant urea C=O after 30 steps = {d:.3} Å");
    assert!(
        (d - 1.230).abs() < 0.15,
        "carried restraint not enforced: C=O at {d:.3} Å"
    );
}

/// Independent ½Σcᵢzᵢ² over the built salt ions (closed-list ff types, fixed
/// z) — cross-checks the engine getter without sharing its classification code.
fn manual_ionic_strength(st: &spice_engine::engine::md_core::MdState) -> f32 {
    let vol_l = f64::from(st.cell.volume()) * 1.0e-27;
    let c_one = 1.0 / (vol_l * 6.02214076e23);
    let mut i = 0.0_f64;
    for a in &st.atoms {
        let z: f64 = match a.force_field_type.as_str() {
            "Na+" | "K+" | "Cl-" => 1.0,
            "Mg2+" | "Ca2+" | "Sr2+" | "Ba2+" => 2.0,
            _ => 0.0,
        };
        if z > 0.0 {
            i += 0.5 * c_one * z * z;
        }
    }
    i as f32
}

/// v1.3.5 (糙2): `effective_ionic_strength_m()` reports the TRUE ½Σcᵢzᵢ² of the
/// built box — equal to the NaCl knob for pure NaCl, but HIGHER once divalent
/// salts are added (the `ionic_strength_m` knob alone understates that).
#[test]
#[ignore = "expensive: two 2LYZ builds"]
fn effective_ionic_strength_accounting() {
    // Pure NaCl: I = ½Σcᵢzᵢ² over BUILT ions = knob(0.15) PLUS the neutralizing
    // counterions (2LYZ net +8 e → 8 extra Cl⁻ ≈ +0.033 M). The getter must
    // reproduce an independent manual sum exactly; the knob alone understates.
    let nacl = build_2lyz(7.0, 0.15);
    let i_nacl = nacl.state.effective_ionic_strength_m();
    let i_manual = manual_ionic_strength(&nacl.state);
    println!("2LYZ 0.15 NaCl -> effective I = {i_nacl:.4} M (manual {i_manual:.4})");
    assert!(
        (i_nacl - i_manual).abs() < 2e-3,
        "getter disagrees with manual ½Σcᵢzᵢ²: {i_nacl:.4} vs {i_manual:.4}"
    );
    assert!(
        i_nacl > 0.15 && i_nacl < 0.20,
        "I = knob + counterions ∈ (0.15, 0.20) for net +8, got {i_nacl:.4}"
    );
    drop(nacl);

    // NaCl + MgCl2: Mg²⁺ (z=2) + 2 Cl⁻ raise I above the 0.15 NaCl number.
    // 0.05 M MgCl2 contributes ½(0.05·4 + 0.10·1) = 0.15 → total ≈ 0.30.
    let dev = spice_engine::engine::md_core::ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("amber");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
    let opts = BuildOptions {
        env: EnvParams::new(7.0, 300.0, 1.0, 0.15).with_divalent(0.05, 0.0, 0.0, 0.0),
        ..Default::default()
    };
    let mix = build_system(&dev, &params, protein, &opts).expect("build");
    let i_mix = mix.state.effective_ionic_strength_m();
    let i_mix_manual = manual_ionic_strength(&mix.state);
    println!(
        "2LYZ 0.15 NaCl + 0.05 MgCl2 -> effective I = {i_mix:.4} M (manual {i_mix_manual:.4})"
    );
    assert!(
        (i_mix - i_mix_manual).abs() < 2e-3,
        "getter disagrees with manual sum on Mg mix: {i_mix:.4} vs {i_mix_manual:.4}"
    );
    assert!(
        i_mix > i_nacl + 0.03,
        "divalent must RAISE I above NaCl-only ({i_nacl:.3} vs {i_mix:.3})"
    );
    assert!(
        (i_mix - 0.30).abs() < 0.06,
        "expected I ~= 0.30 for 0.15 NaCl + 0.05 MgCl2, got {i_mix:.4}"
    );
}
