//! Integration guard for the v1.3.8 observability probes on a REAL system
//! (2LYZ hen lysozyme, pH 7, 0.15 M NaCl): electrostatic potential, per-site
//! SASA, and subset geometry (select / contacts / bottleneck).
//!
//! The unit tests in `analysis.rs` pin the math on synthetic boxes; this file
//! pins the ENGINE WIRING: charge-site packing shared with PME, index spaces
//! aligned with `atom_labels`, and sane physics on a screened, solvated,
//! neutralized box. One build serves all three groups.
//!
//! Run: `cargo test --release --test observables -- --ignored --nocapture --test-threads=1`

use bio_files::MmCif;
use spice_engine::engine::md_core::ComputationDevice;
use spice_engine::engine::md_core::params::FfParamSet;
use spice_engine::env::EnvParams;
use spice_engine::{BuildOptions, build_system};
use std::path::Path;

fn build_2lyz() -> spice_engine::SpiceEngine {
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("load Amber parameters");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
    let opts = BuildOptions {
        env: EnvParams::new(7.0, 300.0, 0.0, 0.15),
        equil: None,
        ..Default::default()
    };
    build_system(&dev, &params, protein, &opts).expect("build engine")
}

#[test]
#[ignore = "expensive: one full 2LYZ build (shared by the three probe groups)"]
fn observables_on_2lyz() {
    let e = build_2lyz();
    let n_atoms = e.state.atoms.len();
    let n_water = e.state.water.len();
    assert!(
        n_atoms > 1500 && n_water > 4000,
        "setup {n_atoms}/{n_water}"
    );

    // ------------------------------------------------------------------
    // (1) ESP probe: finite, deterministic, and sign-dominated by the
    // nearest charge at contact distance. Relative quantities only (the
    // PME gauge constant cancels in differences).
    // ------------------------------------------------------------------
    // Most negative partial charge in the solute (a carboxylate O at pH 7).
    let (worst_i, worst_q) = e
        .state
        .atoms
        .iter()
        .enumerate()
        .map(|(i, a)| (i, f64::from(a.partial_charge)))
        .min_by(|(_, qa), (_, qb)| qa.partial_cmp(qb).unwrap())
        .expect("nonempty");
    assert!(
        worst_q < -5.0,
        "expected a carboxylate-like negative site, got q={worst_q}"
    );
    let r0 = e.state.atoms[worst_i].posit;
    let near = [[r0.x as f64 + 1.5, r0.y as f64, r0.z as f64]];
    let far = [[r0.x as f64 + 8.0, r0.y as f64, r0.z as f64]];
    let phi_near = e.state.electrostatic_potential(&near);
    let phi_far = e.state.electrostatic_potential(&far);
    let d_phi = phi_near[0] - phi_far[0];
    // -0.5e-ish source: Coulomb alone gives ~ -60..-40 kcal/mol/e at 1.5 A vs
    // ~ -0.5..-5 at 8 A; solvent/ion environment moves this a lot but cannot
    // flip the sign at these distances. Require a solid negative shift.
    assert!(
        d_phi < -20.0,
        "probe near negative charge should sit far below a point 6.5 A away, got {d_phi}"
    );
    // Determinism: pure function of state, bit-equal across calls.
    assert_eq!(phi_near, e.state.electrostatic_potential(&near));
    // Catalytic-axis style query: field magnitude between two close points
    // must stay sane (no factor-of-CUS leaks: a unit bug would explode this).
    let pair = [
        [r0.x as f64 + 1.5, r0.y as f64, r0.z as f64],
        [r0.x as f64 + 2.5, r0.y as f64, r0.z as f64],
    ];
    let phi = e.state.electrostatic_potential(&pair);
    let field = (phi[0] - phi[1]).abs(); // per 1 A step, kcal/mol/e/A
    assert!(
        field.is_finite() && field < 500.0,
        "unreasonable probe field {field} - unit convention leak?"
    );

    // ------------------------------------------------------------------
    // (2) SASA: layout contract + physics on the real box.
    // ------------------------------------------------------------------
    let sasa = e.state.atom_sasa(1.4, 200);
    assert_eq!(
        sasa.len(),
        n_atoms,
        "one SASA value per state.atoms entry (water is the probe, not an occluder)"
    );
    assert!(sasa.iter().all(|&s| s.is_finite() && s >= 0.0));
    let solute: f64 = sasa.iter().sum();
    // Lysozyme (129 res) is a small globular protein: naked-surface SASA in
    // the 1.4e4 A^2 range; the built box is post-minimize, water-screened.
    assert!(
        solute > 4_000.0 && solute < 40_000.0,
        "total solute+ion SASA {solute} A^2 outside any sane band for 2LYZ"
    );
    // Buried vs exposed: the most buried non-H solute atom must be nearly
    // closed; the most exposed one must have real water access.
    let mut min_s = f64::MAX;
    let mut max_s = 0.0f64;
    for (i, a) in e.state.atoms.iter().enumerate() {
        if a.element == na_seq::Element::Hydrogen {
            continue;
        }
        min_s = min_s.min(sasa[i]);
        max_s = max_s.max(sasa[i]);
    }
    assert!(min_s < 2.0, "no near-buried core atom (min {min_s})?");
    assert!(
        max_s > 15.0,
        "max heavy-atom SASA {max_s} - everything sealed?"
    );
    println!("[sasa] solute total {solute:.0} A^2, heavy-atom range [{min_s:.2}, {max_s:.1}]");

    // ------------------------------------------------------------------
    // (3) Subset geometry: names / select / contacts / bottleneck.
    // ------------------------------------------------------------------
    let names = e.atom_names();
    assert_eq!(names.len(), n_atoms);
    // 2LYZ catalytic pair: GLU35 and ASP52. Sidechain heavy atoms only.
    let cat = e.select_atoms(&[35, 52], None, true);
    let cat_names: Vec<&str> = cat.iter().map(|&i| names[i].as_str()).collect();
    for &i in &cat {
        let a = &e.state.atoms[i];
        assert_ne!(
            a.element,
            na_seq::Element::Hydrogen,
            "H leaked into selection"
        );
        assert!(
            !matches!(names[i].as_str(), "N" | "CA" | "C" | "O" | "OXT"),
            "backbone atom {} leaked",
            names[i]
        );
    }
    let expected = ["CB", "CG", "CD", "OE1", "OE2", "OD1", "OD2"];
    assert!(
        cat.iter().all(|&i| expected.contains(&names[i].as_str())),
        "unexpected catalytic sidechain names {cat_names:?}"
    );
    assert!(
        cat_names.contains(&"OE1") && cat_names.contains(&"OD1"),
        "catalytic carboxylates missing: {cat_names:?}"
    );
    // Names allow-list roundtrip: exactly the OE1 of Glu35.
    let one = e.select_atoms(&[35], Some(&["OE1".to_string()]), false);
    assert_eq!(one.len(), 1, "names filter returned {one:?}");
    assert_eq!(names[one[0]], "OE1");

    // Contacts within the catalytic sidechains (9 heavy atoms, residue
    // internals at 4 A): GLU ~10 + ASP ~6 pairs; the two residues are ~10 A
    // apart in lysozyme so cross terms are few.
    let contacts = e.contact_count(&cat, &cat, 4.0);
    assert!(
        contacts >= 8 && contacts <= 22,
        "catalytic self-contacts {contacts} implausible (expect ~16)"
    );
    // Cross-set vs itself ordering sanity: whole-solute self-contacts must be
    // far more than a 6-atom subset's.
    let all: Vec<usize> = (0..n_atoms).collect();
    assert!(e.contact_count(&all, &cat, 4.0) > contacts);

    // Bottleneck: a vertical line through the protein's heavy-atom centroid
    // must cross real core matter (negative clearance somewhere); a line in
    // the box corner lane (hydration margin, no solute) stays open.
    let ext = e.state.cell.extent;
    let lo = e.state.cell.bounds_low;
    let cx: f64 = e
        .topology
        .heavy_indices
        .iter()
        .map(|&i| e.state.atoms[i].posit.x as f64)
        .sum::<f64>()
        / e.topology.heavy_indices.len() as f64;
    let cy: f64 = e
        .topology
        .heavy_indices
        .iter()
        .map(|&i| e.state.atoms[i].posit.y as f64)
        .sum::<f64>()
        / e.topology.heavy_indices.len() as f64;
    let through = vec![
        [cx, cy, lo.z as f64 + 0.5],
        [cx, cy, lo.z as f64 + ext.z as f64 - 0.5],
    ];
    let prof_in = e.bottleneck_profile(&through, 0.5, &[], false);
    let bot_in = prof_in.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        bot_in < 0.5,
        "centroid-crossing path found no bottleneck ({bot_in}) - radii or box wrong"
    );
    // Corner lane: exclude the free ions (they roam the whole box and would
    // block any lane) - this then tests the solute-free hydration margin.
    let ions: Vec<usize> = e
        .state
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| {
            matches!(
                a.force_field_type.as_str(),
                "Na+" | "K+" | "Cl-" | "Mg2+" | "Ca2+" | "Sr2+" | "Ba2+"
            )
        })
        .map(|(i, _)| i)
        .collect();
    let corner = vec![
        [lo.x as f64 + 0.5, lo.y as f64 + 0.5, lo.z as f64 + 0.5],
        [
            lo.x as f64 + 0.5,
            lo.y as f64 + 0.5,
            lo.z as f64 + ext.z as f64 - 0.5,
        ],
    ];
    let prof_out = e.bottleneck_profile(&corner, 0.5, &ions, false);
    let bot_out = prof_out.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        bot_out > 1.5,
        "corner lane blocked (bottleneck {bot_out}) despite ion exclusion - box geometry wrong?"
    );
    println!(
        "[bottleneck] through-protein {bot_in:.2} A, corner lane {bot_out:.2} A, \
         ESP dphi {d_phi:.1} kcal/mol/e"
    );
}
