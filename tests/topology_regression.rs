//! Regression tests for the established protein reconstruction/build path.
//! These tests exercise the same in-memory ingestion path used by the FFI, but
//! stop before expensive solvation/minimization to isolate atom ordering and
//! topology invariants.

use std::collections::HashSet;
use std::path::Path;

use bio_files::MmCif;
use na_seq::AtomTypeInRes;
use spice_engine::engine::dynamics::params::{FfParamSet, prepare_peptide_mmcif};
use spice_engine::{BuildOptions, ProteinTopology, atoms_to_mmcif};

#[test]
fn prepared_protein_ca_indices_are_one_per_residue_and_stable() {
    let mut cif = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ");
    let params = FfParamSet::new_amber().expect("load Amber parameters");
    let map = params.peptide_ff_q_map.as_ref().expect("peptide map");
    let (bonds, _) =
        prepare_peptide_mmcif(&mut cif, map, BuildOptions::default().env.ph, None, true)
            .expect("prepare protein");
    assert!(!bonds.is_empty());

    let topology_a = ProteinTopology::from_prepared(&cif).expect("topology");
    let topology_b = ProteinTopology::from_prepared(&cif).expect("topology repeat");
    assert_eq!(topology_a.sequence, topology_b.sequence);
    assert_eq!(topology_a.ca_indices, topology_b.ca_indices);
    assert_eq!(topology_a.residues.len(), topology_a.ca_indices.len());
    assert_eq!(topology_a.residues.len(), topology_a.n_indices.len());
    assert_eq!(topology_a.residues.len(), topology_a.c_indices.len());
    assert_eq!(topology_a.residues.len(), topology_a.o_indices.len());

    let mut seen = HashSet::new();
    for (residue, &ca) in topology_a.residues.iter().zip(&topology_a.ca_indices) {
        assert!(seen.insert(ca), "duplicate CA index {ca}");
        assert!(residue.atom_indices.contains(&ca));
        assert_eq!(cif.atoms[ca].type_in_res, Some(AtomTypeInRes::CA));
    }
}

#[test]
fn structure_input_preserves_residue_order_and_ca_mapping_contract() {
    let input = spice_engine::StructureInput {
        atoms: vec![
            spice_engine::AtomInput::new(
                "A",
                20,
                "ALA",
                "N",
                na_seq::Element::Nitrogen,
                0.,
                0.,
                0.,
            ),
            spice_engine::AtomInput::new("A", 20, "ALA", "CA", na_seq::Element::Carbon, 1., 0., 0.),
            spice_engine::AtomInput::new("A", 20, "ALA", "C", na_seq::Element::Carbon, 2., 0., 0.),
            spice_engine::AtomInput::new(
                "A",
                30,
                "GLY",
                "N",
                na_seq::Element::Nitrogen,
                3.,
                0.,
                0.,
            ),
            spice_engine::AtomInput::new("A", 30, "GLY", "CA", na_seq::Element::Carbon, 4., 0., 0.),
            spice_engine::AtomInput::new("A", 30, "GLY", "C", na_seq::Element::Carbon, 5., 0., 0.),
        ],
    };
    let cif = atoms_to_mmcif(&input).expect("convert structure");
    assert_eq!(cif.residues.len(), 2);
    assert_eq!(cif.residues[0].serial_number, 1);
    assert_eq!(cif.residues[1].serial_number, 2);
    assert_eq!(cif.residues[0].atom_sns.len(), 3);
    assert_eq!(cif.residues[1].atom_sns.len(), 3);
    assert_eq!(cif.atoms[1].type_in_res, Some(AtomTypeInRes::CA));
    assert_eq!(cif.atoms[4].type_in_res, Some(AtomTypeInRes::CA));
}
