use spice_engine::{AtomInput, RnaStructureInput, StructureInput, validate_rna_input};

#[test]
fn accepts_existing_rna_coordinates_with_protein_aligned_input() {
    let mut input: RnaStructureInput = StructureInput::default();
    input.push(AtomInput::new(
        "A",
        1,
        "A",
        "P",
        na_seq::Element::Phosphorus,
        0.0,
        0.0,
        0.0,
    ));
    input.push(AtomInput::new(
        "A",
        2,
        "U",
        "P",
        na_seq::Element::Phosphorus,
        1.0,
        0.0,
        0.0,
    ));
    assert!(validate_rna_input(&input).is_ok());
}

#[test]
fn rejects_empty_rna_coordinates() {
    let input: RnaStructureInput = StructureInput::default();
    assert!(validate_rna_input(&input).is_err());
}
