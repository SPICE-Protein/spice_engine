//! RNA structure input for the SPICE Engine.
//!
//! Folding and conformation generation belong to SPICE Model (for example
//! S4RNA). The Engine receives an already generated/selected 3D RNA structure
//! and prepares it for physical evaluation; it does not predict folding.

use crate::structure::StructureInput;

/// RNA uses the same atom-level transport contract as protein input. The type
/// alias keeps the S4RNA/Engine API explicitly aligned without introducing a
/// second wire format.
pub type RnaStructureInput = StructureInput;

pub fn validate_rna_input(input: &RnaStructureInput) -> Result<(), String> {
    if input.atoms.is_empty() {
        return Err("RnaStructureInput has no atoms".into());
    }
    if input
        .atoms
        .iter()
        .any(|atom| atom.res_name.trim().is_empty())
    {
        return Err("RnaStructureInput contains an atom with an empty residue name".into());
    }
    Ok(())
}

/// Build an Engine from an existing RNA 3D structure.
///
/// RNA topology and nucleic-acid parameter preparation are intentionally kept
/// separate from the protein builder and will be connected to the Amber/CHARMM
/// RNA adapters by the S4RNA integration work.
pub fn build_rna_from_input(
    _dev: &crate::engine::md_core::ComputationDevice,
    _param_set: &crate::engine::md_core::params::FfParamSet,
    input: &RnaStructureInput,
    opts: &crate::builder::BuildOptions,
) -> Result<crate::engine::SpiceEngine, String> {
    validate_rna_input(input)?;
    if opts.computation_content != crate::forcefield::ComputationContent::Rna {
        return Err("build_rna_from_input requires ComputationContent::Rna".into());
    }
    Err("RNA topology/preparation is not connected yet; SPICE Engine expects an existing RNA 3D structure".into())
}
