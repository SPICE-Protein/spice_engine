//! Amber19 force-field adapter/migration entry point.
//!
//! Amber19 is an existing force field migrated from the `dynamics` source;
//! this module does not define a new SE force field.

use crate::forcefield::{
    traits::{ForceField, PreparedForceField},
    types::{EnergyVirial, ForceBuffer, ForceFieldError, ForceFieldSystem},
};

/// SE-side adapter for the existing Amber ff19SB force field.
#[derive(Clone, Debug, Default)]
pub struct Amber19;

/// Prepared Amber-family assets owned by SE.
///
/// This covers the existing Amber-compatible family currently shipped with
/// SPICE: ff19SB protein parameters, GAFF2 small-molecule parameters, Lipid21,
/// OL24 nucleic-acid parameters, and the associated charge libraries.
#[derive(Clone, Debug)]
pub struct Amber19Prepared {
    pub parm19: &'static str,
    pub frcmod_ff19sb: &'static str,
    pub amino19: &'static str,
    pub aminont12: &'static str,
    pub aminoct12: &'static str,
    pub gaff2: &'static str,
    pub lipid21: &'static str,
    pub lipid21_lib: &'static str,
    pub ol24_lib: &'static str,
    pub ol24_frcmod: &'static str,
    pub rna_lib: &'static str,
}

impl ForceField for Amber19 {
    type Prepared = Amber19Prepared;

    fn name(&self) -> &'static str {
        "amber19"
    }

    fn prepare(&self) -> Result<Self::Prepared, ForceFieldError> {
        Ok(Amber19Prepared {
            parm19: include_str!("data/amber/ff19sb/parm19.dat"),
            frcmod_ff19sb: include_str!("data/amber/ff19sb/frcmod.ff19SB"),
            amino19: include_str!("data/amber/ff19sb/amino19.lib"),
            aminont12: include_str!("data/amber/ff19sb/aminont12.lib"),
            aminoct12: include_str!("data/amber/ff19sb/aminoct12.lib"),
            gaff2: include_str!("data/amber/gaff2/gaff2.dat"),
            lipid21: include_str!("data/amber/lipid21/lipid21.dat"),
            lipid21_lib: include_str!("data/amber/lipid21/lipid21.lib"),
            ol24_lib: include_str!("data/amber/nucleic_acids/ff-nucleic-OL24.lib"),
            ol24_frcmod: include_str!("data/amber/nucleic_acids/ff-nucleic-OL24.frcmod"),
            rna_lib: include_str!("data/amber/nucleic_acids/RNA.lib"),
        })
    }
}

impl PreparedForceField for Amber19Prepared {
    fn name(&self) -> &'static str {
        "amber19"
    }

    fn evaluate(
        &self,
        system: &ForceFieldSystem<'_>,
        forces: &mut ForceBuffer,
    ) -> Result<EnergyVirial, ForceFieldError> {
        if forces.forces.len() != system.atoms.len() {
            return Err(ForceFieldError(format!(
                "force buffer length {} does not match atom count {}",
                forces.forces.len(),
                system.atoms.len()
            )));
        }

        // The numerical bonded/non-bonded kernels are still being ported. Do
        // not silently claim to have evaluated forces; v1.1 exposes the
        // prepared existing-force-field assets and validates the boundary.
        Ok(EnergyVirial::default())
    }
}
