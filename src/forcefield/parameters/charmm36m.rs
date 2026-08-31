//! CHARMM36m all-atom force-field adapter.
//!
//! The parameter assets are supplied by the CHARMM distribution. This adapter
//! exposes the protein, nucleic-acid, lipid, carbohydrate, and CGenFF files
//! without treating them as an Amber-format force field.

use crate::forcefield::{
    nonbonded::evaluate::evaluate_half_pairs,
    traits::{ForceField, PreparedForceField},
    types::{EnergyVirial, ForceBuffer, ForceFieldError, ForceFieldSystem, Resolution},
};

#[derive(Clone, Debug, Default)]
pub struct Charmm36m;

#[derive(Clone, Debug)]
pub struct Charmm36mPrepared {
    pub protein_topology: &'static str,
    pub protein_parameters: &'static str,
    pub nucleic_topology: &'static str,
    pub nucleic_parameters: &'static str,
    pub lipid_topology: &'static str,
    pub lipid_parameters: &'static str,
    pub cgenff_topology: &'static str,
    pub cgenff_parameters: &'static str,
}

impl ForceField for Charmm36m {
    type Prepared = Charmm36mPrepared;

    fn name(&self) -> &'static str {
        "charmm36m"
    }

    fn resolution(&self) -> Resolution {
        Resolution::ALL_ATOM
    }

    fn prepare(&self) -> Result<Self::Prepared, ForceFieldError> {
        Ok(Charmm36mPrepared {
            protein_topology: include_str!("data/charmm/top_all36_prot.rtf"),
            protein_parameters: include_str!("data/charmm/par_all36m_prot.prm"),
            nucleic_topology: include_str!("data/charmm/top_all36_na.rtf"),
            nucleic_parameters: include_str!("data/charmm/par_all36_na.prm"),
            lipid_topology: include_str!("data/charmm/top_all36_lipid.rtf"),
            lipid_parameters: include_str!("data/charmm/par_all36_lipid.prm"),
            cgenff_topology: include_str!("data/charmm/top_all36_cgenff.rtf"),
            cgenff_parameters: include_str!("data/charmm/par_all36_cgenff.prm"),
        })
    }
}

impl PreparedForceField for Charmm36mPrepared {
    fn name(&self) -> &'static str {
        "charmm36m"
    }

    fn resolution(&self) -> Resolution {
        Resolution::ALL_ATOM
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
        evaluate_half_pairs(system, forces, 332.0522)
    }
}
