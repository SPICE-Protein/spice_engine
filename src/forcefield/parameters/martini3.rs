//! Martini 3 coarse-grained force-field adapter.
//!
//! Martini 3 is a bead-based coarse-grained model. Its topology/mapping files
//! are intentionally kept separate from all-atom Amber and CHARMM assets.

use crate::forcefield::{
    nonbonded::evaluate::evaluate_half_pairs,
    traits::{ForceField, PreparedForceField},
    types::{EnergyVirial, ForceBuffer, ForceFieldError, ForceFieldSystem, Resolution},
};

#[derive(Clone, Debug, Default)]
pub struct Martini3;

#[derive(Clone, Debug)]
pub struct Martini3Prepared {
    pub particles: &'static str,
    pub ions: &'static str,
    pub proteins: &'static str,
    pub nucleobases: &'static str,
    pub phospholipids: &'static str,
    pub small_molecules: &'static str,
    pub solvents: &'static str,
    pub sugars: &'static str,
}

impl ForceField for Martini3 {
    type Prepared = Martini3Prepared;

    fn name(&self) -> &'static str {
        "martini3"
    }

    /// Martini 3 maps roughly 2–4 heavy atoms to one bead; `0.25` is the
    /// normalized coarse-grained metadata value used by SE for this model.
    fn resolution(&self) -> Resolution {
        Resolution::new(0.25).expect("valid Martini resolution")
    }

    fn prepare(&self) -> Result<Self::Prepared, ForceFieldError> {
        Ok(Martini3Prepared {
            particles: include_str!("data/martini/martini3/martini_v3.0.0.itp"),
            ions: include_str!("data/martini/martini3/martini_v3.0.0_ions_v1.itp"),
            proteins: include_str!(
                "data/martini/martini3/martini_v3.0.0_proteins/force_fields/martini3001/aminoacids.ff"
            ),
            nucleobases: include_str!("data/martini/martini3/martini_v3.0.0_nucleobases_v1.itp"),
            phospholipids: include_str!(
                "data/martini/martini3/martini_v3.0.0_phospholipids_v1.itp"
            ),
            small_molecules: include_str!(
                "data/martini/martini3/martini_v3.0.0_small_molecules_v1.itp"
            ),
            solvents: include_str!("data/martini/martini3/martini_v3.0.0_solvents_v1.itp"),
            sugars: include_str!("data/martini/martini3/martini_v3.0.0_sugars_v1.itp"),
        })
    }
}

impl PreparedForceField for Martini3Prepared {
    fn name(&self) -> &'static str {
        "martini3"
    }

    fn resolution(&self) -> Resolution {
        Resolution::new(0.25).expect("valid Martini resolution")
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
