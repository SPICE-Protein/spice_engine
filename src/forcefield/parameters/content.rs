//! Calculation-content selection for force-field adapters.

use super::{Amber19, Charmm36m, Martini3};
use crate::forcefield::traits::ForceField;
use crate::forcefield::traits::PreparedForceField;
use crate::forcefield::types::{ForceFieldError, Resolution};

/// The physical content represented by the calculation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComputationContent {
    Protein,
    Rna,
    Dna,
    Ligand,
    Lipid,
    CoarseGrainedProtein,
}

impl ComputationContent {
    pub const fn is_coarse_grained(self) -> bool {
        matches!(self, Self::CoarseGrainedProtein)
    }

    pub const fn is_nucleic_acid(self) -> bool {
        matches!(self, Self::Rna | Self::Dna)
    }

    pub const fn requires_mapping(self) -> bool {
        self.is_coarse_grained()
    }
}

/// Selectable existing force-field family for a content type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForceFieldSelection {
    Amber19,
    Charmm36m,
    Martini3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParameterDomain {
    Protein,
    Rna,
    Dna,
    Ligand,
    Lipid,
    CoarseGrainedProtein,
}

impl ForceFieldSelection {
    pub fn for_content(content: ComputationContent) -> Self {
        match content {
            ComputationContent::CoarseGrainedProtein => Self::Martini3,
            ComputationContent::Protein
            | ComputationContent::Rna
            | ComputationContent::Dna
            | ComputationContent::Ligand
            | ComputationContent::Lipid => Self::Amber19,
        }
    }

    pub const fn domain_for_content(content: ComputationContent) -> ParameterDomain {
        match content {
            ComputationContent::Protein => ParameterDomain::Protein,
            ComputationContent::Rna => ParameterDomain::Rna,
            ComputationContent::Dna => ParameterDomain::Dna,
            ComputationContent::Ligand => ParameterDomain::Ligand,
            ComputationContent::Lipid => ParameterDomain::Lipid,
            ComputationContent::CoarseGrainedProtein => ParameterDomain::CoarseGrainedProtein,
        }
    }

    pub fn supports_content(self, content: ComputationContent) -> bool {
        matches!(
            (self, content),
            (
                Self::Amber19,
                ComputationContent::Protein
                    | ComputationContent::Rna
                    | ComputationContent::Dna
                    | ComputationContent::Ligand
                    | ComputationContent::Lipid
            ) | (
                Self::Charmm36m,
                ComputationContent::Protein
                    | ComputationContent::Rna
                    | ComputationContent::Dna
                    | ComputationContent::Ligand
                    | ComputationContent::Lipid
            ) | (Self::Martini3, ComputationContent::CoarseGrainedProtein)
        )
    }

    pub fn resolution(self) -> Resolution {
        match self {
            Self::Amber19 | Self::Charmm36m => Resolution::ALL_ATOM,
            Self::Martini3 => Resolution::new(0.25).expect("valid Martini resolution"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Amber19 => "amber19",
            Self::Charmm36m => "charmm36m",
            Self::Martini3 => "martini3",
        }
    }

    pub fn validate_content(self, content: ComputationContent) -> Result<(), ForceFieldError> {
        if self.supports_content(content) {
            Ok(())
        } else {
            Err(ForceFieldError(format!(
                "force field '{}' does not support computation content {:?}",
                self.name(),
                content
            )))
        }
    }

    pub fn prepare_for_content(
        self,
        content: ComputationContent,
    ) -> Result<Box<dyn PreparedForceField>, ForceFieldError> {
        self.validate_content(content)?;
        self.prepare()
    }

    pub fn prepare(self) -> Result<Box<dyn PreparedForceField>, ForceFieldError> {
        match self {
            Self::Amber19 => Ok(Box::new(Amber19.prepare()?)),
            Self::Charmm36m => Ok(Box::new(Charmm36m.prepare()?)),
            Self::Martini3 => Ok(Box::new(Martini3.prepare()?)),
        }
    }
}
