//! Amber19 force-field adapter/migration entry point.
//!
//! Amber19 is an existing force field migrated from the `dynamics` source;
//! this module does not define a new SE force field.

use std::collections::HashMap;

use crate::forcefield::{
    nonbonded::evaluate::evaluate_half_pairs,
    traits::{ForceField, PreparedForceField},
    types::{EnergyVirial, ForceBuffer, ForceFieldError, ForceFieldSystem},
};

/// A compact SE-owned index of Amber atom and frcmod records.
#[derive(Clone, Debug, Default)]
pub struct AmberParameterIndex {
    pub masses: HashMap<String, AmberMass>,
    pub bonds: HashMap<String, AmberBond>,
    pub angles: HashMap<String, AmberAngle>,
    pub dihedrals: HashMap<String, AmberDihedral>,
    pub impropers: HashMap<String, AmberDihedral>,
    pub nonbonded: HashMap<String, AmberNonbonded>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmberMass {
    pub mass: f32,
    pub radius: Option<f32>,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmberBond {
    pub k: f32,
    pub length: f32,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmberAngle {
    pub k: f32,
    pub angle_deg: f32,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmberDihedral {
    pub pk: f32,
    pub periodicity: f32,
    pub phase_deg: f32,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmberNonbonded {
    pub radius: f32,
    pub depth: f32,
}

impl AmberParameterIndex {
    pub fn from_sources(parm19: &str, frcmod: &str) -> Result<Self, ForceFieldError> {
        let mut out = Self::default();
        for line in parm19.lines() {
            let line = line.split('!').next().unwrap_or("").trim();
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.len() >= 2 && fields[0].len() <= 4 {
                if let Ok(mass) = fields[1].parse::<f32>() {
                    let radius = fields.get(2).and_then(|v| v.parse().ok());
                    out.masses
                        .insert(fields[0].to_ascii_uppercase(), AmberMass { mass, radius });
                }
            }
        }
        let mut section = "";
        for raw in frcmod.lines() {
            let line = raw.split('!').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            if matches!(
                line,
                "MASS" | "BOND" | "ANGL" | "DIHE" | "IMPROPER" | "NONBON"
            ) {
                section = line;
                continue;
            }
            let fields: Vec<_> = line.split_whitespace().collect();
            let key = fields
                .first()
                .map(|v| match section {
                    "BOND" => {
                        let p: Vec<_> = v.split('-').collect();
                        if p.len() == 2 {
                            canonical_pair(p[0], p[1])
                        } else {
                            canonical_key(v)
                        }
                    }
                    _ => canonical_key(v),
                })
                .unwrap_or_default();
            if section == "MASS" && fields.len() >= 2 {
                if let Ok(mass) = fields[1].parse() {
                    let radius = fields.get(2).and_then(|v| v.parse().ok());
                    out.masses.insert(key, AmberMass { mass, radius });
                }
                continue;
            }
            match section {
                "BOND" if fields.len() >= 3 => {
                    if let (Ok(k), Ok(length)) = (fields[1].parse(), fields[2].parse()) {
                        out.bonds.insert(key, AmberBond { k, length });
                    }
                }
                "ANGL" if fields.len() >= 3 => {
                    if let (Ok(k), Ok(angle_deg)) = (fields[1].parse(), fields[2].parse()) {
                        out.angles.insert(key, AmberAngle { k, angle_deg });
                    }
                }
                "DIHE" | "IMPROPER" if fields.len() >= 4 => {
                    if let (Ok(pk), Ok(periodicity), Ok(phase_deg)) =
                        (fields[1].parse(), fields[2].parse(), fields[3].parse())
                    {
                        let target = if section == "DIHE" {
                            &mut out.dihedrals
                        } else {
                            &mut out.impropers
                        };
                        target.insert(
                            key,
                            AmberDihedral {
                                pk,
                                periodicity,
                                phase_deg,
                            },
                        );
                    }
                }
                "NONBON" if fields.len() >= 3 => {
                    if let (Ok(radius), Ok(depth)) = (fields[1].parse(), fields[2].parse()) {
                        out.nonbonded.insert(key, AmberNonbonded { radius, depth });
                    }
                }
                _ => {}
            }
        }
        if out.masses.is_empty() {
            return Err(ForceFieldError(
                "Amber parameter source contains no MASS records".into(),
            ));
        }
        Ok(out)
    }
    pub fn mass(&self, atom_type: &str) -> Option<AmberMass> {
        self.masses.get(&atom_type.to_ascii_uppercase()).copied()
    }
    pub fn bond(&self, a: &str, b: &str) -> Option<AmberBond> {
        self.bonds.get(&canonical_pair(a, b)).copied()
    }
    pub fn angle(&self, a: &str, b: &str, c: &str) -> Option<AmberAngle> {
        self.angles.get(&canonical_triplet(a, b, c)).copied()
    }
    pub fn dihedral(&self, a: &str, b: &str, c: &str, d: &str) -> Option<AmberDihedral> {
        self.dihedrals
            .get(&canonical_key(&format!("{a}-{b}-{c}-{d}")))
            .copied()
    }
    pub fn improper(&self, a: &str, b: &str, c: &str, d: &str) -> Option<AmberDihedral> {
        self.impropers
            .get(&canonical_key(&format!("{a}-{b}-{c}-{d}")))
            .copied()
    }
    pub fn nonbonded(&self, atom_type: &str) -> Option<AmberNonbonded> {
        self.nonbonded.get(&canonical_key(atom_type)).copied()
    }
}

fn canonical_key(key: &str) -> String {
    key.replace('-', " ")
        .split_whitespace()
        .map(str::to_ascii_uppercase)
        .collect::<Vec<_>>()
        .join("-")
}
fn canonical_pair(a: &str, b: &str) -> String {
    if a.to_ascii_uppercase() <= b.to_ascii_uppercase() {
        canonical_key(&format!("{a}-{b}"))
    } else {
        canonical_key(&format!("{b}-{a}"))
    }
}
fn canonical_triplet(a: &str, b: &str, c: &str) -> String {
    canonical_key(&format!("{a}-{b}-{c}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_frcmod_sections_and_canonical_lookups() {
        let parm = "CT 12.01 1.7\nHC 1.008 1.2";
        let frcmod = "MASS\nXX 10.0\nBOND\nHC-CT 300.0 1.09\nANGL\nHC-CT-HC 40.0 109.5\nDIHE\nHC-CT-CT-HC 0.2 3.0 180.0\nIMPROPER\nC-N-CA-H 1.1 2.0 0.0\nNONBON\nCT 1.9080 0.1094";
        let index = AmberParameterIndex::from_sources(parm, frcmod).unwrap();
        assert_eq!(index.masses.len(), 3);
        assert_eq!(index.bond("CT", "HC").unwrap().length, 1.09);
        assert_eq!(index.angle("HC", "CT", "HC").unwrap().angle_deg, 109.5);
        assert_eq!(
            index.dihedral("HC", "CT", "CT", "HC").unwrap().periodicity,
            3.0
        );
        assert_eq!(index.improper("C", "N", "CA", "H").unwrap().pk, 1.1);
        assert_eq!(index.nonbonded("CT").unwrap().depth, 0.1094);
    }
}

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

        evaluate_half_pairs(system, forces, 332.0522)
    }
}
