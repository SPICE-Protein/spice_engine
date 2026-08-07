//! Protein topology: residue table, sequence, and Cα atom index mapping.
//!
//! Built from a *prepared* `MmCif` (i.e. after
//! `dynamics::params::prepare_peptide_mmcif` has assigned hydrogens,
//! force-field types and partial charges). The Cα indices are indices into
//! `MdState.atoms`, so coordinates can be read straight out of the engine.

use std::collections::HashMap;

use bio_files::ResidueType;
use na_seq::{AminoAcid, AtomTypeInRes, Element};

/// Map an amino-acid variant to its one-letter code.
pub fn one_letter(aa: AminoAcid) -> char {
    use AminoAcid::*;
    match aa {
        Arg => 'R',
        His => 'H',
        Lys => 'K',
        Asp => 'D',
        Glu => 'E',
        Ser => 'S',
        Thr => 'T',
        Asn => 'N',
        Gln => 'Q',
        Cys => 'C',
        Sec => 'U',
        Gly => 'G',
        Pro => 'P',
        Ala => 'A',
        Val => 'V',
        Ile => 'I',
        Leu => 'L',
        Met => 'M',
        Phe => 'F',
        Tyr => 'Y',
        Trp => 'W',
    }
}

/// A single residue in the topology table.
#[derive(Debug, Clone)]
pub struct ResidueInfo {
    /// Residue sequence id (from the mmCIF).
    pub seq_id: i32,
    /// One-letter amino-acid code.
    pub one_letter: char,
    /// Indices into `MdState.atoms` for every atom of this residue.
    pub atom_indices: Vec<usize>,
}

/// A lightweight view of the prepared protein that SPICE needs: the sequence,
/// residue→atom mapping, and the backbone/heavy-atom index sets used by the
/// physical metrics (Rg, DSSP-lite hydrogen bonds, clashes, surface charge).
#[derive(Debug, Clone, Default)]
pub struct ProteinTopology {
    /// One-letter sequence, in residue order.
    pub sequence: String,
    pub residues: Vec<ResidueInfo>,
    /// Indices into `MdState.atoms` for each Cα atom, aligned with `residues`.
    pub ca_indices: Vec<usize>,
    /// Backbone amide N indices, aligned with `residues`.
    pub n_indices: Vec<usize>,
    /// Backbone carbonyl C indices, aligned with `residues`.
    pub c_indices: Vec<usize>,
    /// Backbone carbonyl O indices, aligned with `residues`.
    pub o_indices: Vec<usize>,
    /// All protein (non-`hetero`) heavy (non-H) atom indices. Used for Rg,
    /// clash counting and surface detection.
    pub heavy_indices: Vec<usize>,
}

impl ProteinTopology {
    /// Build from a prepared `MmCif` (after `prepare_peptide_mmcif`).
    pub fn from_prepared(protein: &bio_files::MmCif) -> Result<Self, String> {
        let serial_to_idx: HashMap<u32, usize> = protein
            .atoms
            .iter()
            .enumerate()
            .map(|(i, a)| (a.serial_number, i))
            .collect();

        let mut ca_indices: Vec<usize> = Vec::new();
        let mut n_indices: Vec<usize> = Vec::new();
        let mut c_indices: Vec<usize> = Vec::new();
        let mut o_indices: Vec<usize> = Vec::new();
        let mut heavy_indices: Vec<usize> = Vec::new();

        for (i, a) in protein.atoms.iter().enumerate() {
            if a.hetero {
                continue;
            }
            match a.type_in_res {
                Some(AtomTypeInRes::CA) => ca_indices.push(i),
                Some(AtomTypeInRes::N) => n_indices.push(i),
                Some(AtomTypeInRes::C) => c_indices.push(i),
                Some(AtomTypeInRes::O) => o_indices.push(i),
                _ => {}
            }
            if a.element != Element::Hydrogen {
                heavy_indices.push(i);
            }
        }

        if ca_indices.is_empty() {
            return Err("no Cα atoms found — was the peptide prepared?".to_string());
        }

        let residues: Vec<ResidueInfo> = protein
            .residues
            .iter()
            .map(|r| {
                let one = match &r.res_type {
                    ResidueType::AminoAcid(aa) => one_letter(*aa),
                    ResidueType::Water => 'w',
                    ResidueType::Other(name) => {
                        name.chars().next().unwrap_or('X').to_ascii_uppercase()
                    }
                };
                let atom_indices = r
                    .atom_sns
                    .iter()
                    .filter_map(|sn| serial_to_idx.get(sn).copied())
                    .collect();
                ResidueInfo {
                    seq_id: r.serial_number as i32,
                    one_letter: one,
                    atom_indices,
                }
            })
            .collect();

        let sequence: String = residues.iter().map(|r| r.one_letter).collect();
        Ok(Self {
            sequence,
            residues,
            ca_indices,
            n_indices,
            c_indices,
            o_indices,
            heavy_indices,
        })
    }
}
