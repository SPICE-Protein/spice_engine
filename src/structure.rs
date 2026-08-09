//! External structure ingestion — build an engine from an **in-memory**
//! molecular structure passed in from Python.
//!
//! Architecture boundary: the Rust side does **not** read files / Parquet.
//! Reading Parquet, PDB, mmCIF and cleaning the data is the Python side's job
//! (the `spice_protein/` pipeline); Python passes atom arrays into
//! `StructureInput` zero-copy through the PyO3 FFI (P4). This module converts
//! them to `MmCif` and runs the standard build pipeline (H placement → ff
//! types/charges → bond formation → solvation → minimization).
//!
//! This keeps `mutate.rs` lightweight: post-mutation structures are also
//! generated on the Python side and ingested here — Rust never does side-chain
//! modelling.

use std::collections::BTreeMap;
use std::str::FromStr;

use bio_files::{AtomGeneric, ChainGeneric, MmCif, ResidueEnd, ResidueGeneric, ResidueType};
use dynamics::params::FfParamSet;
use dynamics::ComputationDevice;
use na_seq::{AtomTypeInRes, Element};

use crate::builder::{BuildOptions, build_system};
use crate::engine::SpiceEngine;

/// An atom; fields align with the `atoms_*.parquet` dataset columns (in-memory transfer).
#[derive(Debug, Clone)]
pub struct AtomInput {
    pub chain_id: String,
    /// Residue sequence number (for grouping; need not be contiguous).
    pub res_seq: i32,
    /// Three-letter residue name, e.g. `"ALA"`.
    pub res_name: String,
    /// Atom name, e.g. `"CA"`, `"N"`, `"O"`, `"CB"` (case-sensitive).
    pub atom_name: String,
    pub element: Element,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    /// Crystallographic occupancy. CRITICAL for alternate-conformer
    /// (altloc) structures: `dedup_altloc` in the dynamics fork keeps the
    /// HIGHEST-occupancy conformer per (chain, residue, atom name). If this is
    /// dropped (defaults to 1.0 for every altloc), the tie falls back to the
    /// lowest serial number, which can be a different, overlapping conformer
    /// -> hard non-bonded clash that blows up MD. The parquet path MUST pass
    /// this through.
    pub occupancy: f32,
}

impl AtomInput {
    /// Convenient constructor for FFI callers.
    pub fn new(
        chain_id: impl Into<String>,
        res_seq: i32,
        res_name: impl Into<String>,
        atom_name: impl Into<String>,
        element: Element,
        x: f32,
        y: f32,
        z: f32,
    ) -> Self {
        Self {
            chain_id: chain_id.into(),
            res_seq,
            res_name: res_name.into(),
            atom_name: atom_name.into(),
            element,
            x,
            y,
            z,
            occupancy: 1.0,
        }
    }
}

/// A complete structure, generated on the Python side (heavy atoms suffice; H placement is Rust's job).
#[derive(Debug, Clone, Default)]
pub struct StructureInput {
    pub atoms: Vec<AtomInput>,
}

impl StructureInput {
    pub fn push(&mut self, a: AtomInput) {
        self.atoms.push(a);
    }

    /// Number of distinct residues, grouped by `res_seq`.
    pub fn residue_count(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for a in &self.atoms {
            seen.insert(a.res_seq);
        }
        seen.len()
    }

    /// Infer the one-letter sequence from three-letter residue names (ascending `res_seq`).
    pub fn sequence(&self) -> Result<String, String> {
        let mut map: BTreeMap<i32, char> = BTreeMap::new();
        for a in &self.atoms {
            if let Some(ch) = one_letter_from_resname(&a.res_name) {
                map.entry(a.res_seq).or_insert(ch);
            }
        }
        if map.is_empty() {
            return Err("no amino-acid residues found in StructureInput".to_string());
        }
        Ok(map.values().collect())
    }
}

/// Three-letter residue name → one-letter (standard 20 + selenocysteine U).
pub fn one_letter_from_resname(name: &str) -> Option<char> {
    let up = name.trim().to_uppercase();
    let ch = match up.as_str() {
        "ALA" => 'A',
        "ARG" => 'R',
        "ASN" => 'N',
        "ASP" => 'D',
        "CYS" => 'C',
        "GLN" => 'Q',
        "GLU" => 'E',
        "GLY" => 'G',
        "HIS" => 'H',
        "ILE" => 'I',
        "LEU" => 'L',
        "LYS" => 'K',
        "MET" => 'M',
        "PHE" => 'F',
        "PRO" => 'P',
        "SER" => 'S',
        "THR" => 'T',
        "TRP" => 'W',
        "TYR" => 'Y',
        "VAL" => 'V',
        "SEC" => 'U',
        _ => return None,
    };
    Some(ch)
}

/// Convert an in-memory structure to an `MmCif` (heavy atoms only; hydrogens and
/// ff types are assigned later by `prepare_peptide_mmcif`).
pub fn atoms_to_mmcif(input: &StructureInput) -> Result<MmCif, String> {
    if input.atoms.is_empty() {
        return Err("StructureInput has no atoms".to_string());
    }

    // Group atoms by residue (sorted by res_seq). Keep each residue's atoms in
    // input order.
    let mut by_res: BTreeMap<i32, Vec<&AtomInput>> = BTreeMap::new();
    for a in &input.atoms {
        by_res.entry(a.res_seq).or_default().push(a);
    }

    let mut mm_atoms: Vec<AtomGeneric> = Vec::with_capacity(input.atoms.len());
    let mut residues: Vec<ResidueGeneric> = Vec::with_capacity(by_res.len());
    let mut chain_atom_sns: Vec<u32> = Vec::with_capacity(input.atoms.len());
    let mut chain_res_sns: Vec<u32> = Vec::with_capacity(by_res.len());
    let mut next_serial: u32 = 1;

    let n_res = by_res.len();
    for (res_idx, (res_seq, res_atoms)) in by_res.iter().enumerate() {
        // Residue end tagging determines which charge map (internal/n/c-term) is used.
        let end = if res_idx == 0 {
            ResidueEnd::NTerminus
        } else if res_idx + 1 == n_res {
            ResidueEnd::CTerminus
        } else {
            ResidueEnd::Internal
        };

        let res_type = ResidueType::from_str(res_atoms[0].res_name.as_str());
        let res_sn = res_idx as u32 + 1;
        let mut atom_sns: Vec<u32> = Vec::with_capacity(res_atoms.len());

        for a in res_atoms.iter() {
            let sn = next_serial;
            next_serial += 1;
            atom_sns.push(sn);
            chain_atom_sns.push(sn);

            // `type_in_res` is a best-effort enum parse; the string name is kept
            // in `type_in_res_general` (used by H placement / ff typing).
            let type_in_res = AtomTypeInRes::from_str(&a.atom_name).ok();
            mm_atoms.push(AtomGeneric {
                serial_number: sn,
                posit: lin_alg::f64::Vec3::new(a.x as f64, a.y as f64, a.z as f64),
                element: a.element,
                type_in_res,
                type_in_res_general: Some(a.atom_name.clone()),
                force_field_type: None,
                partial_charge: None,
                hetero: false,
                occupancy: Some(a.occupancy),
                alt_conformation_id: None,
            });
        }

        chain_res_sns.push(res_sn);
        residues.push(ResidueGeneric {
            serial_number: res_sn,
            res_type,
            atom_sns,
            end,
        });

        // (res_seq retained only for grouping; MmCif uses internal serial numbers)
        let _ = res_seq;
    }

    let chains = vec![ChainGeneric {
        id: input
            .atoms
            .first()
            .map(|a| a.chain_id.clone())
            .unwrap_or_else(|| "A".to_string()),
        residue_sns: chain_res_sns,
        atom_sns: chain_atom_sns,
    }];

    Ok(MmCif {
        ident: "structure-input".to_string(),
        metadata: Default::default(),
        atoms: mm_atoms,
        chains,
        residues,
        secondary_structure: vec![],
        experimental_method: None,
    })
}

/// Build an engine directly from an in-memory structure (the P4 FFI entry point).
pub fn build_from_input(
    dev: &ComputationDevice,
    param_set: &FfParamSet,
    input: &StructureInput,
    opts: &BuildOptions,
) -> Result<SpiceEngine, String> {
    let protein = atoms_to_mmcif(input)?;
    build_system(dev, param_set, protein, opts)
}
