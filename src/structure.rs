//! External structure ingestion — build an engine from an **in-memory**
//! molecular structure passed in from Python.
//!
//! Architecture boundary: the Rust side does **not** read files / Parquet.
//! Reading Parquet, PDB, mmCIF and cleaning the data is the Python side's job
//! (the caller's data prep); Python passes atom arrays into
//! `StructureInput` zero-copy through the PyO3 FFI (P4). This module converts
//! them to `MmCif` and runs the standard build pipeline (H placement → ff
//! types/charges → bond formation → solvation → minimization).
//!
//! This keeps `mutate.rs` lightweight: post-mutation structures are also
//! generated on the Python side and ingested here — Rust never does side-chain
//! modelling.

use std::collections::BTreeMap;
use std::str::FromStr;

use crate::engine::md_core::ComputationDevice;
use crate::engine::md_core::params::FfParamSet;
use bio_files::{AtomGeneric, ChainGeneric, MmCif, ResidueEnd, ResidueGeneric, ResidueType};
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

    // Group atoms by chain and residue so termini are assigned per chain.
    let mut by_res: BTreeMap<(String, i32), Vec<&AtomInput>> = BTreeMap::new();
    for a in &input.atoms {
        by_res
            .entry((a.chain_id.clone(), a.res_seq))
            .or_default()
            .push(a);
    }

    let mut mm_atoms: Vec<AtomGeneric> = Vec::with_capacity(input.atoms.len());
    let mut residues: Vec<ResidueGeneric> = Vec::with_capacity(by_res.len());
    let mut chain_atom_sns: Vec<u32> = Vec::with_capacity(input.atoms.len());
    let mut chain_res_sns: Vec<u32> = Vec::with_capacity(by_res.len());
    let mut next_serial: u32 = 1;

    // 结晶水（HOH/WAT/SOL/H2O/DOD）不能进蛋白 MD：水 O 的 type_in_res="O" 在
    // 氨基酸模板里没有 FF type → "Atom missing FF type"（1R2I 的 214 个 HOH 触发）。
    // 调用方应先脱去结晶水；这里引擎兜底跳过水残基（与 from_mmcif 一致），保证
    // StructureInput（from_atoms / Tauri GUI）路径也安全。
    let is_water =
        |a: &AtomInput| matches!(a.res_name.as_str(), "HOH" | "WAT" | "SOL" | "H2O" | "DOD");
    let kept_res: Vec<(&(String, i32), &Vec<&AtomInput>)> = by_res
        .iter()
        .filter(|(_, ra)| !ra.is_empty() && !is_water(ra[0]))
        .collect();
    if kept_res.is_empty() {
        return Err("StructureInput has no protein residues (all atoms are waters)".to_string());
    }
    let n_res = kept_res.len();
    for (res_idx, ((chain_id, _), res_atoms)) in kept_res.iter().enumerate() {
        let first_in_chain = res_idx == 0 || kept_res[res_idx - 1].0.0 != *chain_id;
        let last_in_chain = res_idx + 1 == n_res || kept_res[res_idx + 1].0.0 != *chain_id;
        let end = match (first_in_chain, last_in_chain) {
            (true, _) => ResidueEnd::NTerminus,
            (_, true) => ResidueEnd::CTerminus,
            _ => ResidueEnd::Internal,
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
    }

    let chains = {
        let mut chains = Vec::new();
        for (chain_id, _) in kept_res.iter().map(|(key, atoms)| (&key.0, atoms)) {
            if chains.iter().any(|c: &ChainGeneric| c.id == *chain_id) {
                continue;
            }
            let residue_sns: Vec<u32> = kept_res
                .iter()
                .filter(|(key, _)| key.0 == *chain_id)
                .map(|(key, _)| key.1 as u32 + 1)
                .collect();
            let atom_sns: Vec<u32> = residues
                .iter()
                .filter(|r| residue_sns.contains(&r.serial_number))
                .flat_map(|r| r.atom_sns.iter().copied())
                .collect();
            chains.push(ChainGeneric {
                id: chain_id.clone(),
                residue_sns,
                atom_sns,
            });
        }
        chains
    };

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
