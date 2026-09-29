//! Data functionality for Forcefield params. Includes Amber parameters built in to binaries
//! which use this library, and can load params for other sets as required.
//!
//! Uses `bio_files` for the base data structures.

use std::{collections::HashMap, io, path::PathBuf};

use bio_files::{
    AtomGeneric, BondGeneric, ChainGeneric, LipidStandard, MmCif, ResidueEnd, ResidueGeneric,
    ResidueType, create_bonds,
    md_params::{
        ChargeParams, ChargeParamsProtein, ForceFieldParams, NucleotideTemplate,
        load_amino_charges, parse_lib_lipid, parse_lib_nucleic_acid, parse_lib_peptide,
    },
};
use na_seq::{AminoAcid, AminoAcidGeneral, AminoAcidProtenationVariant, AtomTypeInRes, Element};

use crate::{
    Dihedral, ParamError,
    add_hydrogens::ph::{resolve_his_tautomer_by_geometry, resolve_variant},
    merge_params, populate_hydrogens_dihedrals,
};

pub type ProtFfChargeMap = HashMap<AminoAcidGeneral, Vec<ChargeParamsProtein>>;
pub type LipidFfChargeMap = HashMap<LipidStandard, Vec<ChargeParams>>;
pub type NucleicAcidFfChargeMap = HashMap<NucleotideTemplate, Vec<ChargeParams>>;

// We include Amber parameter files with this package.
// Proteins and amino acids:
const PARM_19: &str = include_str!("data/parm19.dat"); // Bonded, and LJ
const FRCMOD_FF19SB: &str = include_str!("data/frcmod.ff19SB"); // Bonded, and LJ: overrides and new types
pub const AMINO_19: &str = include_str!("data/amino19.lib"); // Charge; internal residues
const AMINO_NT12: &str = include_str!("data/aminont12.lib"); // Charge; protonated N-terminus residues
const AMINO_CT12: &str = include_str!("data/aminoct12.lib"); // Charge; protonated C-terminus residues

// Ligands/small organic molecules: *General Amber Force Fields*.
const GAFF2: &str = include_str!("data/gaff2.dat");
// Lipids
const LIPID_21: &str = include_str!("data/lipid21.dat"); // Bonded and LJ

// Public, so we can use it for lipid templates.
pub const LIPID_21_LIB: &str = include_str!("data/lipid21.lib"); // Charge and FF names

// DNA (OL24) and RNA (OL3)
pub const OL24_LIB: &str = include_str!("data/ff-nucleic-OL24.lib");
const OL24_FRCMOD: &str = include_str!("data/ff-nucleic-OL24.frcmod");
// todo: frcmod.protonated_nucleic?
// RNA (I believe this is the OL3 Amber's FF page recommends?)
pub const RNA_LIB: &str = include_str!("data/RNA.lib");
// todo: RNA.YIL.lib? RNA_CI.lib? RNA_Shaw.lib? These are, I believe, "alternative" libraries,
// todo, and not required. YIL: Yildirim torsion refit. CI: Legacy Cornell-style. SHAW: incomplete,
// todo from a person named Shaw.

// Note: Water parameters are concise; we store them directly.

#[derive(Default, Debug)]
/// A set of general parameters that aren't molecule-specific. E.g. from GAFF2, OL3, RNA, or amino19.
/// These are used as a baseline, and in some cases, overridden by molecule-specific parameters.
pub struct FfParamSet {
    pub peptide: Option<ForceFieldParams>,
    pub small_mol: Option<ForceFieldParams>,
    pub dna: Option<ForceFieldParams>,
    pub rna: Option<ForceFieldParams>,
    pub lipids: Option<ForceFieldParams>,
    pub carbohydrates: Option<ForceFieldParams>,
    /// In addition to charge, this also contains the mapping of res type to FF type; required to map
    /// other parameters to protein atoms. E.g. from `amino19.lib`, and its N and C-terminus variants.
    pub peptide_ff_q_map: Option<ProtFfChargeMapSet>,
    pub lipid_ff_q_map: Option<LipidFfChargeMap>,
    // todo: QC these types; lipid as place holder. See how they parse.
    pub dna_ff_q_map: Option<NucleicAcidFfChargeMap>,
    pub rna_ff_q_map: Option<NucleicAcidFfChargeMap>,
}

/// Paths for to general parameter files. Used to create a FfParamSet.
#[derive(Clone, Debug, Default)]
pub struct ParamGeneralPaths {
    /// E.g. parm19.dat
    pub peptide: Option<PathBuf>,
    /// E.g. ff19sb.dat
    pub peptide_mod: Option<PathBuf>,
    /// E.g. amino19.lib
    pub peptide_ff_q: Option<PathBuf>,
    /// E.g. aminoct12.lib
    pub peptide_ff_q_c: Option<PathBuf>,
    /// E.g. aminont12.lib
    pub peptide_ff_q_n: Option<PathBuf>,
    /// e.g. gaff2.dat
    pub small_organic: Option<PathBuf>,
    /// e.g. ff-nucleic-OL24.lib
    pub dna: Option<PathBuf>,
    /// e.g. ff-nucleic-OL24.frcmod
    pub dna_mod: Option<PathBuf>,
    /// e.g. RNA.lib
    pub rna: Option<PathBuf>,
    pub lipid: Option<PathBuf>,
    pub carbohydrate: Option<PathBuf>,
}

impl FfParamSet {
    /// Load general parameter files for the most common classes of organic molecules.
    /// This also populates ff type and charge for protein atoms; these are provided by molecule-specific
    /// formats for small molecules.
    pub fn new(paths: &ParamGeneralPaths) -> io::Result<Self> {
        let mut result = FfParamSet::default();

        if let Some(p) = &paths.peptide {
            let peptide = ForceFieldParams::load_dat(p)?;

            if let Some(p_mod) = &paths.peptide_mod {
                let frcmod = ForceFieldParams::load_frcmod(p_mod)?;
                result.peptide = Some(merge_params(&peptide, &frcmod));
            } else {
                result.peptide = Some(peptide);
            }
        }

        let mut ff_map = ProtFfChargeMapSet::default();
        if let Some(p) = &paths.peptide_ff_q {
            ff_map.internal = load_amino_charges(p)?;
        }
        if let Some(p) = &paths.peptide_ff_q_c {
            ff_map.internal = load_amino_charges(p)?;
        }
        if let Some(p) = &paths.peptide_ff_q_n {
            ff_map.internal = load_amino_charges(p)?;
        }

        result.peptide_ff_q_map = Some(ff_map);

        if let Some(p) = &paths.small_organic {
            result.small_mol = Some(ForceFieldParams::load_dat(p)?);
        }

        if let Some(p) = &paths.dna {
            let peptide = ForceFieldParams::load_dat(p)?;

            if let Some(p_mod) = &paths.dna_mod {
                let frcmod = ForceFieldParams::load_frcmod(p_mod)?;
                result.dna = Some(merge_params(&peptide, &frcmod));
            } else {
                result.dna = Some(peptide);
            }
        }

        if let Some(p) = &paths.rna {
            result.rna = Some(ForceFieldParams::load_dat(p)?);
        }

        if let Some(p) = &paths.lipid {
            result.lipids = Some(ForceFieldParams::load_dat(p)?);
        }

        if let Some(p) = &paths.carbohydrate {
            result.carbohydrates = Some(ForceFieldParams::load_dat(p)?);
        }

        Ok(result)
    }

    /// Create a parameter set using Amber parameters included with this library. This uses
    /// the param sets recommended by Amber, CAO Sept 2025: ff19SB, OL24, OL3, GLYCAM_06j, lipids21,
    /// and gaff2.
    pub fn new_amber() -> io::Result<Self> {
        let mut result = FfParamSet::default();

        // We use parm19 for both peptides, and nucleic acids.
        let parm19 = ForceFieldParams::from_dat(PARM_19)?;

        let peptide_frcmod = ForceFieldParams::from_frcmod(FRCMOD_FF19SB)?;
        result.peptide = Some(merge_params(&parm19, &peptide_frcmod));

        {
            let internal = parse_lib_peptide(AMINO_19)?;
            let n_terminus = parse_lib_peptide(AMINO_NT12)?;
            let c_terminus = parse_lib_peptide(AMINO_CT12)?;

            result.peptide_ff_q_map = Some(ProtFfChargeMapSet {
                internal,
                n_terminus,
                c_terminus,
            });
        }

        let lipid_dat = ForceFieldParams::from_dat(LIPID_21)?;
        result.lipids = Some(lipid_dat);

        let lipid_charges = parse_lib_lipid(LIPID_21_LIB)?;
        result.lipid_ff_q_map = Some(lipid_charges);

        result.small_mol = Some(ForceFieldParams::from_dat(GAFF2)?);

        // todo: Load these, and get them working. They currently trigger a mass-parsing error.
        // todo: You must update your Lib parser in bio_files to handle this variant.

        let dna_frcmod = ForceFieldParams::from_frcmod(OL24_FRCMOD)?;
        result.dna = Some(merge_params(&parm19, &dna_frcmod));

        // todo: A/R
        result.rna = Some(parm19.clone());

        // todo: Currently hardcoded peptide/lipid versions for this lib parsing. Generalize?
        let dna_charges = parse_lib_nucleic_acid(OL24_LIB)?;
        result.dna_ff_q_map = Some(dna_charges);

        // todo: Currently hardcoded peptide/lipid versions for this lib parsing. Generalize?
        let rna_charges = parse_lib_nucleic_acid(RNA_LIB)?;
        result.rna_ff_q_map = Some(rna_charges);

        Ok(result)
    }
}

#[derive(Clone, Default, Debug)]
/// Maps type-in-residue (found in, e.g. mmCIF and PDB files) to Amber FF type, and partial charge.
/// We assume that if one of these is loaded, so are the others. So, these aren't `Options`s, but
/// the field that holds this struct should be one.
pub struct ProtFfChargeMapSet {
    pub internal: ProtFfChargeMap,
    pub n_terminus: ProtFfChargeMap,
    pub c_terminus: ProtFfChargeMap,
}

/// Populate forcefield type, and partial charge on atoms. This should be run on mmCIF
/// files prior to running molecular dynamics on them. These files from RCSB PDB do not
/// natively have this data.
///
/// `residues` must be the full set; this is relevant to how we index it.
pub fn populate_peptide_ff_and_q(
    atoms: &mut [AtomGeneric],
    residues: &[ResidueGeneric],
    ff_type_charge: &ProtFfChargeMapSet,
    disulfide_sg_sns: &std::collections::HashSet<u32>,
    ph: f32,
    custom_protonation: Option<&HashMap<usize, AminoAcidProtenationVariant>>,
) -> Result<(), ParamError> {
    // Tis is slower than if we had an index map already.
    let mut index_map = HashMap::new();
    for (i, atom) in atoms.iter().enumerate() {
        index_map.insert(atom.serial_number, i);
    }

    for (res_i, res) in residues.iter().enumerate() {
        // Resolve the protonation override or His tautomer at the residue level
        // to avoid mutable/immutable borrowing conflicts on `atoms`.
        let mut res_var_override = None;
        if let Some(map) = custom_protonation {
            if let Some(&var) = map.get(&res_i) {
                if !matches!(&res.res_type, ResidueType::AminoAcid(aa) if var.get_standard() == Some(*aa)) {
                    return Err(ParamError::new(&format!(
                        "Residue {res_i}: protonation variant {var} is incompatible with residue"
                    )));
                }
                res_var_override = Some(AminoAcidGeneral::Variant(var));
            }
        }

        if res_var_override.is_none() {
            if let ResidueType::AminoAcid(aa) = &res.res_type {
                let geometry = (*aa == AminoAcid::His)
                    .then(|| resolve_his_tautomer_by_geometry(atoms, res));
                // The pH ladder drives *internal* residues only. The terminal
                // charge libs (aminont12/aminoct12) ship no terminal variants
                // for LYN/CYM/ASH/GLH — a pH-derived variant on a terminal LYS
                // etc. had no unit to bind to and hard-failed the build — and
                // the H-addition pass already resolves terminal digit maps
                // from the internal lib only. His tautomers stay geometry-driven
                // everywhere: HID/HIE/HIP units exist in both terminal libs, and
                // neutral tautomers are the physically sane choice beside the
                // charged α-ammonium/α-carboxylate.
                let variant = if matches!(res.end, ResidueEnd::Internal) {
                    resolve_variant(*aa, ph, None, geometry)
                } else {
                    Ok(geometry)
                }
                .map_err(|e| ParamError::new(&format!("Residue {res_i}: {e}")))?;
                res_var_override = variant.map(AminoAcidGeneral::Variant);
            }
        }

        for sn in &res.atom_sns {
            let atom = match atoms.get_mut(index_map[sn]) {
                Some(a) => a,
                None => {
                    return Err(ParamError::new(&format!(
                        "Unable to populate Charge or FF type for atom {sn}"
                    )));
                }
            };

            if atom.hetero {
                continue;
            }

            let Some(type_in_res) = &atom.type_in_res else {
                return Err(ParamError::new(&format!(
                    "MD failure: Missing type in residue for atom: {atom}"
                )));
            };

            let ResidueType::AminoAcid(aa) = &res.res_type else {
                // e.g. solvent or other hetero atoms; skip.
                continue;
            };

            // todo: Eventually, determine how to load non-standard AA variants from files; set up your
            // todo state to use those labels. They are available in the params.
            // Disulfide-bonded Cys uses the CYX (oxidized) charge/type set, so its
            // SG gets the "S" ff type (and matching charge) instead of the thiol
            // "SH" — required for the S–S bond terms to resolve.
            let is_cyx = *aa == AminoAcid::Cys
                && res.atom_sns.iter().any(|sn| disulfide_sg_sns.contains(sn));
            let aa_gen = if is_cyx {
                AminoAcidGeneral::Variant(AminoAcidProtenationVariant::Cyx)
            } else {
                if let Some(vo) = res_var_override.clone() {
                    vo
                } else {
                    AminoAcidGeneral::Standard(*aa)
                }
            };

            let charge_map = match res.end {
                ResidueEnd::Internal => &ff_type_charge.internal,
                ResidueEnd::NTerminus => &ff_type_charge.n_terminus,
                ResidueEnd::CTerminus => &ff_type_charge.c_terminus,
                ResidueEnd::Hetero => {
                    return Err(ParamError::new(&format!(
                        "Error: Encountered hetero atom when parsing amino acid FF types: {atom}"
                    )));
                }
            };

            let charges = match charge_map.get(&aa_gen) {
                Some(c) => c,
                // A specific workaround to plain "HIS" being absent from amino19.lib (2025.
                // Choose one of "HID", "HIE", "HIP arbitrarily.
                // todo: Re-evaluate this, e.g. which one of the three to load.
                None if aa_gen == AminoAcidGeneral::Standard(AminoAcid::His) => charge_map
                    .get(&AminoAcidGeneral::Variant(AminoAcidProtenationVariant::Hid))
                    .ok_or_else(|| ParamError::new("Unable to find AA mapping"))?,
                None => {
                    return Err(ParamError::new(&format!(
                        "Unable to find AA mapping for {aa_gen:?} in {:?}",
                        res.end
                    )))
                }
            };

            let mut found = false;

            for charge in charges {
                // todo: Note that we have multiple branches in some case, due to Amber names like
                // todo: "HYP" for variants on AAs for different protenation states. Handle this.
                if charge.type_in_res == *type_in_res {
                    atom.force_field_type = Some(charge.ff_type.clone());
                    atom.partial_charge = Some(charge.charge);

                    // Deprotonated Tyr (pH > pKa): the H-addition pass omits
                    // the phenol HH (pH-filtered by the digit map) but that
                    // alone leaves the unit ~0.6 e shy of a real phenolate
                    // (糙1-residual (i)). Amber's phenolate model keeps
                    // ring/backbone charges and puts the full −1 on the unit
                    // via the oxygen, so shift the missing portion (q_HH − 1)
                    // onto OH. Internal residues only — termini keep standard
                    // protonation at any pH. Mirror of the engine tree fix.
                    if *aa == AminoAcid::Tyr
                        && matches!(res.end, ResidueEnd::Internal)
                        && ph > crate::add_hydrogens::ph::PKA_TYR
                        && matches!(type_in_res, AtomTypeInRes::OH)
                        && let Some(hh) = charges
                            .iter()
                            .find(|c| c.type_in_res == AtomTypeInRes::H("HH".to_string()))
                        && let Some(q) = &mut atom.partial_charge
                    {
                        *q += hh.charge - 1.0;
                    }

                    found = true;
                    break;
                }
            }

            // Code below is mainly for the case of missing data; otherwise, the logic for this operation
            // is complete.

            if !found {
                match type_in_res {
                    // todo: This is a workaround for having trouble with H types. LIkely
                    // todo when we create them. For now, this meets the intent.
                    AtomTypeInRes::H(_) => {
                        // todo: This is a workaround for the above; try other HIS variants.
                        // A HIS ring H can be named per a DIFFERENT protonation
                        // variant than the default one used for typing (e.g. an
                        // "HE2" H — HIE pattern — while the residue is typed via
                        // the HID variant at pH ~7). Try the sibling variant.
                        //
                        // NOTE: do NOT `break` out of the outer residue-atom loop
                        // here — `break` targets the nearest enclosing loop, which
                        // silently skips every subsequent atom of the residue
                        // (leaving them untyped → "Atom missing FF type"). Guard
                        // the remaining fallbacks with `!found` instead.
                        if aa_gen == AminoAcidGeneral::Standard(AminoAcid::His) {
                            let charges = charge_map
                                .get(&AminoAcidGeneral::Variant(AminoAcidProtenationVariant::Hie))
                                .ok_or_else(|| {
                                    ParamError::new("Unable to find AA mapping for HIE")
                                })?;

                            // todo: You may need HIP too, even with this workaround.
                            // todo: DRY

                            for charge in charges {
                                if charge.type_in_res == *type_in_res {
                                    atom.force_field_type = Some(charge.ff_type.clone());
                                    atom.partial_charge = Some(charge.charge);

                                    found = true;
                                    break;
                                }
                            }
                        }

                        // The amber template doesn't have HH23; only 2 Hs on that. I believe
                        // this may be an omission.
                        if !found
                            && aa_gen == AminoAcidGeneral::Standard(AminoAcid::Arg)
                            && *type_in_res == AtomTypeInRes::H("HH23".to_owned())
                        {
                            for charge in charges {
                                if charge.type_in_res == AtomTypeInRes::H("HH22".to_string()) {
                                    atom.force_field_type = Some(charge.ff_type.clone());
                                    atom.partial_charge = Some(charge.charge);

                                    found = true;
                                    break;
                                }
                            }
                        }

                        // N-terminus: the deposited/structure H on the backbone N is
                        // usually named "H" (amide style), but the N-term charge map
                        // (aminont12) expects H1/H2/H3 for the NH3+ ammonium. Map the
                        // first one to H1 so it types as a proper N-term amine H
                        // instead of the generic "HB2" (a β-carbon H type).
                        if !found
                            && res.end == ResidueEnd::NTerminus
                            && *type_in_res == AtomTypeInRes::H("H".to_owned())
                        {
                            for charge in charges {
                                if charge.type_in_res == AtomTypeInRes::H("H1".to_string()) {
                                    atom.force_field_type = Some(charge.ff_type.clone());
                                    atom.partial_charge = Some(charge.charge);

                                    found = true;
                                    break;
                                }
                            }
                        }

                        // Generic H fallback (only if the variant workarounds above
                        // did not match). We've witnessed unmatched H types due to
                        // errors in mmCIF files, e.g. on ASP #88 on 9GLS.
                        if !found {
                            eprintln!(
                                "Error assigning FF type and q based on atom type in res: Failed to match H type. Res #{}, Atom #{}, {type_in_res}, {aa_gen:?}. \
                             Falling back to a generic H",
                                res.serial_number, atom.serial_number,
                            );

                            for charge in charges {
                                if charge.type_in_res == AtomTypeInRes::H("H".to_string())
                                    || charge.type_in_res == AtomTypeInRes::H("HA".to_string())
                                {
                                    atom.force_field_type = Some("HB2".to_string());
                                    atom.partial_charge = Some(charge.charge);

                                    found = true;
                                    break;
                                }
                            }
                        }
                    }
                    _ => (),
                }

                // i.e. if still not found after our specific workarounds above.
                if !found {
                    eprintln!("Problem populating FF or Q: {}", atom);
                    continue;
                }
            }
        }
    }

    Ok(())
}

/// Combines several functions that should be run after loading protein files from PDB. Add hydrogens,
/// load force field parameters and partial charge, and add bonds.
pub fn prepare_peptide(
    atoms: &mut Vec<AtomGeneric>,
    bonds: &mut Vec<BondGeneric>,
    residues: &mut Vec<ResidueGeneric>,
    chains: &mut [ChainGeneric],
    ff_map: &ProtFfChargeMapSet,
    ph: f32,
    custom_protonation: Option<&HashMap<usize, AminoAcidProtenationVariant>>,
) -> Result<Vec<Dihedral>, ParamError> {
    let mut dihedrals = Vec::new();

    dihedrals =
        populate_hydrogens_dihedrals(atoms, residues, chains, ff_map, ph, custom_protonation, 0.0)?;

    let disulfide_sg_sns = crate::add_hydrogens::find_disulfide_sgs(atoms, 0.0);

    // todo: Similar checks for empty etc.
    populate_peptide_ff_and_q(
        atoms,
        residues,
        ff_map,
        &disulfide_sg_sns,
        ph,
        custom_protonation,
    )?;

    if bonds.is_empty() {
        *bonds = create_bonds(atoms);
    }

    Ok(dihedrals)
}

/// Remove alternate-conformer (altloc) duplicates from an `MmCif`.
///
/// mmCIF/PDB files represent alternative conformations as separate atoms with
/// the SAME atom name inside the SAME residue (e.g. residue ARG 14 has `CA`
/// with altloc "A" and "B"). Keeping both produces two atoms typed `XC` (Cα)
/// in one residue, and the spurious `XC-N-XC` angle then fails the FF lookup
/// (`Missing valence angle params for XC-N-XC`).
/// Standard PDB practice: keep only the highest-occupancy conformer (ties →
/// lowest serial number / first-listed) and drop the rest, fixing residue and
/// chain atom references. This also drops the duplicate side-chain atoms of the
/// discarded conformer (they share atom names with the kept one).
pub fn dedup_altloc(mol: &mut MmCif) {
    // Map serial -> chain id so alternate conformers are keyed per chain
    // (residue serial numbers repeat across chains).
    let mut sn_to_chain: HashMap<u32, String> = HashMap::new();
    for ch in &mol.chains {
        for &sn in &ch.atom_sns {
            sn_to_chain.insert(sn, ch.id.clone());
        }
    }

    // Atom name helper (type_in_res preferred, else general name / element).
    let name_of = |a: &AtomGeneric| -> String {
        a.type_in_res
            .as_ref()
            .map(|t| t.to_string())
            .or_else(|| a.type_in_res_general.clone())
            .unwrap_or_else(|| format!("{:?}", a.element))
    };

    // For each (chain, residue, atom name): keep the best conformer.
    #[derive(Clone, Copy)]
    struct Best {
        sn: u32,
        occ: f32,
    }
    let mut best: HashMap<(String, u32, String), Best> = HashMap::new();
    for r in &mol.residues {
        for &sn in &r.atom_sns {
            let Some(a) = mol.atoms.iter().find(|a| a.serial_number == sn) else {
                continue;
            };
            let key = (
                sn_to_chain.get(&sn).cloned().unwrap_or_default(),
                r.serial_number,
                name_of(a),
            );
            let occ = a.occupancy.unwrap_or(0.0);
            // Existing entry wins if it has strictly higher occupancy, or equal
            // occupancy with a lower serial number (first-listed conformer).
            if !best
                .get(&key)
                .is_some_and(|b| b.occ > occ || (b.occ == occ && b.sn < sn))
            {
                best.insert(key, Best { sn, occ });
            }
        }
    }

    let before = mol.atoms.len();
    let keep: std::collections::HashSet<u32> = best.values().map(|b| b.sn).collect();
    mol.atoms.retain(|a| keep.contains(&a.serial_number));
    for r in &mut mol.residues {
        r.atom_sns.retain(|sn| keep.contains(sn));
    }
    mol.residues.retain(|r| !r.atom_sns.is_empty());
    let keep_res: std::collections::HashSet<u32> =
        mol.residues.iter().map(|r| r.serial_number).collect();
    for ch in &mut mol.chains {
        ch.atom_sns.retain(|sn| keep.contains(sn));
        ch.residue_sns.retain(|sn| keep_res.contains(sn));
    }
    mol.chains.retain(|c| !c.atom_sns.is_empty());

    if mol.atoms.len() < before {
        eprintln!(
            "dedup_altloc: dropped {} alternate-conformer atoms ({} -> {})",
            before - mol.atoms.len(),
            before,
            mol.atoms.len()
        );
    }
}

// Mirror-sync note: the incomplete-residue policy below intentionally matches
// the engine tree (`src/engine/forcefield/params.rs`, 糙1-residual (iii)
// upgrade) — per-residue charge-lib sidechain heavy-atom set, not merely
// "zero sidechain atoms". Keep both copies in lockstep until the trees merge.

/// Sidechain heavy atoms the charge lib defines for `aa` (everything in its
/// internal unit except backbone N/CA/C/O, terminal OXT, and hydrogens).
/// Protonation variants (ASH/GLH/CYM/CYX/His tautomers) never differ in HEAVY
/// atoms, so the plain standard internal unit is the right reference for any
/// protonation state; termini likewise (the (ii) OXT completion handles the
/// only heavy-atom terminal difference, and it runs before this on rebuilt
/// mols — here we exclude OXT from expectations anyway and check after it).
fn expected_sidechain_heavies(
    ff_map: &ProtFfChargeMapSet,
    aa: &AminoAcid,
) -> Option<Vec<AtomTypeInRes>> {
    let unit = match ff_map.internal.get(&AminoAcidGeneral::Standard(*aa)) {
        Some(u) => u,
        None => {
            // e.g. His: amino19 ships only HID/HIE/HIP units (no plain HIS
            // key). Variants/tautomers of one residue differ ONLY in H
            // placement — heavy-atom sidechain names are identical — so any
            // variant unit whose standard parent is `aa` is a valid reference.
            let mut found = None;
            for (k, u) in &ff_map.internal {
                if let AminoAcidGeneral::Variant(v) = k
                    && v.get_standard() == Some(*aa)
                {
                    found = Some(u);
                    break;
                }
            }
            found?
        }
    };
    let mut out: Vec<AtomTypeInRes> = Vec::new();
    for cp in unit {
        let t = &cp.type_in_res;
        if matches!(t, AtomTypeInRes::H(_))
            || matches!(
                t,
                AtomTypeInRes::N
                    | AtomTypeInRes::CA
                    | AtomTypeInRes::C
                    | AtomTypeInRes::O
                    | AtomTypeInRes::OXT
            )
        {
            continue;
        }
        out.push(t.clone());
    }
    Some(out)
}

/// Residues whose charge-lib SIDECHAIN heavy atoms are not all present
/// (disordered/truncated crystal sidechains — 糙1-residual (iii)). The old
/// check only caught *backbone-only* residues (zero sidechain heavies), which
/// let partially truncated ones through (e.g. 4LPX His@95/96/97: CB present,
/// entire imidazole missing) → silent −0.2 e charge holes and fractional
/// system net charge. Selenomethionine alias: SE stands in for SD.
/// Returns (chain id, residue serial, 3-letter name, missing atom names).
fn find_incomplete_residues(
    mol: &MmCif,
    ff_map: &ProtFfChargeMapSet,
) -> Vec<(String, u32, String, Vec<String>)> {
    let mut sn_to_chain: HashMap<u32, String> = HashMap::new();
    for ch in &mol.chains {
        for &sn in &ch.atom_sns {
            sn_to_chain.insert(sn, ch.id.clone());
        }
    }
    let mut out = Vec::new();
    for r in &mol.residues {
        let ResidueType::AminoAcid(aa) = &r.res_type else {
            continue;
        };
        let Some(expected) = expected_sidechain_heavies(ff_map, aa) else {
            continue;
        };
        if expected.is_empty() {
            continue; // Gly (and anything the lib defines sidechain-free)
        }
        let present: Vec<AtomTypeInRes> = r
            .atom_sns
            .iter()
            .filter_map(|sn| mol.atoms.iter().find(|a| &a.serial_number == sn))
            .filter(|a| a.element != Element::Hydrogen && !a.hetero)
            .filter_map(|a| a.type_in_res.clone())
            .collect();
        let has = |t: &AtomTypeInRes| {
            present.contains(t)
                || (*t == AtomTypeInRes::SD && present.contains(&AtomTypeInRes::SE))
        };
        let missing: Vec<String> = expected
            .iter()
            .filter(|t| !has(t))
            .map(|t| format!("{t:?}"))
            .collect();
        if !missing.is_empty() {
            let chain = sn_to_chain
                .get(&r.serial_number)
                .cloned()
                .unwrap_or_default();
            let name = aa.to_str(na_seq::AaIdent::ThreeLetters).to_string();
            out.push((chain, r.serial_number, name, missing));
        }
    }
    out
}

/// See docs on `prepare_peptide`. This is a convenience variant that uses an `MmCif` file.
///
/// `strict_incomplete`: reject structures with residues missing any charge-lib
/// sidechain heavy atom (disordered/truncated crystal sidechains; per-residue
/// check, also catches "only CB survived"). Default policy is strict (`true`):
/// a truncated residue silently corrupts the physics, so the caller should
/// filter/repair it upstream. Set `false` to build them with just the atoms
/// present (warning).
pub fn prepare_peptide_mmcif(
    mol: &mut MmCif,
    ff_map: &ProtFfChargeMapSet,
    ph: f32,
    custom_protonation: Option<&HashMap<usize, AminoAcidProtenationVariant>>,
    strict_incomplete: bool,
    // Fraction of detected disulfide bridges to chemically REDUCE at build
    // time (0.0 = fully oxidized, the historical behavior). See
    // `find_disulfide_sgs`.
    reducing_fraction: f32,
) -> Result<(Vec<BondGeneric>, Vec<Dihedral>), ParamError> {
    let mut dihedrals = Vec::new();

    // Collapse alternate conformations (altloc A/B duplicates) before any
    // hydrogen addition, FF typing, or bond inference. This is the single choke
    // point for BOTH the mmCIF path (`Structure.from_mmcif`) and the in-memory
    // path (`Structure.from_atoms` / parquet), which both funnel through here.
    dedup_altloc(mol);

    // Drop hetero atoms (water, ions, ligands) up front. `MdState::new` filters
    // `!hetero` atoms for peptides, so bonds must be created on the same atom set
    // or the serial-number lookup in `build_adjacency_list` fails.
    let hetero_sns: std::collections::HashSet<u32> = mol
        .atoms
        .iter()
        .filter(|a| a.hetero)
        .map(|a| a.serial_number)
        .collect();
    if !hetero_sns.is_empty() {
        mol.atoms.retain(|a| !a.hetero);
        // Drop residues/water whose atoms were all removed; fix refs of the rest.
        mol.residues
            .retain(|r| r.atom_sns.iter().any(|sn| !hetero_sns.contains(sn)));
        for res in &mut mol.residues {
            res.atom_sns.retain(|sn| !hetero_sns.contains(sn));
        }
        // Update chain atom/residue references.
        let keep_res: std::collections::HashSet<u32> =
            mol.residues.iter().map(|r| r.serial_number).collect();
        for ch in &mut mol.chains {
            ch.atom_sns.retain(|sn| !hetero_sns.contains(sn));
            ch.residue_sns.retain(|sn| keep_res.contains(sn));
        }
    }

    // Reject/flag residues whose charge-lib SIDECHAIN heavy atoms are not all
    // present (disordered/truncated crystal sidechains). The old version only
    // caught backbone-only residues (zero sidechain heavies) and let partially
    // truncated ones through (e.g. 4LPX His@95: CB present, imidazole missing)
    // → silent charge holes and fractional net charge (糙1-residual (iii)).
    // Gly is exempt (the lib defines no sidechain atoms). Strict (default):
    // fail with a per-residue missing-atom list so the caller can filter/
    // repair upstream. Lenient: warn and build with whatever atoms exist.
    let incomplete = find_incomplete_residues(mol, ff_map);
    if !incomplete.is_empty() {
        let list = incomplete
            .iter()
            .map(|(c, s, n, miss)| {
                format!("{n} (chain {c}, res {s}) missing [{}]", miss.join(", "))
            })
            .collect::<Vec<_>>()
            .join("; ");
        if strict_incomplete {
            return Err(ParamError::new(&format!(
                "Incomplete structure: {} residue(s) missing sidechain heavy atoms defined by the charge lib: {list}. \
                 Filter/repair upstream, or rebuild with strict_incomplete=false to degrade (builds truncated).",
                incomplete.len()
            )));
        }
        eprintln!(
            "WARNING: {} residue(s) missing sidechain heavy atoms; building with atoms present: {list}",
            incomplete.len()
        );
    }

    dihedrals = populate_hydrogens_dihedrals(
        &mut mol.atoms,
        &mut mol.residues,
        &mut mol.chains,
        ff_map,
        ph,
        custom_protonation,
        reducing_fraction,
    )?;

    // Detect disulfide bridges (SG–SG < 2.4 Å) — selects the CYX (oxidized Cys)
    // charge/type set so bridged SGs get the "S" ff type, not thiol "SH".
    let disulfide_sg_sns = crate::add_hydrogens::find_disulfide_sgs(&mol.atoms, reducing_fraction);

    // todo: Similar checks for empty etc.
    populate_peptide_ff_and_q(
        &mut mol.atoms,
        &mol.residues,
        ff_map,
        &disulfide_sg_sns,
        ph,
        custom_protonation,
    )?;

    let bonds = create_bonds(&mol.atoms);
    // Distance-based bond inference creates spurious cross-residue bonds in
    // folded proteins (e.g. CB of residue i close to NH1 of residue i+3).
    // Keep only physically sensible bonds: intra-residue, backbone peptide
    // bonds C(i)-N(i+1), and Cys-Cys disulfides.
    let bonds = filter_protein_bonds(&mol.atoms, &mol.residues, bonds, &disulfide_sg_sns);
    // `create_bonds` has no S–S bond spec, so disulfide bridges are never
    // distance-inferred. Add them explicitly (filter keeps SG–SG bonds).
    let bonds = add_disulfide_bonds(&mol.atoms, &mol.residues, bonds, &disulfide_sg_sns);

    Ok((bonds, dihedrals))
}

/// Filter distance-inferred bonds to a physically sensible protein bond set.
///
/// `create_bonds` infers bonds from inter-atomic distance + element pairs, which
/// is fine for isolated small molecules but wrong for folded proteins: two atoms
/// from distant residues can sit within covalent-bond distance and get a spurious
/// "bond". We therefore keep only:
///   1. intra-residue bonds (both atoms in the same residue),
///   2. backbone peptide bonds C(i)-N(i+1) between consecutive residues,
///   3. Cys-Cys disulfide bridges (SG-SG).
/// `create_bonds` has no S–S bond spec, so disulfide bridges are never
/// distance-inferred. Add SG–SG bonds for cysteine pairs within disulfide-bond
/// distance (< 2.4 Å), so the bridge is bonded (excluded from nonbonded, correct
/// geometry) and the FF applies S–S terms.
fn add_disulfide_bonds(
    atoms: &[AtomGeneric],
    _residues: &[ResidueGeneric],
    mut bonds: Vec<BondGeneric>,
    retained_sg_sns: &std::collections::HashSet<u32>,
) -> Vec<BondGeneric> {
    use AtomTypeInRes::SG;
    let is_sg = |a: &AtomGeneric| matches!(a.type_in_res, Some(SG));
    let sg: Vec<usize> = atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| is_sg(a))
        .map(|(i, _)| i)
        .collect();
    let existing: std::collections::HashSet<(u32, u32)> = bonds
        .iter()
        .map(|b| (b.atom_0_sn.min(b.atom_1_sn), b.atom_0_sn.max(b.atom_1_sn)))
        .collect();
    for a in 0..sg.len() {
        for b in (a + 1)..sg.len() {
            let (i, j) = (sg[a], sg[b]);
            let d = (atoms[i].posit - atoms[j].posit).magnitude();
            if d < 2.4 {
                let key = (
                    atoms[i].serial_number.min(atoms[j].serial_number),
                    atoms[i].serial_number.max(atoms[j].serial_number),
                );
                // Reduced bridges are absent from the retained set: they get
                // no bond record and their CYS stay protonated thiols.
                if !retained_sg_sns.contains(&key.0) || !retained_sg_sns.contains(&key.1) {
                    continue;
                }
                if !existing.contains(&key) {
                    bonds.push(BondGeneric {
                        bond_type: bio_files::BondType::Single,
                        atom_0_sn: atoms[i].serial_number,
                        atom_1_sn: atoms[j].serial_number,
                    });
                }
            }
        }
    }
    bonds
}

fn filter_protein_bonds(
    atoms: &[AtomGeneric],
    residues: &[ResidueGeneric],
    bonds: Vec<BondGeneric>,
    retained_sg_sns: &std::collections::HashSet<u32>,
) -> Vec<BondGeneric> {
    use AtomTypeInRes::*;

    let mut atom_idx: HashMap<u32, usize> = HashMap::new();
    for (i, a) in atoms.iter().enumerate() {
        atom_idx.insert(a.serial_number, i);
    }
    let mut res_by_sn: HashMap<u32, usize> = HashMap::new();
    for (ri, res) in residues.iter().enumerate() {
        for sn in &res.atom_sns {
            res_by_sn.insert(*sn, ri);
        }
    }
    // Peptide order (residues may be stored in any order).
    let mut order: Vec<usize> = (0..residues.len()).collect();
    order.sort_by_key(|&i| residues[i].serial_number);
    let mut res_pos: HashMap<usize, usize> = HashMap::new();
    for (pos, &ri) in order.iter().enumerate() {
        res_pos.insert(ri, pos);
    }

    let is_sg = |a: &AtomGeneric| matches!(a.type_in_res, Some(SG));

    bonds
        .into_iter()
        .filter(|b| {
            let (Some(&i0), Some(&i1)) = (atom_idx.get(&b.atom_0_sn), atom_idx.get(&b.atom_1_sn))
            else {
                return false;
            };
            let (Some(&r0), Some(&r1)) = (res_by_sn.get(&b.atom_0_sn), res_by_sn.get(&b.atom_1_sn))
            else {
                return false;
            };

            if r0 == r1 {
                return true; // intra-residue
            }

            // Cross-residue bonds must be a retained disulfide or a backbone
            // peptide bond; SG–SG bonds of *reduced* bridges drop out here.
            if is_sg(&atoms[i0]) && is_sg(&atoms[i1]) {
                let a = atoms[i0].serial_number;
                let b = atoms[i1].serial_number;
                return retained_sg_sns.contains(&a) && retained_sg_sns.contains(&b);
            }

            let (Some(&p0), Some(&p1)) = (res_pos.get(&r0), res_pos.get(&r1)) else {
                return false;
            };
            if p0.abs_diff(p1) != 1 {
                return false;
            }
            let (earlier, later) = if p0 < p1 { (r0, r1) } else { (r1, r0) };
            if residues[earlier].end == ResidueEnd::CTerminus
                || residues[later].end == ResidueEnd::NTerminus
            {
                return false; // chain break
            }
            let (c_idx, n_idx) = if p0 < p1 { (i0, i1) } else { (i1, i0) };
            atoms[c_idx].type_in_res == Some(C) && atoms[n_idx].type_in_res == Some(N)
        })
        .collect()
}
