//! Handles adding hydrogen based on geometry (See also aa_coords/mod.rs, which this calls),
//! and assigns H types that map to Amber params. Note that in addition to using these types to
//! assign FF params, we can also use them to QC which H atoms should be present on specific
//! parents in each AA. (It has helped us catch several errors, like extra Hs in Proline and Trp rings.)
//!
//! todo: Handle differnet protenation states, and assign atom-types in a way that's
//! , todo for a given residue, consistent with a single protenation state. The current approach
//! is an innacurate hybrid.
//!
//! This page has atom labels for all AAs; use it as a ref and QC: https://ccpn.ac.uk/manual/v3/NEFAtomNames.html

use std::collections::{HashMap, HashSet};

use bio_files::{AtomGeneric, ChainGeneric, ResidueEnd, ResidueGeneric, ResidueType};
use na_seq::{AminoAcid, AminoAcidGeneral, AminoAcidProtenationVariant, AtomTypeInRes, Element};

use crate::engine::md_core::{
    ParamError,
    add_hydrogens::{
        add_hydrogens_2::{Dihedral, aa_data_from_coords, planar_posit},
        bond_vecs::{LEN_CP_O, init_local_bond_vecs},
        ph::{
            PKA_TYR, his_choice, resolve_his_tautomer_by_geometry, resolve_variant,
            standard_allowed_at_ph, variant_allowed_at_ph,
        },
    },
    params::{ProtFfChargeMap, ProtFfChargeMapSet},
};

pub(crate) mod add_hydrogens_2;
pub mod bond_vecs;
pub(crate) mod ph;
mod sidechain;

// We use the normal AA, vice general form here, as that's the one available in the mmCIF files
// we're parsing. This is despite the Amber data we are using for the source using the general versions.
pub type DigitMap = HashMap<AminoAcid, HashMap<char, Vec<u8>>>;
// pub type DigitMap = HashMap<AminoAcidGeneral, HashMap<char, Vec<u8>>>;
// todo: Perhaps we use General here, since that's what we need to adjust protenation based on pH.

/// We use this to validate H atom type assignments. We derive this directly from `amino19.lib` (Amber)
/// Returns `true` if valid.
/// Note that this does not ensure completeness of the H set for a given AA; only if a given
/// value is valid for that AA.
/// h_num=0 means it's just "HE" or similar.
///
/// We use the `digit_map` vice the `ff_map` directly, so we can merge protenation variants, e.g. for  His.
fn validate_h_atom_type(
    depth: char,
    digit: u8,
    aa: AminoAcid,
    digit_map: &DigitMap,
) -> Result<bool, ParamError> {
    let data = digit_map.get(&aa).ok_or_else(|| {
        ParamError::new(&format!(
            "No parm19_data entry for amino acid {:?}",
            AminoAcidGeneral::Standard(aa)
        ))
    })?;

    let data_this_depth = data.get(&depth).ok_or_else(|| {
        ParamError::new(&format!(
            "No parm19_data entry for amino acid (Depth) {:?}",
            AminoAcidGeneral::Standard(aa)
        ))
    })?;
    if data_this_depth.contains(&digit) {
        return Ok(true);
    }

    Ok(false)
}

/// The COMPLETE set of hydrogens the charge lib expects on `parent_tir` of amino
/// acid `aa`, in naming order — i.e. `h_type_in_res_sidechain(0..)` until the
/// first `None`. This is the authoritative H *count* for a residue position:
/// crystal bond angles (sp3 C-C-C runs 111–117°) cannot tell a distorted CH2
/// from a planar CH, and geometry-guessed counts silently dropped second H's
/// (missing HB3/HG3 → fractional, under-counted net charge). Callers place
/// exactly `len()` H's; a count of 0 means "no H here" (carbonyl C, ring-fusion
/// C, deprotonated position at this pH — the protonation filtering in
/// [`make_h_digit_map_custom`] is what makes the 0 meaningful).
pub(crate) fn h_expect(
    aa: AminoAcid,
    parent_tir: &AtomTypeInRes,
    digit_map: &DigitMap,
) -> Result<Vec<AtomTypeInRes>, ParamError> {
    let mut out = Vec::new();
    for i in 0..4 {
        match h_type_in_res_sidechain(i, parent_tir, Some(aa), digit_map)? {
            Some(t) => out.push(t),
            None => break,
        }
    }
    Ok(out)
}

// todo: Include N and C terminus maps A/R.
pub(crate) fn make_h_digit_map_custom(
    ff_map: &ProtFfChargeMap,
    ph: f32,
    custom_variant: Option<AminoAcidProtenationVariant>,
) -> DigitMap {
    let mut result: DigitMap = HashMap::new();

    // Preselect a single HIS state at this pH so we don't mix HID/HIE/HIP digits.
    let his_selected = if let Some(v) = custom_variant {
        if matches!(
            v,
            AminoAcidProtenationVariant::Hid
                | AminoAcidProtenationVariant::Hie
                | AminoAcidProtenationVariant::Hip
        ) {
            Some(v)
        } else {
            his_choice(ph)
        }
    } else {
        his_choice(ph)
    };

    for (&aa_gen, params) in ff_map {
        // Filter by pH/custom variant:
        let allowed = match aa_gen {
            AminoAcidGeneral::Standard(aa) => {
                if let Some(v) = custom_variant {
                    match (aa, v) {
                        (AminoAcid::Asp, AminoAcidProtenationVariant::Ash) => false,
                        (AminoAcid::Glu, AminoAcidProtenationVariant::Glh) => false,
                        (AminoAcid::Cys, AminoAcidProtenationVariant::Cym) => false,
                        (AminoAcid::Lys, AminoAcidProtenationVariant::Lyn) => false,
                        (AminoAcid::His, _) => false,
                        _ => true,
                    }
                } else {
                    standard_allowed_at_ph(aa, ph)
                }
            }
            AminoAcidGeneral::Variant(v) => {
                if let Some(cv) = custom_variant {
                    cv == v
                } else {
                    if matches!(
                        v,
                        AminoAcidProtenationVariant::Hid
                            | AminoAcidProtenationVariant::Hie
                            | AminoAcidProtenationVariant::Hip
                    ) {
                        matches!(his_selected, Some(sel) if sel == v)
                    } else {
                        variant_allowed_at_ph(v, ph)
                    }
                }
            }
        };
        if !allowed {
            continue;
        }

        let mut per_heavy: HashMap<char, Vec<u8>> = HashMap::new();

        for cp in params {
            let tir = &cp.type_in_res; // adjust accessor as needed

            match tir {
                AtomTypeInRes::H(name) => {
                    // Split:  H  <designator-char>  <digits...>
                    let mut chars = name.chars();
                    chars.next(); // discard the leading 'H'

                    // Heavy-atom designator is always a single alphabetic char
                    let designator = match chars.next() {
                        Some(c) if c.is_ascii_alphabetic() => c,
                        _ => continue, // malformed – ignore
                    };

                    // Collect *all* trailing digits (handles "11", "21", ...)
                    let digits: String = chars.filter(|c| c.is_ascii_digit()).collect();
                    if digits.is_empty() {
                        // We will handle this designation appropriately downstream. For "HG", for example.
                        per_heavy.entry(designator).or_default().push(0);
                    } else {
                        // Safe because Amber never goes beyond two digits
                        let num: u8 = digits.parse().unwrap();
                        per_heavy.entry(designator).or_default().push(num);
                    }
                }
                // We only care about hydrogens that *do* carry a numeric suffix
                _ => (),
            }
        }

        if per_heavy.is_empty() {
            continue;
        }

        let aa = match aa_gen {
            AminoAcidGeneral::Standard(a) => a,
            AminoAcidGeneral::Variant(av) => av.get_standard().unwrap(), // todo: Unwrap OK?
        };

        // Make the relationship deterministic (ordinal 0 → smallest digit, …)
        for v in per_heavy.values_mut() {
            v.sort_unstable();
            v.dedup();
        }

        // This combines entries in the case of duplicates: This happens in the case of protenation
        // variants, like HIE and HID for HIS.
        if let Some(existing) = result.get_mut(&aa) {
            for (designator, mut digits) in per_heavy {
                existing.entry(designator).or_default().append(&mut digits);
            }
            for v in existing.values_mut() {
                v.sort_unstable();
                v.dedup();
            }
        } else {
            result.insert(aa, per_heavy);
        }
    }

    // Override for His, to combine

    // Tyr phenol OH deprotonates above its pKa (~10.5): remove the 'H' depth entry
    // (which holds HH for the OH parent) so h_type_in_res_sidechain returns None for
    // Tyr OH at high pH.  Ring H atoms use depths 'D', 'E', 'Z', 'B' and are unaffected.
    // (amino19.lib has no TYM variant, so we handle this here directly.)
    if ph > PKA_TYR {
        if let Some(tyr_map) = result.get_mut(&AminoAcid::Tyr) {
            tyr_map.remove(&'H');
        }
    }

    result
}

/// Assign atom-type-in-res for hydrogen atoms in polypeptides. This is not for small molecules,
/// which use GAFF types, nor generally required for them: Files for those tend to include H atoms,
/// while mmCIF and PDF files for proteins generally don't.
///
/// This function is for sidechain only; Backbone H are always "H" for on N, and "HA", "HA2", or "HA3"
/// for on Cα (The latter two for the case of Glycine only, which has no sidechain).
///
/// `neighbors` is atoms bonded to the atom the H is bonded to ?
/// Reference `amino19.lib`, which shows which atom-in-res types we should expect (including)
/// for these H atoms.
///
/// We need to correctly populate these atom-in-res types, to properly assign Amber FF type, and
/// partial charge downstream.
///
/// Example. For Asp, we should have one each of "H", "HA", "HB2", and "HB3".
///
/// `h_num_this_parent` increments from 0. We use a table to map these to digits, e.g. 0 and 1 might mean the
/// `2` and `3` in "HB2" and "HB3". Increments for a given parent that has multiple H.
/// Assigns the numerical value in the result, e.g. the "2" in "NE2". `parent_depth` provides the letter
/// e.g. the "D" in "HD1". (WHere "H" means Hydrogen, and "1" means the first hydrogen attached to this parent.
///
/// This can also be used for hetero atoms, or for that matter, ligands.
///
/// Returns None if the H should be absent due to protonation state. (?)
pub(crate) fn h_type_in_res_sidechain(
    h_num_this_parent: usize,
    parent_tir: &AtomTypeInRes,
    aa: Option<AminoAcid>, // None for hetero/ligand.
    h_digit_map: &DigitMap,
) -> Result<Option<AtomTypeInRes>, ParamError> {
    let Some(aa) = aa else {
        // Hetero. We can determine the naming scheme directly from the parent.
        let val = match parent_tir {
            AtomTypeInRes::Hetero(name_parent) => {
                // if parent looks like "C<digits>" (e.g. "C23"), drop the "C" and append the H‑index
                let mut chars = name_parent.chars();
                let elem = chars.next().unwrap(); // the leading letter, e.g. 'C' or 'O'
                let rest: String = chars.collect(); // the trailing digits, e.g. "23" or "5"

                if elem == 'C' && rest.chars().all(|c| c.is_ascii_digit()) {
                    // C23 → H231, H232, … depending on h_num_this_parent
                    let idx = h_num_this_parent + 1;
                    format!("H{}{}", rest, idx)
                } else {
                    // everything else → just prefix with "H", so "O5" → "HO5"
                    format!("H{}", name_parent)
                }
            }
            _ => {
                return Err(ParamError::new(
                    "Error assigning H type: Non-hetero parent, but missing AA.",
                ));
            }
        };

        return Ok(Some(AtomTypeInRes::Hetero(val)));
    };

    // todo: Assign the number based on parent type as well??
    let depth = match parent_tir {
        AtomTypeInRes::CB => 'B',
        AtomTypeInRes::CD | AtomTypeInRes::CD1 | AtomTypeInRes::CD2 => 'D',
        AtomTypeInRes::CE | AtomTypeInRes::CE1 | AtomTypeInRes::CE2 | AtomTypeInRes::CE3 => 'E',
        AtomTypeInRes::CG | AtomTypeInRes::CG1 | AtomTypeInRes::CG2 => 'G',
        AtomTypeInRes::CH2 | AtomTypeInRes::CH3 => 'H',
        AtomTypeInRes::CZ | AtomTypeInRes::CZ1 | AtomTypeInRes::CZ2 | AtomTypeInRes::CZ3 => 'Z',
        AtomTypeInRes::OD1 | AtomTypeInRes::OD2 => 'D',
        AtomTypeInRes::OG | AtomTypeInRes::OG1 | AtomTypeInRes::OG2 => 'G',
        AtomTypeInRes::OH => 'H',
        AtomTypeInRes::OE1 | AtomTypeInRes::OE2 => 'E',
        AtomTypeInRes::ND1 | AtomTypeInRes::ND2 => 'D',
        AtomTypeInRes::NH1 | AtomTypeInRes::NH2 => 'H',
        AtomTypeInRes::NE | AtomTypeInRes::NE1 | AtomTypeInRes::NE2 => 'E',
        AtomTypeInRes::NZ => 'Z',
        AtomTypeInRes::SE => 'E',
        AtomTypeInRes::SG => 'G',
        AtomTypeInRes::OXT => 'X', // todo: What should this be? Observed in glycine at the C terminus.
        _ => {
            return Err(ParamError::new(&format!(
                "Invalid parent type in res on H assignment. AA: {aa}. {parent_tir:?}",
            )));
        }
    };

    // Manual overrides here. Perhaps a more general algorithm will prevent needing these.
    // The naive approach of always applying 21 incremented to Cx2 doesn't always work,
    // so these individual overrides may be the easiest approach.
    // todo: See teh pattern here? Put in a mechanism to add the 2 prefix.
    // NOTE: each override MUST return None once the parent's lib capacity is
    // reached — h_expect counts expected H's by calling through i = 0.. and
    // stopping at the first None; an unbounded override fabricates phantom
    // atoms (HD3 on Phe ring CHs, HH23 inflating Arg NH2 to 3 H = +0.448 e).
    match aa {
        AminoAcid::Thr => {
            if *parent_tir == AtomTypeInRes::CG2 {
                // HG21, 22, 23 — methyl capacity 3.
                if h_num_this_parent >= 3 {
                    return Ok(None);
                }
                let digit = h_num_this_parent + 21;
                return Ok(Some(AtomTypeInRes::H(format!("HG{digit}"))));
            }
        }
        AminoAcid::Arg => {
            if *parent_tir == AtomTypeInRes::NH2 {
                // HH21, HH22 — guanidinium NH2 capacity 2 (NH1 falls through to
                // the leading-digit grouping below, [11,12]).
                if h_num_this_parent >= 2 {
                    return Ok(None);
                }
                let digit = h_num_this_parent + 21;
                return Ok(Some(AtomTypeInRes::H(format!("HH{digit}"))));
            }
        }
        AminoAcid::Phe => match parent_tir {
            AtomTypeInRes::CD2 | AtomTypeInRes::CE2 => {
                // Exactly one ring H on this CH carbon.
                if h_num_this_parent >= 1 {
                    return Ok(None);
                }
                let d = if matches!(parent_tir, AtomTypeInRes::CD2) {
                    'D'
                } else {
                    'E'
                };
                let prefix = format!("H{d}");
                return Ok(Some(AtomTypeInRes::H(format!("{prefix}2"))));
            }
            _ => (),
        },
        AminoAcid::Leu => {
            if *parent_tir == AtomTypeInRes::CD2 {
                // HD21, 22, 23 — methyl capacity 3.
                if h_num_this_parent >= 3 {
                    return Ok(None);
                }
                let digit = h_num_this_parent + 21;
                return Ok(Some(AtomTypeInRes::H(format!("HD{digit}"))));
            }
        }
        AminoAcid::Ile => {
            if *parent_tir == AtomTypeInRes::CG2 {
                // HG21, 22, 23 — methyl capacity 3.
                if h_num_this_parent >= 3 {
                    return Ok(None);
                }
                let digit = h_num_this_parent + 21;
                return Ok(Some(AtomTypeInRes::H(format!("HG{digit}"))));
            }
        }
        // OE1 is a carbonyl oxygen on GLN (HE21/HE22 belong to NE2, not OE1).
        // Without this, depth 'E' → [21,22] from the digit map causes HE21 to be
        // placed on OE1, producing a bogus C-O-H bond and a missing valence angle.
        AminoAcid::Gln if matches!(parent_tir, AtomTypeInRes::OE1 | AtomTypeInRes::OE2) => {
            return Ok(None);
        }
        // Same issue for ASN: OD1 is carbonyl; HD21/HD22 belong to ND2.
        AminoAcid::Asn if matches!(parent_tir, AtomTypeInRes::OD1 | AtomTypeInRes::OD2) => {
            return Ok(None);
        }
        _ => (),
    }

    let Some(digits_this_aa) = h_digit_map.get(&aa) else {
        return Err(ParamError::new(&format!(
            "Missing AA {aa} in digits map, which has {:?}",
            h_digit_map.keys()
        )));
    };

    let Some(digits) = digits_this_aa.get(&depth) else {
        // return Err(ParamError::new(&format!(
        //     "Missing H digits: Depth: {depth} not in {digits_this_aa:?} - {parent_tir:?} , {aa}",
        // )));
        return Ok(None);
    };

    let digit = {
        // Use Debug so variants like CE3/CZ3 carry the numeral
        let suffix_digit = format!("{:?}", parent_tir)
            .chars()
            .rev()
            .find(|c| c.is_ascii_digit())
            .and_then(|c| c.to_digit(10))
            .map(|d| d as usize);

        if let Some(sd) = suffix_digit {
            if let Some(pos) = digits.iter().position(|&d| d == sd as u8) {
                // Exact match (CE3 → 3, CZ2 → 2, Ile CG1 → "HG1" …): the digit
                // identifies the PARENT, so such a parent carries exactly one H.
                // (A count-driven caller looping h_num would otherwise re-receive
                // the same name forever.)
                if h_num_this_parent == 0 {
                    digits[pos]
                } else {
                    return Ok(None);
                }
            } else if digits.iter().all(|&d| d < 10) {
                // All H names at this depth are single-digit (e.g. HD1, HD2). The parent's
                // suffix is absent, meaning this H doesn't exist in the current protonation
                // state — e.g. ND1 (suffix=1) has no HD1 in HIE whose 'D'→[2].
                return Ok(None);
            } else {
                // Multi-digit H families (HG11/12/13 vs HG21/22/23 on CG1/CG2): the
                // parent's suffix selects the LEADING decimal digit, then attachment
                // order walks within that group. (Walking the unfiltered vec — the old
                // behavior — named both Val methyls HG11/12/13.)
                let group: Vec<u8> = digits
                    .iter()
                    .copied()
                    .filter(|&d| d >= 10 && (d / 10) as usize == sd)
                    .collect();
                match group.get(h_num_this_parent) {
                    Some(&d) => d,
                    None => return Ok(None),
                }
            }
        } else {
            // No numeric suffix on parent (e.g., OG, ND, etc.) → use attachment
            // order; running off the end of the lib's digit list honestly means
            // "this parent carries no more H" (returning the last entry again —
            // the old wrap — fabricates duplicate names under a count-driven loop).
            match digits.get(h_num_this_parent) {
                Some(&d) => d,
                None => return Ok(None),
            }
        }
    };

    // todo: Handle the N term and C term cases; pass those params in?

    // todo: Consider adding a completeness validator for the AA, ensuring all expected
    // todo: Hs are present.

    let val = if digit == 0 {
        format!("H{depth}") // e.g. HG. We use 0 as a flag when building the map.
    } else {
        format!("H{depth}{digit}")
    };

    let result = AtomTypeInRes::H(val);

    if !validate_h_atom_type(depth, digit, aa, h_digit_map)? {
        return Err(ParamError::new(&format!(
            "Invalid H type: {result} on {aa}. Parent: {parent_tir}"
        )));
    }

    Ok(Some(result))
}

/// Adds hydrogens to a molecule, and populdates residue dihedral angles.
/// This is useful in particular for mmCIF files from RCSB PDB, as they don't have these.
/// Uses Amber (or similar)-provided parameters as a guide.
///
/// Returns dihedrals.
/// todo: This needs to add bonds too!
/// Resolve hard clashes introduced by idealized H placement.
///
/// The H-placement uses ideal per-residue geometry without checking the rest of
/// the (folded) protein, so in tightly packed regions an added H can sit ~1 Å from
/// a non-bonded atom of another residue and blow the system up within a few MD
/// steps. Same-residue atoms are excluded from the check: they are 1-2/1-3 bonded
/// (or template-adjacent) and legitimately close.
///
/// Deleting the clashing H was the original remedy — but that silently removes
/// CHARGE-BEARING hydrogens (a buried Arg's guanidinium HH11/HH22 at +0.448 e
/// each, a Cys CB-H, …), leaving the solute net charge fractional and
/// systematically under-counted (2LYZ: 7 deletions → +4.8 e vs ~+7 e true),
/// which corrupts counterion balancing and every downstream electrostatic.
/// The remedy is therefore now three-phase:
///
/// 1. Keep if the H is ≥ [`CLASH_DIST`] from every foreign atom (the bulk case).
/// 2. **Reposition**: rigid-bond cone search — keep the parent–H distance,
///    scan the placement direction on the cone around the original bond vector
///    (polar angle ascending, so the least-distorted clearing wins).
/// 3. **Local relax**: if the cone misses, hill-climb the H on the parent sphere
///    (tangent pseudo-repulsion from nearby foreign atoms) from the cone's best
///    point.
///
/// Only if phases 2–3 cannot reach `CLASH_DIST` is the H removed — and that now
/// prints a named warning (residue, atom type, best distance reached).
///
/// Returns the number of H's removed (0 on healthy systems).
fn resolve_h_clashes(
    atoms: &mut Vec<AtomGeneric>,
    residues: &mut [ResidueGeneric],
    chains: &mut [ChainGeneric],
    new_sn_start: u32,
) -> usize {
    const CLASH_DIST: f64 = 1.2; // Å — clearly-overlapping non-bonded contact
    const TARGET_DIST: f64 = 1.3; // Å — search early-exit margin above the threshold
    const PARENT_DIST: f64 = 1.55; // Å — longest heavy-H bond (S-H ~1.35 Å)
    const NEAR_WINDOW: f64 = 4.5; // Å — foreign atoms relevant to a ~1 Å H excursion

    // serial -> residue index, and residue index -> set of atom serials
    let mut res_of: HashMap<u32, usize> = HashMap::new();
    let mut res_atoms: Vec<HashSet<u32>> = Vec::with_capacity(residues.len());
    for (ri, res) in residues.iter().enumerate() {
        let set: HashSet<u32> = res.atom_sns.iter().copied().collect();
        for sn in &set {
            res_of.insert(*sn, ri);
        }
        res_atoms.push(set);
    }

    // serial -> index for fast lookup
    let mut idx_of: HashMap<u32, usize> = HashMap::new();
    for (i, a) in atoms.iter().enumerate() {
        idx_of.insert(a.serial_number, i);
    }

    let h_serials: Vec<u32> = atoms
        .iter()
        // Only HYDROGENS are removal candidates: added heavy atoms (the C-term
        // OXT completion below) are chemistry-critical — they can never be
        // deleted for a clash, only relaxed around by the build minimizer.
        .filter(|a| a.serial_number >= new_sn_start && a.element == Element::Hydrogen)
        .map(|a| a.serial_number)
        .collect();

    let mut to_remove: HashSet<u32> = HashSet::new();
    let mut removed_detail: Vec<String> = Vec::new();
    let mut repositioned = 0usize;

    for &sn in &h_serials {
        let Some(&hi) = idx_of.get(&sn) else {
            continue;
        };
        let h_pos = atoms[hi].posit;

        // parent = nearest heavy atom within PARENT_DIST
        let mut parent_j: Option<usize> = None;
        let mut best = PARENT_DIST;
        for (j, a) in atoms.iter().enumerate() {
            if j == hi || a.element == Element::Hydrogen {
                continue;
            }
            let d = (a.posit - h_pos).magnitude();
            if d < best {
                best = d;
                parent_j = Some(j);
            }
        }
        let Some(pj) = parent_j else { continue };
        let psn = atoms[pj].serial_number;
        let Some(&pri) = res_of.get(&psn) else {
            continue;
        };
        let excluded = &res_atoms[pri];

        // min distance to a non-excluded atom
        let mut min_d = f64::INFINITY;
        for (j, a) in atoms.iter().enumerate() {
            if j == hi || excluded.contains(&a.serial_number) {
                continue;
            }
            let d = (a.posit - h_pos).magnitude();
            if d < min_d {
                min_d = d;
            }
        }
        if min_d >= CLASH_DIST {
            continue; // phase 1: no clash
        }

        // Phases 2–3: move the H on the sphere of radius r0 around its parent
        // (bond length preserved exactly, so bonded terms and any bond
        // inference downstream see an undistorted bond; only angles bend).
        let p_pos = atoms[pj].posit;
        let r0 = (h_pos - p_pos).magnitude();
        let name_of = |a: &AtomGeneric| -> String {
            match a.type_in_res_general.as_deref() {
                Some(s) if !s.is_empty() => s.to_string(),
                _ => match &a.type_in_res {
                    Some(t) => format!("{t:?}"),
                    None => format!("serial {sn}"),
                },
            }
        };
        if r0 < 1e-3 {
            to_remove.insert(sn);
            removed_detail.push(format!(
                "{} in {:?}@{} (degenerate bond placement)",
                name_of(&atoms[hi]),
                residues[pri].res_type,
                residues[pri].serial_number
            ));
            continue;
        }
        let d0 = (h_pos - p_pos).to_normalized();
        // Orthonormal tangent basis (u, v) ⊥ d0, deterministic.
        let arb = if d0.x.abs() < 0.9 {
            lin_alg::f64::Vec3::new(1.0, 0.0, 0.0)
        } else {
            lin_alg::f64::Vec3::new(0.0, 1.0, 0.0)
        };
        let u = arb.cross(d0).to_normalized();
        let v = d0.cross(u);
        // Foreign atoms near the original spot — the excursion is at most ~1 Å,
        // so a 4.5 Å window cannot miss a sub-CLASH_DIST contact.
        let near: Vec<usize> = atoms
            .iter()
            .enumerate()
            .filter(|(j, a)| {
                *j != hi
                    && !excluded.contains(&a.serial_number)
                    && (a.posit - h_pos).magnitude() < NEAR_WINDOW
            })
            .map(|(j, _)| j)
            .collect();
        let score = |dir: lin_alg::f64::Vec3| -> f64 {
            let pos = p_pos + dir * r0;
            let mut m = f64::INFINITY;
            for &j in &near {
                let d = (atoms[j].posit - pos).magnitude();
                if d < m {
                    m = d;
                }
            }
            m
        };

        // Phase 2: cone search, smallest distortion first.
        let mut best_dir = d0;
        let mut best_d = min_d;
        'cone: for &theta in &[15.0f64, 30.0, 45.0, 60.0] {
            let (ct, st) = (theta.to_radians().cos(), theta.to_radians().sin());
            for k in 0..12 {
                let phi = k as f64 * std::f64::consts::FRAC_PI_6;
                let dir = d0 * ct + (u * phi.cos() + v * phi.sin()) * st;
                let d = score(dir);
                if d > best_d {
                    best_d = d;
                    best_dir = dir;
                }
                if best_d >= TARGET_DIST {
                    break 'cone;
                }
            }
        }
        // Phase 3: hill-climb the min-distance on the sphere (only downhill
        // attempts shrink the step; strictly accepting improvements keeps this
        // deterministic).
        if best_d < TARGET_DIST {
            let mut dir = best_dir;
            let mut step = 0.4f64;
            for _ in 0..64 {
                let pos = p_pos + dir * r0;
                let mut f = lin_alg::f64::Vec3::new(0.0, 0.0, 0.0);
                for &j in &near {
                    let dv = pos - atoms[j].posit;
                    let d = dv.magnitude();
                    if d < 2.2 && d > 1e-6 {
                        f += dv.to_normalized() * (1.0 - d / 2.2);
                    }
                }
                let t = (f - dir * f.dot(dir)).to_normalized();
                if t.magnitude_squared() < 1e-12 {
                    break;
                }
                dir = (dir + t * step).to_normalized();
                let d = score(dir);
                if d > best_d {
                    best_d = d;
                    best_dir = dir;
                    if best_d >= TARGET_DIST {
                        break;
                    }
                } else {
                    step *= 0.8;
                }
            }
        }

        if best_d >= CLASH_DIST {
            atoms[hi].posit = p_pos + best_dir * r0;
            repositioned += 1;
        } else {
            // Last resort — and it is now LOUD: a removed H is a removed charge.
            to_remove.insert(sn);
            removed_detail.push(format!(
                "{} in {:?}@{} (best {:.2} A after reposition+relax)",
                name_of(&atoms[hi]),
                residues[pri].res_type,
                residues[pri].serial_number,
                best_d
            ));
        }
    }

    // Remove marked atoms (descending index), clean residue/chain refs.
    let mut idxs: Vec<usize> = atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| to_remove.contains(&a.serial_number))
        .map(|(i, _)| i)
        .collect();
    idxs.sort_unstable_by(|a, b| b.cmp(a));
    for i in idxs {
        atoms.remove(i);
    }
    for res in residues.iter_mut() {
        res.atom_sns.retain(|sn| !to_remove.contains(sn));
    }
    for ch in chains.iter_mut() {
        ch.atom_sns.retain(|sn| !to_remove.contains(sn));
    }

    let n = to_remove.len();
    if repositioned > 0 || n > 0 {
        eprintln!(
            "H clash resolution: {repositioned} repositioned on parent bond sphere, \
             {n} removed (could not clear {CLASH_DIST} A)."
        );
    }
    for d in &removed_detail {
        eprintln!("  WARNING: removed charge-bearing H: {d}");
    }
    n
}

/// Find Cys SG atoms that form a disulfide bridge (SG–SG distance < 2.4 Å).
/// Used to prevent protonating bridged cysteines during H placement.
/// Serial numbers of SG atoms that remain in the *oxidized* (CYX) state
/// under a redox environment of `reducing_fraction`.
///
/// Bridges are detected as before (SG–SG < 2.4 Å). `reducing_fraction` ∈
/// [0,1]: a seeded deterministic shuffle picks ceil(f·n_bridges) bridges to
/// REDUCE; their SGs are dropped from the set, so those cysteines take the
/// standard CYS unit — thiol "SH" ff type, an HG hydrogen, and (via
/// `add_disulfide_bonds`/`filter_protein_bonds` in the params trees) no S–S
/// bond record. f=0 is the historical all-oxidized behavior bit for bit.
/// Reduction keeps the two SG positions as found (≈2 Å apart); the following
/// solvation-relax/minimize pass is what separates them — a reduced bridge
/// is a *starting* condition for reduction physics, not an equilibrated one.
pub fn find_disulfide_sgs(atoms: &[AtomGeneric], reducing_fraction: f32) -> HashSet<u32> {
    let sg: Vec<usize> = atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| matches!(&a.type_in_res, Some(AtomTypeInRes::SG)))
        .map(|(i, _)| i)
        .collect();
    let mut bridges: Vec<(u32, u32)> = Vec::new();
    for a in 0..sg.len() {
        for b in (a + 1)..sg.len() {
            let d = (atoms[sg[a]].posit - atoms[sg[b]].posit).magnitude();
            if d < 2.4 {
                bridges.push((
                    atoms[sg[a]].serial_number.min(atoms[sg[b]].serial_number),
                    atoms[sg[a]].serial_number.max(atoms[sg[b]].serial_number),
                ));
            }
        }
    }
    let mut out = HashSet::new();
    let reducing_fraction = reducing_fraction.clamp(0.0, 1.0);
    if reducing_fraction <= 0.0 {
        for &(a, b) in &bridges {
            out.insert(a);
            out.insert(b);
        }
        return out;
    }
    let mut order: Vec<usize> = (0..bridges.len()).collect();
    {
        use rand::SeedableRng;
        use rand::seq::SliceRandom;
        // Fixed seed: same protein + same fraction ⇒ same reduced bridges,
        // matching the engine's reproducible-build policy for placement-time
        // decisions (unlike the per-build-entropy thermo RNG).
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5EED_1018);
        order.shuffle(&mut rng);
    }
    let n_reduce = (reducing_fraction * bridges.len() as f32).ceil() as usize;
    let reduce: HashSet<(u32, u32)> = order
        .iter()
        .copied()
        .take(n_reduce)
        .map(|i| bridges[i])
        .collect();
    for (idx, &(a, b)) in bridges.iter().enumerate() {
        if reduce.contains(&(a, b)) {
            continue;
        }
        out.insert(a);
        out.insert(b);
        let _ = idx;
    }
    if !bridges.is_empty() {
        eprintln!(
            "[redox] reducing {}/{} disulfide bridge(s) (fraction {reducing_fraction}).",
            reduce.len(),
            bridges.len()
        );
    }
    out
}

pub fn populate_hydrogens_dihedrals(
    atoms: &mut Vec<AtomGeneric>,
    residues: &mut [ResidueGeneric],
    chains: &mut [ChainGeneric],
    ff_map: &ProtFfChargeMapSet,
    ph: f32,
    custom_protonation: Option<&HashMap<usize, AminoAcidProtenationVariant>>,
    reducing_fraction: f32,
) -> Result<Vec<Dihedral>, ParamError> {
    // todo: Move this fn to this module? Split this and its diehdral component, or not?

    // Rebuild protein hydrogens from the selected pH state. Deposited hydrogen
    // names frequently encode a different titration state; retaining them would
    // create duplicates or leave stale HIP/HID/CYM/LYN atoms. Heavy atoms and
    // coordinates are preserved, making repeated preparation idempotent.
    let removed_h: HashSet<u32> = atoms
        .iter()
        .filter(|a| a.element == Element::Hydrogen && !a.hetero)
        .map(|a| a.serial_number)
        .collect();
    if !removed_h.is_empty() {
        atoms.retain(|a| !removed_h.contains(&a.serial_number));
        for res in residues.iter_mut() {
            res.atom_sns.retain(|sn| !removed_h.contains(sn));
        }
        for chain in chains.iter_mut() {
            chain.atom_sns.retain(|sn| !removed_h.contains(sn));
        }
    }

    // Sets up write-once static muts.
    init_local_bond_vecs(); // This is a bit hacky, and only needs to be run once.

    let mut index_map = HashMap::new();
    for (i, atom) in atoms.iter().enumerate() {
        index_map.insert(atom.serial_number, i);
    }

    // Detect Cys-Cys disulfide bridges (SG–SG < 2.4 Å) so the hydrogen
    // placement below does not protonate bonded SG atoms — otherwise every
    // disulfide S gets a thiol H clashing inside the bridge (huge forces).
    let disulfide_sg_sns = find_disulfide_sgs(atoms, reducing_fraction);

    let mut dihedrals = Vec::with_capacity(residues.len());

    let mut prev_cp_ca = None;

    let res_len = residues.len();

    // todo: The Clone avoids a double-borrow error below. Come back to /avoid if possible.
    let res_clone = residues.to_owned();

    // todo: Handle the N and C term A/R.

    // Increment H serial number, starting with the final atom present prior to adding H + 1)
    let mut highest_sn = 0;
    for atom in atoms.iter() {
        if atom.serial_number > highest_sn {
            highest_sn = atom.serial_number;
        }
    }
    let mut next_sn = highest_sn + 1;

    for (res_i, res) in residues.iter_mut().enumerate() {
        let mut n_next_pos = None;
        // todo: Messy DRY from the aa_data_from_coords fn.
        if res_i < res_len - 1 {
            let res_next = &res_clone[res_i + 1];

            let n_next = res_next.atom_sns.iter().find(|i| {
                if let Some(&idx) = index_map.get(*i) {
                    matches!(&atoms[idx].type_in_res, Some(tir) if *tir == AtomTypeInRes::N)
                } else {
                    false
                }
            });

            if let Some(n_next) = n_next {
                if let Some(&idx) = index_map.get(n_next) {
                    n_next_pos = Some(atoms[idx].posit);
                }
            }
        }

        // filter_map skips any serial numbers not yet in index_map (e.g. H atoms
        // pushed by a previous call that haven't been re-indexed, or atoms absent
        // from the atoms list due to alternate conformations / filtering upstream).
        let atoms_this_res: Vec<&_> = res
            .atom_sns
            .iter()
            .filter_map(|i| index_map.get(i).map(|&idx| &atoms[idx]))
            .collect();

        let custom_variant = custom_protonation.and_then(|map| map.get(&res_i).copied());
        let geometry_variant = if matches!(res.res_type, ResidueType::AminoAcid(AminoAcid::His)) {
            Some(resolve_his_tautomer_by_geometry(atoms, res))
        } else {
            None
        };
        let selected_variant = match res.res_type {
            // Gate matches `populate_peptide_ff_and_q`: terminal residues keep
            // standard side-chain protonation at any pH (their charge units have
            // no pH variants — LYN/CYM/ASH/GLH ship internal-only); custom
            // overrides and His tautomer geometry still apply everywhere.
            ResidueType::AminoAcid(aa) => {
                if custom_variant.is_none() && !matches!(res.end, ResidueEnd::Internal) {
                    geometry_variant
                } else {
                    resolve_variant(aa, ph, custom_variant, geometry_variant)
                        .map_err(|e| ParamError::new(&format!("Residue {res_i}: {e}")))?
                }
            }
            _ => None,
        };
        let this_digit_map = make_h_digit_map_custom(&ff_map.internal, ph, selected_variant);

        // todo: Handle the N term and C term cases; pass those params in.
        let (dihedral, mut h_added_this_res, this_cp_ca) = aa_data_from_coords(
            &atoms_this_res,
            &res_clone,
            &res.res_type,
            prev_cp_ca,
            n_next_pos,
            &this_digit_map,
            &disulfide_sg_sns,
        )?;

        // C-terminal carboxylate completion (糙1-residual (ii)): `aminoct12.lib`
        // types the C-terminus with O *and* OXT (each ≈ −0.82 → residue sum
        // exactly −1.000), but many X-ray CIFs deposit only one carboxyl O (or
        // altloc-dedup drops the second). Without the atom the OXT charge has
        // nowhere to live: the residue sums to ≈ −0.18 and the solute net
        // charge goes fractional (observed: 1XJ3 CTERM Leu@116). Synthesize the
        // missing OXT in the carboxyl sp2 plane from C's two existing
        // substituents (CA and O) at C–O carbonyl length; downstream bond
        // inference sees the 1.23 Å C–OXT and records the bond. N-terminus H1/
        // H2/H3 are already rebuilt inside aa_data_from_coords.
        if matches!(res.end, ResidueEnd::CTerminus) {
            let has_oxt = atoms_this_res
                .iter()
                .any(|a| matches!(a.type_in_res, Some(AtomTypeInRes::OXT)));
            if !has_oxt {
                let ca = atoms_this_res
                    .iter()
                    .find(|a| matches!(a.type_in_res, Some(AtomTypeInRes::CA)));
                let c = atoms_this_res
                    .iter()
                    .find(|a| matches!(a.type_in_res, Some(AtomTypeInRes::C)));
                let o = atoms_this_res
                    .iter()
                    .find(|a| matches!(a.type_in_res, Some(AtomTypeInRes::O)));
                if let (Some(ca), Some(c), Some(o)) = (ca, c, o) {
                    let posit =
                        planar_posit(c.posit, c.posit - ca.posit, o.posit - c.posit, LEN_CP_O);
                    h_added_this_res.push(AtomGeneric {
                        posit,
                        element: Element::Oxygen,
                        type_in_res: Some(AtomTypeInRes::OXT),
                        ..Default::default()
                    });
                }
            }
        }

        // Get the first atom's chain; probably OK for assigning a chain to H.
        let mut chain_i = 0;
        if !atoms_this_res.is_empty() {
            for (i, chain) in chains.iter().enumerate() {
                if chain.atom_sns.contains(&atoms_this_res[0].serial_number) {
                    chain_i = i;
                    break;
                }
            }
        }

        for mut h in h_added_this_res {
            h.serial_number = next_sn;
            atoms.push(h);
            index_map.insert(next_sn, atoms.len() - 1); // keep map in sync

            res.atom_sns.push(next_sn);
            chains[chain_i].atom_sns.push(next_sn);

            next_sn += 1;
        }

        prev_cp_ca = this_cp_ca;
        // res.dihedral = Some(dihedral);
        dihedrals.push(dihedral);
    }

    // Resolve clashes introduced by idealized H placement: an H can sit < 1.2 Å
    // from an atom of another residue in a folded protein and blow the system up
    // within a few MD steps. Clashing H's are first moved (or relaxed) on their
    // parent bond sphere; deletion is a flagged last resort (see resolve_h_clashes).
    resolve_h_clashes(atoms, residues, chains, highest_sn + 1);

    Ok(dihedrals)
}
