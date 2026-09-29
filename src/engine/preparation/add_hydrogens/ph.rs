//! Logic for pH-related adjustment to the DigitMap.
//!
//! How this affects protenation based on pH:
//! At low pH:
//! - HIP provides both HD* and HE* hydrogens on His.
//! - ASH, GLH hydrogens become valid; standard ASP/GLU are excluded.
//! - CYS stays protonated; CYM excluded.
//! - LYS stays protonated; LYN excluded.
//!
//! At neutral pH (~7):
//! - His becomes neutral (we pick HIE by default here). Only HE* hydrogens will be “valid”; HD* are not,
//!   unless we later swap to HID by an environment rule.
//! - ASP/GLU are deprotonated; ASH/GLH excluded.
//! - CYS is protonated; CYM excluded.
//! - LYS remains protonated; LYN excluded.
//!
//! At high pH:
//! - LYN appears (no LYS hydrogens).
//! - CYM may appear if pH is sufficiently high.
//! - His neutral (HIE) unless you push very low pH again.

// todo: pick HID vs HIE by local geometry:
// Compute ND1/NE2 distances to nearby acceptors (O, carboxylate O, backbone carbonyl O, etc.) and
// choose the proton on the ring nitrogen farther from a strong acceptor (so the closer one can
// accept H-bond). If it picks ND1 → use HID; if NE2 → HIE. Then pass that choice into
// make_h_digit_map_custom (e.g. extra arg) or cache a per-run his_selected before you call it.
// That’s still a localized change.

use bio_files::{AtomGeneric, ResidueGeneric};
use na_seq::{AminoAcid, AminoAcidProtenationVariant, AtomTypeInRes, Element};

// Intrinsic pKas (typical, solvent-exposed). These are deliberately simple.
// todo: Make them better?
// todo: Add Arg, and Termini on AminoAcidGeneric? Are they in the Amber data?
pub(crate) const PKA_ASP: f32 = 3.9;
pub(crate) const PKA_GLU: f32 = 4.2;
const PKA_HIS: f32 = 6.0;
const PKA_CYS: f32 = 8.3;
const PKA_LYS: f32 = 10.5;
pub(crate) const PKA_TYR: f32 = 10.5;

pub(crate) fn resolve_his_tautomer_by_geometry(
    atoms: &[AtomGeneric],
    his_res: &ResidueGeneric,
) -> AminoAcidProtenationVariant {
    // 1. Locate ND1 and NE2 atoms of the His residue
    let mut nd1_pos = None;
    let mut ne2_pos = None;
    for &sn in &his_res.atom_sns {
        if let Some(atom) = atoms.iter().find(|a| a.serial_number == sn) {
            if let Some(tir) = &atom.type_in_res {
                if *tir == AtomTypeInRes::ND1 {
                    nd1_pos = Some(atom.posit);
                } else if *tir == AtomTypeInRes::NE2 {
                    ne2_pos = Some(atom.posit);
                }
            }
        }
    }

    let (Some(p_nd1), Some(p_ne2)) = (nd1_pos, ne2_pos) else {
        return AminoAcidProtenationVariant::Hie;
    };

    // 2. Scan for hydrogen bond acceptors in all other residues
    let mut min_d_nd1 = 999.0f64;
    let mut min_d_ne2 = 999.0f64;

    for atom in atoms {
        if his_res.atom_sns.contains(&atom.serial_number) {
            continue;
        }

        // Be conservative for untyped atoms: an unknown nitrogen may be an
        // amide or protonated center and must not decide the His tautomer.
        let is_acceptor = if let Some(tir) = &atom.type_in_res {
            matches!(
                tir,
                AtomTypeInRes::O
                    | AtomTypeInRes::OD1
                    | AtomTypeInRes::OD2
                    | AtomTypeInRes::OE1
                    | AtomTypeInRes::OE2
                    | AtomTypeInRes::OH
            )
        } else {
            atom.element == Element::Oxygen
        };

        if is_acceptor {
            let d_nd1 = (atom.posit - p_nd1).magnitude() as f64;
            let d_ne2 = (atom.posit - p_ne2).magnitude() as f64;
            if d_nd1 < min_d_nd1 {
                min_d_nd1 = d_nd1;
            }
            if d_ne2 < min_d_ne2 {
                min_d_ne2 = d_ne2;
            }
        }
    }

    const ACCEPTOR_CUTOFF: f64 = 4.0;
    if min_d_nd1 > ACCEPTOR_CUTOFF && min_d_ne2 > ACCEPTOR_CUTOFF {
        return AminoAcidProtenationVariant::Hie;
    }
    // The closer ring nitrogen remains unprotonated so it can accept a H bond.
    // Prefer HIE on ties for a deterministic, conservative default.
    if min_d_nd1 + 0.1 < min_d_ne2 {
        AminoAcidProtenationVariant::Hie
    } else if min_d_ne2 + 0.1 < min_d_nd1 {
        AminoAcidProtenationVariant::Hid
    } else {
        AminoAcidProtenationVariant::Hie
    }
}

pub(crate) fn his_choice(ph: f32) -> Option<AminoAcidProtenationVariant> {
    if ph < PKA_HIS {
        Some(AminoAcidProtenationVariant::Hip)
    } else {
        Some(AminoAcidProtenationVariant::Hie)
    }
}

/// Resolve the pH-dependent side-chain state. Explicit overrides are validated
/// against the residue before they are returned; neutral His is the only state
/// that uses local geometry to choose HID versus HIE.
pub(crate) fn resolve_variant(
    aa: AminoAcid,
    ph: f32,
    override_variant: Option<AminoAcidProtenationVariant>,
    his_geometry: Option<AminoAcidProtenationVariant>,
) -> Result<Option<AminoAcidProtenationVariant>, String> {
    if let Some(v) = override_variant {
        if v.get_standard() != Some(aa) {
            return Err(format!(
                "protonation variant {v} is incompatible with residue {aa}"
            ));
        }
        return Ok(Some(v));
    }

    let selected = match aa {
        AminoAcid::Asp if ph < PKA_ASP => Some(AminoAcidProtenationVariant::Ash),
        AminoAcid::Glu if ph < PKA_GLU => Some(AminoAcidProtenationVariant::Glh),
        AminoAcid::Cys if ph >= PKA_CYS => Some(AminoAcidProtenationVariant::Cym),
        AminoAcid::Lys if ph > PKA_LYS => Some(AminoAcidProtenationVariant::Lyn),
        AminoAcid::His if ph < PKA_HIS => Some(AminoAcidProtenationVariant::Hip),
        AminoAcid::His => Some(match his_geometry {
            Some(AminoAcidProtenationVariant::Hid) => AminoAcidProtenationVariant::Hid,
            _ => AminoAcidProtenationVariant::Hie,
        }),
        _ => None,
    };
    Ok(selected)
}

pub(crate) fn variant_allowed_at_ph(aa_var: AminoAcidProtenationVariant, ph: f32) -> bool {
    match aa_var {
        // Histidine variants are selected by his_choice(); these entries keep
        // variant_allowed_at_ph consistent but are not used inside make_h_digit_map_custom.
        AminoAcidProtenationVariant::Hip => ph < PKA_HIS,
        AminoAcidProtenationVariant::Hid | AminoAcidProtenationVariant::Hie => ph >= PKA_HIS,

        // Acids: protonated variants (ASH/GLH) when pH is below pKa.
        // Together with standard_allowed_at_ph these exactly partition all pH values.
        AminoAcidProtenationVariant::Ash => ph < PKA_ASP,
        AminoAcidProtenationVariant::Glh => ph < PKA_GLU,

        // Cys: thiolate (CYM) only above pKa
        AminoAcidProtenationVariant::Cym => ph >= PKA_CYS,

        // Disulfide cystine (CYX) is not pH-driven; don’t include here via pH.
        AminoAcidProtenationVariant::Cyx => false,

        // Lys: neutral LYN only above pKa
        AminoAcidProtenationVariant::Lyn => ph > PKA_LYS,

        // Capping or special forms (not pH-driven)
        AminoAcidProtenationVariant::Ace
        | AminoAcidProtenationVariant::Nhe
        | AminoAcidProtenationVariant::Nme
        | AminoAcidProtenationVariant::Hyp => false,
    }
}

/// Only restrict standards when Amber has an alternate protonation variant.
/// Thresholds use exact pKa so that standard + variant together cover all pH
/// values with no dead zone (the previous ±WINDOW offset created gaps where
/// neither form was selected, causing missing digit_map entries and errors).
pub(crate) fn standard_allowed_at_ph(aa: na_seq::AminoAcid, ph: f32) -> bool {
    match aa {
        // Amber "ASP"/"GLU" are deprotonated; allow at or above pKa.
        na_seq::AminoAcid::Asp => ph >= PKA_ASP,
        na_seq::AminoAcid::Glu => ph >= PKA_GLU,

        // Amber "LYS" is protonated; allow at or below pKa.
        na_seq::AminoAcid::Lys => ph <= PKA_LYS,

        // Amber "CYS" is protonated; allow below pKa (CYM covers above).
        na_seq::AminoAcid::Cys => ph < PKA_CYS,

        // Histidine: Amber uses variants (HID/HIE/HIP) not the bare "HIS" residue.
        na_seq::AminoAcid::His => false,

        // Others unaffected
        _ => true,
    }
}
