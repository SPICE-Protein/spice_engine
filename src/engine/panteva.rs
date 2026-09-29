//! Panteva–Giambasu–York (PGY) site corrections for divalent–nucleic-acid
//! 12-6-4 interactions, expressed through the engine's pair-C4 rule.
//!
//! Provenance (verified 2026-09-18):
//! - Panteva, Giambasu & York, *J. Phys. Chem. B* 2015, 119, 15460
//!   (doi:10.1021/acs.jpcb.5b10423) — "Force Field for Mg²⁺, Mn²⁺, Zn²⁺ and
//!   Cd²⁺ Ions That Have Balanced Interactions with Nucleic Acids". Their
//!   m12-6-4 re-optimizes the ion–site C4 coefficients for three representative
//!   sites (phosphate non-bridging O, adenine N7, guanine N7) and defines every
//!   other site's pair coefficient proportional to its polarizability:
//!   `C4(ion–site) = C4(ion–water) · α(site) / α(water)`.
//! - The effective site polarizabilities are exactly what Amber's `parmed
//!   add12_6_4` consumes (`lj_1264_pol.dat`, distributed with the York-group
//!   tutorial for this paper): OW 1.444 (Eisenberg & Kauzmann), base N 1.090 /
//!   carbonyl O 0.569 / hydroxyl O 0.637 / sp2 C 1.352 (Miller et al., *JACS*
//!   112, 8533, 1990), and the Mg-specific fitted overrides 1.910 (A N7),
//!   1.925 (G N7), 0.170 (phosphate OP).
//!
//! How it rides on this engine: the std–std pair rule in
//! `non_bonded::combine_lj_params` combines two atoms that BOTH carry `lj_c4`
//! by geometric mean (Amber's own 12-6-4 combining convention). PGY
//! coefficients are PAIR constants, so a site that should reach `pair`
//! against an ion of water-pair constant `c4_ion` stores
//! `slot = pair² / c4_ion` — [`atom_slot`] does that bookkeeping;
//! [`pair_c4`] answers the direct question.
//!
//! Activation caveat (this paragraph is the record): the engine loads the
//! OL24/OL3 nucleic-acid maps but has no NA build path yet, so nothing stores
//! a site slot today. When `populate_nucleic_ff_and_q` lands, tag the OP/N7
//! atom types via [`atom_slot`] and this module becomes live. Also note:
//! Panteva's fitted overrides were published for the TIP4P-Ew-based ion set;
//! applying their published proportionality formula against OUR OPC ion–water
//! C4 (Mg 127) is the formula's documented use, but the constants below are
//! therefore an OPC extrapolation, not a re-fit. A mixed-ion box also cannot
//! share one slot value (slot is ion-specific) — per-pair tables are the
//! designed future fix, stated here so nobody re-derives the limitation.

/// Water polarizability (Å³) used by PGY's scaling denominator.
pub const ALPHA_WATER: f32 = 1.444;

/// Effective site polarizabilities (Å³) as shipped in `lj_1264_pol.dat`.
pub mod alpha {
    /// Adenine N7 × Mg²⁺ (Panteva 2015 fit).
    pub const A_N7_MG: f32 = 1.910;
    /// Guanine N7 × Mg²⁺ (Panteva 2015 fit).
    pub const G_N7_MG: f32 = 1.925;
    /// Phosphate non-bridging O × Mg²⁺ (Panteva 2015 fit — the WEAKENING
    /// correction: Mg–phosphate was over-stabilized by plain 12-6-4).
    pub const OP_MG: f32 = 0.170;
    /// Base nitrogen (generic, Miller 1990).
    pub const BASE_N: f32 = 1.090;
    /// Carbonyl-type oxygen (Miller 1990).
    pub const CARBONYL_O: f32 = 0.569;
    /// Hydroxyl/ether oxygen (Miller 1990).
    pub const HYDROXYL_O: f32 = 0.637;
    /// Aromatic/sp2 carbon (Miller 1990).
    pub const SP2_C: f32 = 1.352;
}

/// Mg²⁺ Li–Merz **OPC** ion–water pair constant (must equal `ION_MG.c4`;
/// locked by test).
pub const MG_WATER_C4: f32 = 127.0;

/// PGY's published prescription: the ion–site pair C4.
pub fn pair_c4(ion_water_c4: f32, alpha_site: f32) -> f32 {
    ion_water_c4 * alpha_site / ALPHA_WATER
}

/// Value to store in a site atom's `lj_c4` so the engine's geometric-mean
/// pair rule reproduces [`pair_c4`] against an ion of water-pair constant
/// `ion_water_c4`. Guards: a non-positive ion constant or site polarizability
/// yields 0 (no correction channel).
pub fn atom_slot(ion_water_c4: f32, alpha_site: f32) -> f32 {
    if ion_water_c4 <= 0.0 || alpha_site <= 0.0 {
        return 0.0;
    }
    let pair = pair_c4(ion_water_c4, alpha_site);
    pair * pair / ion_water_c4
}

/// Mg²⁺ (OPC) against the three fitted sites — const-folded PGY pairs.
pub mod mg_opc {
    use super::alpha;
    pub const OP: f32 = super::MG_WATER_C4 * alpha::OP_MG / super::ALPHA_WATER;
    pub const A_N7: f32 = super::MG_WATER_C4 * alpha::A_N7_MG / super::ALPHA_WATER;
    pub const G_N7: f32 = super::MG_WATER_C4 * alpha::G_N7_MG / super::ALPHA_WATER;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::md_core::ION_MG;
    use crate::engine::md_core::non_bonded::combine_lj_params;
    use crate::engine::md_core::solvent::O_SIGMA;

    #[test]
    fn ion_c4_literal_matches_the_shipped_magnesium() {
        assert_eq!(ION_MG.c4, MG_WATER_C4);
        // Water oxygen is stored as true σ; the R*/2 it came from:
        let _ = O_SIGMA;
    }

    #[test]
    fn derived_mg_opc_pair_constants_match_hand_arithmetic() {
        // 127 × 0.170 / 1.444 = 14.9515…; × 1.910/1.444 = 167.985…;
        // × 1.925/1.444 = 169.302…
        assert!((mg_opc::OP - 14.9515).abs() < 1e-3, "{}", mg_opc::OP);
        assert!((mg_opc::A_N7 - 167.985).abs() < 1e-2, "{}", mg_opc::A_N7);
        assert!((mg_opc::G_N7 - 169.302).abs() < 1e-2, "{}", mg_opc::G_N7);
    }

    #[test]
    fn atom_slot_round_trips_through_the_engine_pair_rule() {
        let mut ion = crate::engine::md_core::AtomDynamics::default();
        ion.lj_c4 = MG_WATER_C4;
        for a in [alpha::OP_MG, alpha::A_N7_MG, alpha::G_N7_MG, alpha::BASE_N] {
            let mut site = crate::engine::md_core::AtomDynamics::default();
            site.lj_c4 = atom_slot(MG_WATER_C4, a);
            let (_, _, c4_pair) = combine_lj_params(&ion, &site);
            assert!(
                (c4_pair - pair_c4(MG_WATER_C4, a)).abs() < 1e-2 * (1.0 + a),
                "sqrt(ion·slot) must reproduce the PGY pair constant for α={a}"
            );
        }
    }

    #[test]
    fn single_sided_and_guard_cases_combine_to_zero() {
        let mut ion = crate::engine::md_core::AtomDynamics::default();
        ion.lj_c4 = MG_WATER_C4;
        let plain = crate::engine::md_core::AtomDynamics::default();
        let (_, _, c4_pair) = combine_lj_params(&ion, &plain);
        assert_eq!(c4_pair, 0.0); // protein stays pure 12-6 against Mg
        assert_eq!(atom_slot(0.0, 1.0), 0.0);
        assert_eq!(atom_slot(127.0, 0.0), 0.0);
    }
}
