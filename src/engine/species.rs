//! Build-time "insertable species" abstraction (v1.3.2).
//!
//! Everything the engine can drop into a displaced water slot — a bare
//! monatomic ion, a MgCl₂-style formula unit, a multi-site cosolute — is
//! described here as ONE data shape: [`SpeciesSpec`] = 1..n [`SiteSpec`]
//! sites hung off the slot's water oxygen, plus optional internal bonds
//! `(i, j, k, r0)` (force constant BEFORE equilibrium distance — the order
//! [`CosolventSpec::bonds`] documents and the probe tests lock).
//!
//! Design stance: *behavior is not polymorphic; data is normalized.* The
//! trait [`Insertable`] only converts inputs into `SpeciesSpec`; the single
//! insertion algorithm lives in `md_core` (`insert_instance`) and never
//! sees a `dyn`. That keeps the MD hot path untouched (consumers already
//! treat ions/cosolutes as plain `AtomDynamics`) and keeps every insertion
//! decision deterministic under the seeded water-slot shuffle.
//!
//! Single-atom ions are the degenerate species: one site, zero offset, no
//! bonds — `IonParams::to_species()` is the only bridge, and the provenance
//! locks on `ION_*` keep working through it. Multi-site species now carry
//! `c4` per site (the v1.3.1 gap where `add_cosolvent` dropped the 12-6-4
//! induction coefficient is fixed by threading `SiteSpec::c4` into
//! `AtomDynamics::lj_c4`).
//!
//! [`SaltSpec`] pairs any cation species with any anion species and derives
//! the electroneutral formula-unit stoichiometry from the integer charges
//! via [`balance_stoichiometry`], so KCl (1:1), MgCl₂ (1:2) and future
//! Na₂SO₄ (2:1) all route through the same code path.

use super::{CHARGE_UNIT_SCALER, CosolventSpec, IonParams};
use lin_alg::f32::Vec3;
use na_seq::Element;

/// One site of an insertable species: LJ in the engine's true-σ convention
/// (see `R_STAR_TO_SIGMA`), charge in scaled engine units, `offset` relative
/// to the displaced water's oxygen. A monatomic ion is a single site with a
/// zero offset.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteSpec {
    pub ff_type: String,
    pub element: Element,
    pub mass: f32,
    pub charge_scaled: f32,
    pub sigma: f32,
    pub eps: f32,
    /// 12-6-4 induction coefficient (ion–water channel and, against another
    /// c4-bearing site, the geometric-mean std–std pair). 0 = plain 12-6.
    /// Negative values have no consumer today (`combine_lj_params` only
    /// pairs positive–positive), matching the anion-side caveat on
    /// `IonParams`.
    pub c4: f32,
    pub offset: Vec3,
}

/// A complete insertable template. Contract: one instance occupies exactly
/// one water slot; its sites hang off that slot's O by `offset`.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeciesSpec {
    pub name: String,
    pub sites: Vec<SiteSpec>,
    /// Internal connectivity as `(site_i, site_j, k, r0)` — force constant
    /// first. Each bond becomes one harmonic distance restraint
    /// (E = ½·k·Δr²) and one 1-2 nonbonded exclusion; sites in no bond
    /// float free within the slot.
    pub bonds: Vec<(usize, usize, f32, f32)>,
}

impl SpeciesSpec {
    /// Degenerate single-site species with no internal bonds and a zero
    /// offset — the shape `insert_instance` must place bit-identically to
    /// the historical `insert_ion` (ion slot position taken verbatim, no
    /// offset addition at all).
    pub fn is_atomic(&self) -> bool {
        self.sites.len() == 1
            && self.bonds.is_empty()
            && self.sites[0].offset.x == 0.0
            && self.sites[0].offset.y == 0.0
            && self.sites[0].offset.z == 0.0
    }

    /// Net charge of one instance in units of *e*.
    pub fn instance_charge_e(&self) -> f32 {
        self.sites
            .iter()
            .map(|s| f64::from(s.charge_scaled))
            .sum::<f64>() as f32
            / CHARGE_UNIT_SCALER
    }

    /// View an `IonParams` constant as a species (single site, zero offset,
    /// full LJ + C4 carry-over).
    pub fn from_ion(ion: &IonParams) -> Self {
        SpeciesSpec {
            name: ion.ff_type.to_string(),
            sites: vec![SiteSpec {
                ff_type: ion.ff_type.to_string(),
                element: ion.element,
                mass: ion.mass,
                charge_scaled: ion.charge_scaled,
                sigma: ion.sigma,
                eps: ion.eps,
                c4: ion.c4,
                offset: Vec3::new(0.0, 0.0, 0.0),
            }],
            bonds: Vec::new(),
        }
    }
}

/// Normalization contract for anything the insertion layer accepts.
/// Implement this to plug a new *parameter carrier* (a future polyatomic
/// ion table, a molecule-editor product, …) into `add_salt` /
/// `insert_formula_unit_salts` without touching the insertion algorithm.
pub trait Insertable {
    fn species_name(&self) -> &str;
    fn to_species(&self) -> SpeciesSpec;
    fn instance_charge_e(&self) -> f32 {
        self.to_species().instance_charge_e()
    }
}

impl Insertable for IonParams {
    fn species_name(&self) -> &str {
        self.ff_type
    }
    fn to_species(&self) -> SpeciesSpec {
        SpeciesSpec::from_ion(self)
    }
}

impl Insertable for SpeciesSpec {
    fn species_name(&self) -> &str {
        &self.name
    }
    fn to_species(&self) -> SpeciesSpec {
        self.clone()
    }
}

impl Insertable for CosolventSpec {
    fn species_name(&self) -> &str {
        &self.name
    }
    fn to_species(&self) -> SpeciesSpec {
        SpeciesSpec {
            name: self.name.clone(),
            sites: self.sites.clone(),
            bonds: self.bonds.clone(),
        }
    }
}

/// Why a (cation, anion) pair cannot form an electroneutral formula unit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SaltBalanceError {
    /// Either charge is 0 (or rounds to 0): no ion, nothing to balance.
    ZeroCharge,
    /// Both charges have the same sign — cation/anion were swapped or both
    /// are of one class. Never silently exchanged: a swapped pair is a
    /// configuration bug the caller must see.
    SameSign,
    /// |q − round(q)| > 1e-3 — formal ion charges are integers in units of
    /// *e*; anything else means scaled units leaked in (wiring bug).
    NonIntegerCharge { q_e: f32 },
    /// |q| > 4 — no shipped species is beyond tetravalent; guards against
    /// feeding e-scaled floats here by mistake.
    OutOfRange { q_e: f32 },
}

impl std::fmt::Display for SaltBalanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SaltBalanceError::ZeroCharge => write!(f, "salt balance: zero charge"),
            SaltBalanceError::SameSign => write!(f, "salt balance: cation and anion same sign"),
            SaltBalanceError::NonIntegerCharge { q_e } => {
                write!(f, "salt balance: non-integer charge {q_e} e")
            }
            SaltBalanceError::OutOfRange { q_e } => {
                write!(f, "salt balance: charge {q_e} e out of |q|<=4 range")
            }
        }
    }
}

/// Smallest positive integer (n_cation, n_anion) with
/// `n_cation * q_cation + n_anion * q_anion = 0`, charges in units of *e*
/// (signed). (+1,−1)→(1,1); (+2,−1)→(1,2); (+1,−2)→(2,1); (+2,−3)→(3,2)
/// (Ca₃(PO₄)₂ order). See [`SaltBalanceError`] for the defense rules.
pub fn balance_stoichiometry(
    q_cation_e: f32,
    q_anion_e: f32,
) -> Result<(u32, u32), SaltBalanceError> {
    let integerize = |q_e: f32| -> Result<i64, SaltBalanceError> {
        let r = q_e.round();
        if !q_e.is_finite() || (q_e - r).abs() > 1e-3 {
            return Err(SaltBalanceError::NonIntegerCharge { q_e });
        }
        let ri = r as i64;
        if ri == 0 {
            return Err(SaltBalanceError::ZeroCharge);
        }
        if ri.abs() > 4 {
            return Err(SaltBalanceError::OutOfRange { q_e });
        }
        Ok(ri)
    };
    let qc = integerize(q_cation_e)?;
    let qa = integerize(q_anion_e)?;
    if qc <= 0 || qa >= 0 {
        return Err(SaltBalanceError::SameSign);
    }
    let ac = qc.unsigned_abs() as u32;
    let aa = qa.unsigned_abs() as u32;
    let g = gcd(ac, aa);
    Ok((aa / g, ac / g))
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Closed registry of shipped species by force-field name (case-insensitive;
/// both the charge-suffixed ff_type and the bare element symbol resolve).
/// Every entry is a provenance-locked `ION_*` constant — the registry is a
/// name map, never a place to type new numbers. Future polyatomic ions join
/// this table when their vetted `SpeciesSpec` presets land — a parameter
/// provenance project (blocked on published OPC-compatible ion params and a
/// fit), never a parser feature: JSON carries names, numbers enter only
/// through provenance-locked constants guarded by `ion_provenance_tests`.
pub fn ion_by_name(name: &str) -> Option<IonParams> {
    use super::{ION_BA, ION_CA, ION_CL, ION_K, ION_MG, ION_NA, ION_SR};
    // Interior spaces are stripped too: " k + " is the same request as "K+".
    let key: String = name
        .chars()
        .filter(|c| !c.is_ascii_whitespace())
        .collect::<String>()
        .to_ascii_uppercase();
    Some(match key.as_str() {
        "NA+" | "NA" => ION_NA,
        "K+" | "K" => ION_K,
        "CL-" | "CL" => ION_CL,
        "MG2+" | "MG2" | "MG" => ION_MG,
        "CA2+" | "CA2" | "CA" => ION_CA,
        "SR2+" | "SR2" | "SR" => ION_SR,
        "BA2+" | "BA2" | "BA" => ION_BA,
        _ => return None,
    })
}

/// Names accepted by [`ion_by_name`], for error messages and docs.
pub const SALT_ION_NAMES: [&str; 7] = ["Na+", "K+", "Cl-", "Mg2+", "Ca2+", "Sr2+", "Ba2+"];

#[derive(serde::Deserialize)]
struct JsSalt {
    cation: String,
    anion: String,
    #[serde(default)]
    molarity: f32,
}

/// Parse the Python/FFI `salts_json` payload (v1.3.2): a JSON array of
/// `{"cation": "K+", "anion": "Cl-", "molarity": 0.15}`. Species resolve
/// through the closed [`ion_by_name`] registry; stoichiometry is derived
/// from their charges (so `"cation":"Na+","anion":"CL-"`-style typos in
/// the *numbers* are impossible — names only). Empty string → empty list.
pub fn parse_salts_json(s: &str) -> Result<Vec<SaltSpec>, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Vec::new());
    }
    let items: Vec<JsSalt> = serde_json::from_str(s).map_err(|e| format!("salts_json: {e}"))?;
    items.iter().try_fold(Vec::new(), |mut acc, item| {
        let cation = ion_by_name(&item.cation).ok_or_else(|| {
            format!(
                "salts_json: cation '{}' not in the shipped registry {SALT_ION_NAMES:?}",
                item.cation
            )
        })?;
        let anion = ion_by_name(&item.anion).ok_or_else(|| {
            format!(
                "salts_json: anion '{}' not in the shipped registry {SALT_ION_NAMES:?}",
                item.anion
            )
        })?;
        if !item.molarity.is_finite() || item.molarity < 0.0 {
            return Err(format!(
                "salts_json: {}-{} molarity must be >= 0",
                item.cation, item.anion
            ));
        }
        let salt = SaltSpec::new(cation, anion, item.molarity);
        // Fail fast on pairings that cannot form a formula unit — better a
        // build-time Err than a silent skip at insertion.
        salt.stoichiometry()
            .map_err(|e| format!("salts_json: {}-{}: {e}", item.cation, item.anion))?;
        acc.push(salt);
        Ok::<_, String>(acc)
    })
}

/// An electroneutral electrolyte: cation species × n⁺ + anion species × n⁻
/// per formula unit, with the stoichiometry derived from the charges.
/// `molarity` counts FORMULA UNITS per litre (like the divalent MgCl₂
/// knobs), so c·V·N_A units × (n⁺+n⁻) water slots get displaced.
#[derive(Clone, Debug, PartialEq)]
pub struct SaltSpec {
    pub cation: SpeciesSpec,
    pub anion: SpeciesSpec,
    pub molarity: f32,
}

impl SaltSpec {
    pub fn new(cation: impl Insertable, anion: impl Insertable, molarity: f32) -> Self {
        SaltSpec {
            cation: cation.to_species(),
            anion: anion.to_species(),
            molarity,
        }
    }

    /// (n_cation, n_anion) of one formula unit; errors are surfaced (not
    /// silently fixed) — see [`SaltBalanceError`].
    pub fn stoichiometry(&self) -> Result<(u32, u32), SaltBalanceError> {
        balance_stoichiometry(
            self.cation.instance_charge_e(),
            self.anion.instance_charge_e(),
        )
    }

    /// The insertion sequence of one formula unit: `n_cation` copies of the
    /// cation species followed by `n_anion` copies of the anion species.
    /// Each instance occupies its own water slot, exactly like the
    /// historical MgCl₂ cycle [cation, anion, anion] per unit.
    pub fn formula_unit_parts(&self) -> Result<Vec<&SpeciesSpec>, SaltBalanceError> {
        let (nc, na) = self.stoichiometry()?;
        let mut v = Vec::with_capacity((nc + na) as usize);
        v.extend(std::iter::repeat_n(&self.cation, nc as usize));
        v.extend(std::iter::repeat_n(&self.anion, na as usize));
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::md_core::{ION_CL, ION_MG, ION_NA};

    #[test]
    fn balances_the_shipped_families() {
        assert_eq!(balance_stoichiometry(1.0, -1.0), Ok((1, 1))); // NaCl, KCl
        assert_eq!(balance_stoichiometry(2.0, -1.0), Ok((1, 2))); // MgCl2
        assert_eq!(balance_stoichiometry(1.0, -2.0), Ok((2, 1))); // Na2SO4 (future)
        assert_eq!(balance_stoichiometry(2.0, -3.0), Ok((3, 2))); // Ca3(PO4)2
        assert_eq!(
            balance_stoichiometry(-1.0, 1.0).unwrap_err(),
            SaltBalanceError::SameSign
        );
    }

    #[test]
    fn rejects_bad_wiring_instead_of_guessing() {
        assert_eq!(
            balance_stoichiometry(0.0, -1.0),
            Err(SaltBalanceError::ZeroCharge)
        );
        assert_eq!(
            balance_stoichiometry(1.0, 0.0),
            Err(SaltBalanceError::ZeroCharge)
        );
        assert_eq!(
            balance_stoichiometry(1.0, 1.0),
            Err(SaltBalanceError::SameSign)
        );
        // 18.2223 ≈ one elementary charge *scaled* — the classic unit leak;
        // it trips the integerity guard before the range guard.
        assert!(matches!(
            balance_stoichiometry(18.2223, -1.0),
            Err(SaltBalanceError::NonIntegerCharge { .. })
        ));
        assert!(matches!(
            balance_stoichiometry(5.0, -1.0),
            Err(SaltBalanceError::OutOfRange { .. })
        ));
        assert!(matches!(
            balance_stoichiometry(1.5, -1.0),
            Err(SaltBalanceError::NonIntegerCharge { .. })
        ));
        // Tiny float slivers from charge_scaled/CHARGE_UNIT_SCALER round-trips
        // must NOT trip the guard.
        assert_eq!(balance_stoichiometry(1.0 + 1e-6, -1.0), Ok((1, 1)));
    }

    #[test]
    fn ions_view_as_atomic_species_losslessly() {
        for ion in [ION_NA, ION_CL, ION_MG] {
            let sp = ion.to_species();
            assert!(sp.is_atomic(), "{:?}", ion.ff_type);
            assert_eq!(sp.sites[0].sigma.to_bits(), ion.sigma.to_bits());
            assert_eq!(sp.sites[0].eps.to_bits(), ion.eps.to_bits());
            assert_eq!(sp.sites[0].c4.to_bits(), ion.c4.to_bits());
            assert_eq!(sp.sites[0].mass, ion.mass);
            assert_eq!(sp.sites[0].ff_type, ion.ff_type);
        }
        assert_eq!(ION_NA.instance_charge_e(), 1.0);
        assert_eq!(ION_CL.instance_charge_e(), -1.0);
        assert_eq!(ION_MG.instance_charge_e(), 2.0);
    }

    #[test]
    fn cosolvent_specs_keep_their_shape() {
        use crate::engine::md_core::cosolvent_presets::{tmao, urea};
        let u = urea().to_species();
        assert_eq!(u.sites.len(), 8);
        assert_eq!(u.bonds.len(), 7);
        assert!(!u.is_atomic());
        // CGenFF-derived presets are 12-6 only: every c4 must be 0.
        assert!(u.sites.iter().all(|s| s.c4 == 0.0));
        assert!(tmao().to_species().instance_charge_e().abs() < 1e-4);
    }

    #[test]
    fn registry_resolves_every_shipped_name_and_rejects_rest() {
        for want in ["Na+", "K+", "Cl-", "Mg2+", "Ca2+", "Sr2+", "Ba2+"] {
            let got = ion_by_name(want).unwrap_or_else(|| panic!("name {want} must resolve"));
            assert_eq!(got.ff_type, want);
            // Bare element forms and case/mess resolve identically.
            assert_eq!(
                got.ff_type,
                ion_by_name(&want.to_lowercase()).unwrap().ff_type
            );
        }
        assert!(
            ion_by_name("SO4--").is_none(),
            "polyatomic channels not shipped yet"
        );
        assert!(ion_by_name("water").is_none());
        assert_eq!(ion_by_name(" k + ").unwrap().ff_type, "K+");
    }

    #[test]
    fn salts_json_round_trips_and_rejects() {
        let salts = parse_salts_json(r#"[{"cation":"K+","anion":"Cl-","molarity":0.15}]"#).unwrap();
        assert_eq!(salts.len(), 1);
        assert_eq!(salts[0].stoichiometry(), Ok((1, 1)));
        assert_eq!(salts[0].cation.name, "K+");
        // Missing molarity defaults to 0 (parsed, inserted as nothing).
        let z = parse_salts_json(r#"[{"cation":"MG2","anion":"cl-"}]"#).unwrap();
        assert_eq!(z[0].stoichiometry(), Ok((1, 2)));
        assert_eq!(z[0].molarity, 0.0);
        assert_eq!(parse_salts_json("").unwrap().len(), 0);
        // Errors: unknown name, negative molarity, unbalanceable pair.
        assert!(parse_salts_json(r#"[{"cation":"Xx","anion":"Cl-"}]"#).is_err());
        assert!(parse_salts_json(r#"[{"cation":"K+","anion":"Cl-","molarity":-1}]"#).is_err());
        assert!(parse_salts_json(r#"[{"cation":"Na+","anion":"K+"}]"#).is_err());
        assert!(parse_salts_json("not json").is_err());
    }

    #[test]
    fn kplus_carries_the_published_opc_row() {
        // Cross-check against the same paper constants ion_provenance_tests
        // locks — kept here so a registry edit can't dodge the provenance
        // table (Sengupta 2021 Table 2 OPC: 1.702 / 0.13953816).
        let k = ion_by_name("K+").unwrap();
        assert_eq!(k.sigma, 1.702 * super::super::R_STAR_TO_SIGMA);
        assert_eq!(k.eps, 0.139_538_16);
        assert_eq!(k.c4, 0.0);
        assert_eq!(k.mass, 39.0983);
    }

    #[test]
    fn salt_spec_expands_formula_units_in_order() {
        let mgcl2 = SaltSpec::new(ION_MG, ION_CL, 0.025);
        assert_eq!(mgcl2.stoichiometry(), Ok((1, 2)));
        let parts = mgcl2.formula_unit_parts().unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].name, "Mg2+");
        assert_eq!(parts[1].name, "Cl-");
        assert_eq!(parts[2].name, "Cl-");

        // Synthetic 2:1 electrolyte (a stand-in for a divalent anion — no
        // real polyatomic parameters are invented here): cation Na+ ×2 +
        // fake A2- ×1.
        let mut fake2 = ION_CL.to_species();
        fake2.name = "SYNTH-A2".to_string();
        fake2.sites[0].charge_scaled = -2.0 * CHARGE_UNIT_SCALER;
        let na2a = SaltSpec::new(ION_NA, fake2, 0.1);
        assert_eq!(na2a.stoichiometry(), Ok((2, 1)));
        let parts = na2a.formula_unit_parts().unwrap();
        assert_eq!(
            parts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
            vec!["Na+", "Na+", "SYNTH-A2"]
        );
    }
}
