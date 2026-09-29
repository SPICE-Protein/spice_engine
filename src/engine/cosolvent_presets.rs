//! Vendored cosolvent parameter sets + JSON ingestion for the FFI boundary.
//!
//! Urea, TMAO and guanidinium chloride are the three denaturant/stabilizer
//! cosolvents the SPICE env vector scans. v1.3 shipped `CosolventSpec` as a
//! mechanism with NO built-in chemistry (fabricating parameters was banned);
//! v1.3.1 fills that gap from provenance that is already in this repo: the
//! CHARMM **CGenFF v4.6** distributions bundled under
//! `src/forcefield/parameters/data/charmm/`:
//! - `top_all36_cgenff_v4.6.rtf` — `RESI UREA` (8 atoms, net 0),
//!   `RESI TMAO` (14 atoms, net 0), `RESI GUAN` (10 atoms, net +1);
//!   per-atom types + partial charges (elementary charge).
//! - `par_all36_cgenff_v4.6.prm` — `NONBONDED` (ε, R_min/2 per type) and
//!   `BONDS` (CHARMM K [kcal/mol/Å²], r₀ [Å] per type pair).
//!
//! Conversions applied here (no judgment calls, just unit bookkeeping):
//! - LJ stored as **true σ** (`R*/2 × R_STAR_TO_SIGMA`), engine convention
//!   identical to protein atoms (bio_files `LjParams::from_line`).
//! - Bond stiffness: CHARMM's `E = K(b−b₀)²` becomes the engine restraint's
//!   `E = ½k(b−b₀)²` form, so `k = 2K`.
//! - Charges ×`CHARGE_UNIT_SCALER` at site construction.
//!
//! Known caveats, deliberately cataloged instead of hidden:
//! - CGenFF was fitted with CHARMM's *geometric* σ combination rule; the
//!   engine pairs by arithmetic (Lorentz-Berthelot). Cross-terms against
//!   protein/water shift mildly — standard hybrid-FF risk, and the proper
//!   fix is a QC/GEFF re-fit against our combining rule, NOT a mixing-rule
//!   change in the engine (a per-pair re-fit is the only proven route;
//!   engineering around it here would silently change every other pair).
//! - Site `offset` scaffolds are ideal geometries built from the published
//!   bond lengths (angles: standard D3h/C2v/tetrahedral placement). Geometry
//!   is a starting layout, NOT a parameter — bond restraints pin the
//!   published distances, minimization relaxes everything else.
//! - `GDMCL` carries its Cl⁻ as an 11th, UNBONDED site at 5 Å off-plane so
//!   each inserted formula unit is net-neutral (engine displacement is one
//!   water slot per molecule; the pair may contact-relax — Gdm⁺Cl⁻ contact
//!   ion pairs are real solution chemistry).

use crate::engine::md_core::{CHARGE_UNIT_SCALER, R_STAR_TO_SIGMA};
use crate::engine::md_core::{CosolventSite, CosolventSpec};
use lin_alg::f32::Vec3;

/// Amber/CHARMM R_min/2 → engine true σ.
const fn rs(r_star_over_2: f32) -> f32 {
    r_star_over_2 * R_STAR_TO_SIGMA
}

/// CHARMM bond K (E = K·Δr²) → engine restraint k (E = ½·k·Δr²).
const fn ck(k_charm: f32) -> f32 {
    2.0 * k_charm
}

fn q(e_charge: f32) -> f32 {
    e_charge * CHARGE_UNIT_SCALER
}

fn v3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

/// Rotate an in-plane vector by `deg` degrees about z.
fn rot_z(v: Vec3, deg: f32) -> Vec3 {
    let r = deg.to_radians();
    v3(
        v.x * r.cos() - v.y * r.sin(),
        v.x * r.sin() + v.y * r.cos(),
        v.z,
    )
}

/// Amide/guanidinium NH2 pair: two H at 1.00 Å from N, directions obtained
/// by rotating the N→C vector ±120° in the molecular plane (published bond
/// length; planar sp2 nitrogen ⇒ 120° skeleton).
fn nh2_planar(n: Vec3, c: Vec3, n_h_len: f32) -> [Vec3; 2] {
    let mut v = c - n;
    v = v * (1.0 / v.magnitude());
    [
        n + rot_z(v, 120.0) * n_h_len,
        n + rot_z(v, -120.0) * n_h_len,
    ]
}

/// Methyl tripod: three H at C–H length, H–C–N = 109.47° (tetrahedral),
/// azimuths 0/120/240 around the N→C axis.
fn ch3_tripod(n: Vec3, c: Vec3, c_h_len: f32) -> [Vec3; 3] {
    let u = (c - n) * (1.0 / (c - n).magnitude());
    // Any perpendicular; y-cross trick robust except when u is near y.
    let seed = if u.y.abs() < 0.9 {
        v3(0.0, 1.0, 0.0)
    } else {
        v3(1.0, 0.0, 0.0)
    };
    let mut p = seed - u * (seed.dot(u));
    p = p * (1.0 / p.magnitude());
    let qq = u.cross(p);
    // H–C–N = 109.47° ⇒ the H direction makes 70.53° with the N→C axis:
    // dot(h,u) = cos(70.53°) = 1/3.
    let along = 1.0f32 / 3.0f32;
    let perp = (1.0 - along * along).sqrt();
    (0..3)
        .map(|i| {
            let phi = std::f32::consts::TAU * (i as f32) / 3.0;
            let dir = u * along + (p * phi.cos() + qq * phi.sin()) * perp;
            c + dir * c_h_len
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

/// Urea, CO(NH2)2, neutral — 8 sites.
///
/// Charges/types: `top_all36_cgenff_v4.6.rtf` `RESI UREA` (N1 NG2S2 −0.69,
/// H11/H12/H31/H32 HGP1 +0.34, C2 CG2O6 +0.60, O2 OG2D1 −0.58).
/// LJ: `par_all36_cgenff_v4.6.prm` NONBONDED (NG2S2 0.200/1.850, HGP1
/// 0.046/0.2245, CG2O6 0.070/2.000, OG2D1 0.120/1.700).
/// Bonds: C=O 650/1.2300, C–N 430/1.3600, N–H 480/1.00 (BONDS section).
pub fn urea() -> CosolventSpec {
    // Planar C2v scaffold, C2 at the water-slot origin.
    let (c2, o2) = (v3(0.0, 0.0, 0.0), v3(1.23, 0.0, 0.0));
    let n1 = v3(
        1.36 * 120f32.to_radians().cos(),
        1.36 * 120f32.to_radians().sin(),
        0.0,
    ); // (−0.68, +1.1778): azimuth 120°, r(C–N)=1.36
    let n3 = v3(n1.x, -n1.y, 0.0);
    let [h11, h12] = nh2_planar(n1, c2, 1.00);
    let [h31, h32] = nh2_planar(n3, c2, 1.00);

    let ng = |offset| CosolventSite {
        ff_type: "NG2S2".into(),
        element: na_seq::Element::Nitrogen,
        mass: 14.0067,
        charge_scaled: q(-0.69),
        sigma: rs(1.850),
        eps: 0.2000,
        c4: 0.0,
        offset,
    };
    let hg = |offset| CosolventSite {
        ff_type: "HGP1".into(),
        element: na_seq::Element::Hydrogen,
        mass: 1.00795,
        charge_scaled: q(0.34),
        sigma: rs(0.2245),
        eps: 0.0460,
        c4: 0.0,
        offset,
    };
    CosolventSpec {
        name: "UREA".into(),
        molarity: 0.0,
        sites: vec![
            ng(n1),
            hg(h11),
            hg(h12),
            CosolventSite {
                ff_type: "CG2O6".into(),
                element: na_seq::Element::Carbon,
                mass: 12.011,
                charge_scaled: q(0.60),
                sigma: rs(2.000),
                eps: 0.0700,
                c4: 0.0,
                offset: c2,
            },
            CosolventSite {
                ff_type: "OG2D1".into(),
                element: na_seq::Element::Oxygen,
                mass: 15.9994,
                charge_scaled: q(-0.58),
                sigma: rs(1.700),
                eps: 0.1200,
                c4: 0.0,
                offset: o2,
            },
            ng(n3),
            hg(h31),
            hg(h32),
        ],
        // (i,j,k,r0) — the v1.3 `CosolventSpec::bonds` order; k = 2×CHARMM K
        // so E = ½kΔr² matches CHARMM's KΔr².
        bonds: vec![
            (0, 3, ck(430.0), 1.3600),
            (3, 4, ck(650.0), 1.2300),
            (3, 5, ck(430.0), 1.3600),
            (0, 1, ck(480.0), 1.00),
            (0, 2, ck(480.0), 1.00),
            (5, 6, ck(480.0), 1.00),
            (5, 7, ck(480.0), 1.00),
        ],
    }
}

/// TMAO, (CH3)3N→O, neutral — 14 sites.
///
/// RTF `RESI TMAO`: N NG3P0 −0.83, C1..C3 CG334 −0.35, O1 OG312 −0.37,
/// H×9 HGP5 +0.25. NONBONDED: NG3P0 0.200/1.850, CG334 0.077/2.2150,
/// OG312 0.120/1.750, HGP5 0.046/0.7000. BONDS: C–N 215/1.5100,
/// C–H 300/1.0800, N–O 310/1.4000.
pub fn tmao() -> CosolventSpec {
    let n = v3(0.0, 0.0, 0.0);
    let o1 = v3(0.0, 0.0, 1.40);
    // Three methyl C at tetrahedral angle from the N→O axis, r(N–C)=1.51.
    let cs: Vec<Vec3> = (0..3)
        .map(|i| {
            let phi = std::f32::consts::TAU * (i as f32) / 3.0;
            let (sin_t, cos_t) = (109.47f32).to_radians().sin_cos(); // from +z
            v3(
                1.51 * sin_t * phi.cos(),
                1.51 * sin_t * phi.sin(),
                1.51 * cos_t,
            )
        })
        .collect();
    let hs: Vec<Vec3> = cs.iter().flat_map(|c| ch3_tripod(n, *c, 1.08)).collect();

    let c334 = |offset| CosolventSite {
        ff_type: "CG334".into(),
        element: na_seq::Element::Carbon,
        mass: 12.011,
        charge_scaled: q(-0.35),
        sigma: rs(2.2150),
        eps: 0.0770,
        c4: 0.0,
        offset,
    };
    let hgp = |offset| CosolventSite {
        ff_type: "HP".into(),
        element: na_seq::Element::Hydrogen,
        mass: 1.00795,
        charge_scaled: q(0.25),
        sigma: rs(0.7000),
        eps: 0.0460,
        c4: 0.0,
        offset,
    };
    let mut sites = vec![CosolventSite {
        ff_type: "NG3P0".into(),
        element: na_seq::Element::Nitrogen,
        mass: 14.0067,
        charge_scaled: q(-0.83),
        sigma: rs(1.850),
        eps: 0.2000,
        c4: 0.0,
        offset: n,
    }];
    for i in 0..3 {
        sites.push(c334(cs[i]));
        sites.push(hgp(hs[3 * i]));
        sites.push(hgp(hs[3 * i + 1]));
        sites.push(hgp(hs[3 * i + 2]));
    }
    sites.push(CosolventSite {
        ff_type: "OG312".into(),
        element: na_seq::Element::Oxygen,
        mass: 15.9994,
        charge_scaled: q(-0.37),
        sigma: rs(1.750),
        eps: 0.1200,
        c4: 0.0,
        offset: o1,
    });
    // site order: 0=N, then per methyl [C,H,H,H] (1..4, 5..8, 9..12), 13=O.
    // (i,j,k,r0) order (see module docs).
    let mut bonds = vec![
        (0usize, 1usize, ck(215.0), 1.5100f32),
        (0, 13, ck(310.0), 1.4000),
    ];
    for i in 0..3 {
        bonds.push((0, 1 + 4 * i, ck(215.0), 1.5100));
        for j in 0..3 {
            bonds.push((1 + 4 * i, 2 + 4 * i + j, ck(300.0), 1.0800));
        }
    }
    bonds.remove(0); // the first N–C was re-added in the loop
    CosolventSpec {
        name: "TMAO".into(),
        molarity: 0.0,
        sites,
        bonds,
    }
}

/// Guanidinium chloride, Gdm⁺Cl⁻, net-neutral formula unit — 11 sites.
///
/// RTF `RESI GUAN` (net +1): C CG2N1 +0.64, N1..N3 NG2P1 −0.80,
/// H×6 HGP2 +0.46; Cl⁻ as site 10 (engine's Li–Merz-OPC Cl⁻ exactly).
/// BONDS: C–N 463/1.3650, N–H 455/1.00. The Cl⁻ is deliberately UNBONDED
/// (module docs); it starts 5 Å off the guanidinium plane.
pub fn gdmcl() -> CosolventSpec {
    let c = v3(0.0, 0.0, 0.0);
    let ns: Vec<Vec3> = (0..3)
        .map(|i| {
            let phi = std::f32::consts::TAU * (i as f32) / 3.0;
            v3(1.365 * phi.cos(), 1.365 * phi.sin(), 0.0)
        })
        .collect();
    let mut sites = Vec::new();
    for npos in ns.iter() {
        let [ha, hb] = nh2_planar(*npos, c, 1.00);
        sites.push(CosolventSite {
            ff_type: "NG2P1".into(),
            element: na_seq::Element::Nitrogen,
            mass: 14.0067,
            charge_scaled: q(-0.80),
            sigma: rs(1.850),
            eps: 0.2000,
            c4: 0.0,
            offset: *npos,
        });
        for h in [ha, hb] {
            sites.push(CosolventSite {
                ff_type: "HGP2".into(),
                element: na_seq::Element::Hydrogen,
                mass: 1.00795,
                charge_scaled: q(0.46),
                sigma: rs(0.2245),
                eps: 0.0460,
                c4: 0.0,
                offset: h,
            });
        }
    }
    // Interleave would be nicer but RTF order GROUP is N,H,H per N; C first:
    let mut ordered = vec![CosolventSite {
        ff_type: "CG2N1".into(),
        element: na_seq::Element::Carbon,
        mass: 12.011,
        charge_scaled: q(0.64),
        sigma: rs(2.000),
        eps: 0.1100,
        c4: 0.0,
        offset: c,
    }];
    ordered.extend(sites);
    ordered.push(CosolventSite {
        ff_type: "Cl-".into(),
        element: na_seq::Element::Chlorine,
        mass: 35.45,
        charge_scaled: -CHARGE_UNIT_SCALER,
        sigma: crate::engine::md_core::ION_CL.sigma,
        eps: crate::engine::md_core::ION_CL.eps,
        c4: 0.0,
        offset: v3(0.0, 0.0, -5.0),
    });
    let mut bonds = Vec::new();
    // site order: 0=C; N,H,H triplets at 1, 4, 7; Cl⁻ at 10 (unbonded).
    // (i,j,k,r0) order.
    for k in [1usize, 4, 7] {
        bonds.push((0usize, k, ck(463.0), 1.3650f32));
        for j in 0..2 {
            bonds.push((k, k + 1 + j, ck(455.0), 1.00));
        }
    }
    CosolventSpec {
        name: "GDMCL".into(),
        molarity: 0.0,
        sites: ordered,
        bonds,
    }
}

/// Case-insensitive lookup of the vendored presets. Accepts the common
/// spellings so Python users do not need to memorize keys.
pub fn cosolvent_preset(name: &str) -> Option<CosolventSpec> {
    match name.trim().to_ascii_uppercase().as_str() {
        "UREA" => Some(urea()),
        "TMAO" => Some(tmao()),
        "GDMCL" | "GDM+" | "GDM_CL" | "GUAN" => Some(gdmcl()),
        _ => None,
    }
}

/// Vendored preset names, for error messages / docs.
pub const PRESET_NAMES: [&str; 3] = ["UREA", "TMAO", "GDMCL"];

// ---------------------------------------------------------------- JSON I/O

#[derive(serde::Deserialize)]
struct JsSpec {
    name: String,
    #[serde(default)]
    molarity: f32,
    #[serde(default)]
    sites: Option<Vec<JsSite>>,
    #[serde(default)]
    bonds: Option<Vec<[f32; 4]>>,
}

#[derive(serde::Deserialize)]
struct JsSite {
    ff_type: String,
    element: String,
    mass: f32,
    /// Partial charge in elementary charges (scaled by the engine at ingest).
    charge: f32,
    /// LJ parameters: exactly one of `sigma` (true σ, Å) or `rmin_over_2`
    /// (Amber/CHARMM R_min/2) must be given.
    #[serde(default)]
    sigma: Option<f32>,
    #[serde(rename = "rmin_over_2", default)]
    rmin_over_2: Option<f32>,
    eps: f32,
    /// 12-6-4 induction coefficient (Å⁴·kcal/mol); omit or 0 = plain 12-6.
    #[serde(default)]
    c4: f32,
    offset: [f32; 3],
}

/// Parse the Python/FFI `cosolvents_json` payload.
///
/// Grammar: a JSON array. Each element is either a preset reference
/// `{"name": "UREA", "molarity": 3.0}` or a full custom spec
/// `{"name": "X", "molarity": 0.5, "sites": [{"ff_type":"C","element":"C",
/// "mass":12.011,"charge":-0.2,"rmin_over_2":1.908,"eps":0.12,
/// "offset":[-1.0,0.0,0.0]}], "bonds": [[0,1,900.0,2.0]]}`
/// (`bonds` = [i, j, k, r0] — same tuple order as `CosolventSpec::bonds`,
/// engine restraint form `E = ½kΔr²`; a site may carry `c4` — 12-6-4
/// induction coefficient, default 0).
/// An empty string yields the empty list.
pub fn parse_cosolvents_json(s: &str) -> Result<Vec<CosolventSpec>, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Vec::new());
    }
    let items: Vec<JsSpec> =
        serde_json::from_str(s).map_err(|e| format!("cosolvents_json: {e}"))?;
    items.iter().try_fold(Vec::new(), |mut acc, item| {
        acc.push(spec_from_json(item)?);
        Ok::<_, String>(acc)
    })
}

fn spec_from_json(item: &JsSpec) -> Result<CosolventSpec, String> {
    let custom = item.sites.as_ref().is_some() || item.bonds.as_ref().is_some();
    if !custom {
        let mut spec = cosolvent_preset(&item.name).ok_or_else(|| {
            format!(
                "cosolvents_json: '{}' is not a vendored preset ({:?}) and carries no sites[]",
                item.name, PRESET_NAMES
            )
        })?;
        if item.molarity.is_nan() || item.molarity < 0.0 {
            return Err(format!(
                "cosolvents_json: {} molarity must be >= 0",
                item.name
            ));
        }
        spec.molarity = item.molarity;
        return Ok(spec);
    }
    let js_sites = item.sites.as_ref().ok_or_else(|| {
        format!(
            "cosolvents_json: '{}' sites[] required with bonds[]",
            item.name
        )
    })?;
    if js_sites.is_empty() {
        return Err(format!("cosolvents_json: '{}' sites[] empty", item.name));
    }
    let mut sites = Vec::with_capacity(js_sites.len());
    for (i, js) in js_sites.iter().enumerate() {
        let sigma = match (js.sigma, js.rmin_over_2) {
            (Some(s_), None) => s_,
            (None, Some(r)) => rs(r),
            _ => {
                return Err(format!(
                    "cosolvents_json: '{}': site {i} needs exactly one of sigma | rmin_over_2",
                    item.name
                ));
            }
        };
        let element = na_seq::Element::from_letter(&js.element).map_err(|_| {
            format!(
                "cosolvents_json: unknown element '{}' (site {i})",
                js.element
            )
        })?;
        sites.push(CosolventSite {
            ff_type: js.ff_type.clone(),
            element,
            mass: js.mass,
            charge_scaled: q(js.charge),
            sigma,
            eps: js.eps,
            c4: js.c4,
            offset: v3(js.offset[0], js.offset[1], js.offset[2]),
        });
    }
    let bonds = item.bonds.clone().unwrap_or_default();
    let bonds = bonds
        .into_iter()
        .map(|b| {
            let (i, j) = (b[0] as usize, b[1] as usize);
            if i >= sites.len() || j >= sites.len() || i == j {
                return Err(format!(
                    "cosolvents_json: '{}': bond ({}, {}) out of site range",
                    item.name, b[0], b[1]
                ));
            }
            Ok((i, j, b[2], b[3]))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CosolventSpec {
        name: item.name.clone().to_ascii_uppercase(),
        molarity: item.molarity,
        sites,
        bonds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn net_charge(spec: &CosolventSpec) -> f64 {
        spec.sites
            .iter()
            .map(|s| f64::from(s.charge_scaled))
            .sum::<f64>()
            / (CHARGE_UNIT_SCALER as f64)
    }

    fn check_bond_scaffolds(spec: &CosolventSpec) {
        for &(i, j, _, r0) in &spec.bonds {
            let d = (spec.sites[i].offset - spec.sites[j].offset).magnitude();
            assert!(
                (d - r0).abs() < 1e-3,
                "{} bond ({i},{j}) scaffold distance {d:.4} vs r0 {r0}",
                spec.name
            );
        }
    }

    #[test]
    fn urea_preset_is_neutral_and_self_consistent() {
        let u = urea();
        assert_eq!(u.sites.len(), 8);
        assert!((net_charge(&u)).abs() < 1e-4);
        assert_eq!(u.bonds.len(), 7);
        check_bond_scaffolds(&u);
    }

    #[test]
    fn tmao_preset_is_neutral_and_self_consistent() {
        let t = tmao();
        assert_eq!(t.sites.len(), 14);
        assert!((net_charge(&t)).abs() < 1e-4);
        // 3×N–C + N–O + 3×3 C–H = 3 + 1 + 9 = 13 bonds, no duplicates.
        assert_eq!(t.bonds.len(), 13);
        let mut keys: Vec<(usize, usize)> = t
            .bonds
            .iter()
            .map(|&(i, j, _, _)| (i.min(j), i.max(j)))
            .collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), 13);
        check_bond_scaffolds(&t);
    }

    #[test]
    fn gdmcl_preset_is_net_neutral() {
        let g = gdmcl();
        assert_eq!(g.sites.len(), 11);
        assert!((net_charge(&g)).abs() < 1e-4);
        // The guanidinium core alone must be exactly +1 e.
        let core: f64 = g.sites[..10]
            .iter()
            .map(|s| f64::from(s.charge_scaled))
            .sum::<f64>()
            / (CHARGE_UNIT_SCALER as f64);
        assert!((core - 1.0).abs() < 1e-4);
        assert_eq!(g.bonds.len(), 3 + 6);
        check_bond_scaffolds(&g);
        // The chloride rides unbonded, 5 Å off-plane, engine Cl− exactly.
        assert_eq!(g.sites[10].ff_type, "Cl-");
        assert!(g.bonds.iter().all(|&(i, j, _, _)| i != 10 && j != 10));
        assert_eq!(g.sites[10].sigma, crate::engine::md_core::ION_CL.sigma);
    }

    #[test]
    fn preset_lookup_is_case_insensitive() {
        assert!(cosolvent_preset("urea").is_some());
        assert!(cosolvent_preset(" Gdm+ ").is_some());
        assert!(cosolvent_preset("TMAO").is_some());
        assert!(cosolvent_preset("water").is_none());
    }

    #[test]
    fn json_preset_reference_and_custom_round_trip() {
        let empty = parse_cosolvents_json("").unwrap();
        assert!(empty.is_empty());
        let specs = parse_cosolvents_json(
            r#"[{"name":"UREA","molarity":3.0},
                {"name":"custom","molarity":0.5,
                 "sites":[{"ff_type":"A","element":"C","mass":12.0,"charge":0.25,
                           "rmin_over_2":1.9,"eps":0.1,"offset":[-1,0,0]},
                          {"ff_type":"B","element":"O","mass":16.0,"charge":-0.25,
                           "sigma":3.0,"eps":0.2,"offset":[1,0,0]}],
                 "bonds":[[0,1,900.0,2.0]]}]"#,
        )
        .unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].name, "UREA");
        assert_eq!(specs[0].molarity, 3.0);
        assert_eq!(specs[1].sites.len(), 2);
        assert_eq!(specs[1].bonds[0], (0, 1, 900.0, 2.0));
        // rmin_over_2 honored:
        assert_eq!(specs[1].sites[0].sigma, rs(1.9));
    }

    #[test]
    fn json_rejects_bad_inputs() {
        // Unknown preset without sites[] → clear error naming it.
        let e = parse_cosolvents_json(r#"[{"name":"MYSTERY","molarity":1.0}]"#).unwrap_err();
        assert!(e.contains("MYSTERY"), "{e}");
        // Both sigma fields given.
        let e = parse_cosolvents_json(
            r#"[{"name":"X","molarity":1.0,"sites":[
                {"ff_type":"A","element":"C","mass":12.0,"charge":0.0,
                 "sigma":3.0,"rmin_over_2":1.5,"eps":0.1,"offset":[0,0,0]}]}]"#,
        )
        .unwrap_err();
        assert!(e.contains("exactly one"), "{e}");
        // Bond index out of range.
        let e = parse_cosolvents_json(
            r#"[{"name":"X","molarity":1.0,"sites":[
                {"ff_type":"A","element":"C","mass":12.0,"charge":0.0,
                 "sigma":3.0,"eps":0.1,"offset":[0,0,0]}],
                "bonds":[[0,1,2.0,100.0]]}]"#,
        )
        .unwrap_err();
        assert!(e.contains("out of site range"), "{e}");
        // Unknown element (na_seq from_letter is strict).
        let e = parse_cosolvents_json(
            r#"[{"name":"X","molarity":1.0,"sites":[
                {"ff_type":"A","element":"Zz9","mass":1.0,"charge":0.0,
                 "sigma":3.0,"eps":0.1,"offset":[0,0,0]}]}]"#,
        )
        .unwrap_err();
        assert!(e.contains("unknown element"), "{e}");
    }
}
