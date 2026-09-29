use super::*;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default)]
struct IndexHasher(u64);

impl Hasher for IndexHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        let mut value = 0u64;
        for (shift, byte) in bytes.iter().take(8).enumerate() {
            value |= (*byte as u64) << (shift * 8);
        }
        self.0 = value;
    }
    fn write_usize(&mut self, value: usize) {
        self.0 = value as u64;
    }
}

type IndexMap<K, V> = HashMap<K, V, BuildHasherDefault<IndexHasher>>;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NumericSimdPair {
    pub(super) tgt: u32,
    pub(super) src: u32,
    pub(super) sigma: f32,
    pub(super) epsilon: f32,
    /// 12-6-4 pair constant (0 = bitwise no-op in the fused kernel).
    pub(super) c4: f32,
    pub(super) charge_product: f32,
}

use crate::engine::md_core::{
    AtomDynamics, MdOverrides,
    barostat::SimBox,
    solvent::{ForcesOnWaterMol, O_EPS, O_SIGMA, WaterMolOpc, WaterSite},
};
#[allow(unused)]
#[cfg(target_arch = "x86_64")]
use crate::engine::md_core::{AtomDynamicsx8, AtomDynamicsx16};

// // Å. 9-12 should be fine; there is very little VDW force > this range due to
// // the ^-7 falloff.
// pub const CUTOFF_VDW: f32 = 12.0;

// Ewald SPME approximation for Coulomb force

// Instead of a hard cutoff between short and long-range forces, these
// parameters control a smooth taper.
// Our neighbor list must use the same cutoff as this, so we use it directly.

// The distance beyond which we truncate the real-space erfc-screened interaction.
// This is not used for the reciprical part.
// We don't use a taper, for now.
// const LONG_RANGE_SWITCH_START: f64 = 8.0; // start switching (Å)

// pub const LONG_RANGE_CUTOFF: f32 = 12.0; // Å

// // A bigger α means more damping, and a smaller real-space contribution. (Cheaper real), but larger
// // reciprocal load.
// // Common rule for α: erfc(α r_c) ≲ 10⁻⁴…10⁻⁵
// pub const EWALD_ALPHA: f32 = 0.35; // Å^-1. 0.35 is good for cutoff of 10–12 Å.

// See Amber RM, section 15, "1-4 Non-Bonded Interaction Scaling"
// "Non-bonded interactions between atoms separated by three consecutive bonds... require a special
// treatment in Amber force fields."
// "By default, vdW 1-4 interactions are divided (scaled down) by a factor of 2.0, electrostatic 1-4 terms by a factor
// of 1.2."
pub(super) const SCALE_LJ_14: f32 = 0.5;
pub const SCALE_COUL_14: f32 = 1.0 / 1.2;

// Multiply by this to convert partial charges from elementary charge (What we store in Atoms loaded from mol2
// files and amino19.lib.) to the self-consistent amber units required to calculate Coulomb force.
// We apply this to dynamic and static atoms when building Indexed params, and to solvent molecules
// on their construction. We do not apply this during integration.
// Electrostatic constant: 332.0522 kcal·Å/(mol·e²). This is the square root of that.
pub const CHARGE_UNIT_SCALER: f32 = 18.2223;

/// We use this to load the correct data from LJ lookup tables. Since we use indices,
/// we must index correctly into the dynamic, or static tables. We have single-index lookups
/// for atoms acting on solvent, since there is only one O LJ type.
#[derive(Debug, Clone)]
pub enum LjTableIndices {
    /// (tgt, src)
    StdStd((usize, usize)),
    /// (dyn tgt or src))
    StdWater(usize),
    /// One value, stored as a constant (Water O -> Water O)
    WaterWater,
}

/// We cache σ and ε on the first step, then use it on the others. This increases
/// memory use, and reduces CPU use. We use indices, as they're faster than HashMaps.
/// The indices are flattened, of each interaction pair. Values are (σ, ε).
///
/// Water-solvent is not included, as it's a single, hard-coded parameter pair.
#[derive(Default, Clone)]
pub struct LjTables {
    /// Non-solvent, non-solvent interactions. Upper triangle, (σ, ε, c4).
    /// `c4` is the pair 12-6-4 constant: geometric mean when BOTH atoms
    /// carry one (PGY-style site×ion pair), else 0 — two ions never induce
    /// through a shared site, and single-sided solute C4 is reserved for
    /// the ion–water-O channel (water_std) so existing protein behavior is
    /// bit-for-bit untouched.
    pub std: Vec<(f32, f32, f32)>,
    /// Water O, non-solvent interactions: (σ, ε, c4) with c4 = the solute
    /// atom's own ion–water Li–Merz constant (0 for every non-ion).
    pub water_std: Vec<(f32, f32, f32)>,
    pub n_std: usize,
}

#[allow(unused)]
#[cfg(target_arch = "x86_64")]
#[derive(Default)]
pub struct LjTablesx8 {
    /// Non-solvent, non-solvent interactions. Upper triangle.
    pub std: Vec<(f32x8, f32x8)>,
    /// Water, non-solvent interactions.
    pub water_std: Vec<(f32x8, f32x8)>,
    pub n_std: [usize; 8],
}

#[allow(unused)]
#[cfg(target_arch = "x86_64")]
#[derive(Default)]
pub struct LjTablesx16 {
    /// Non-solvent, non-solvent interactions. Upper triangle.
    pub std: Vec<(f32x16, f32x16)>,
    /// Water, non-solvent interactions.
    pub water_std: Vec<(f32x16, f32x16)>,
    pub n_std: [usize; 8],
}

// todo note: On large systems, this can have very high memory use. Consider
// todo setting up your table by atom type, instead of by atom, if that proves to be a problem.
impl LjTables {
    /// Create an indexed table, flattened.
    pub fn new(atoms: &[AtomDynamics]) -> Self {
        let n_std = atoms.len();

        if n_std == 0 {
            // Otherwise, we will get an out-of-bounds error when subtracting.
            return Default::default();
        }

        // Construct an upper triangle table, excluding reverse order, and self interactions.
        let mut std = Vec::with_capacity(n_std * (n_std - 1) / 2);

        for (i_0, atom_0) in atoms.iter().enumerate() {
            for (i_1, atom_1) in atoms.iter().enumerate() {
                if i_1 <= i_0 {
                    continue;
                }
                let (σ, ε, c4) = combine_lj_params(atom_0, atom_1);
                std.push((σ, ε, c4));
            }
        }

        // One LJ pair per dynamic atom vs solvent O:
        let mut water_std = Vec::with_capacity(n_std);
        for atom in atoms {
            let σ = 0.5 * (atom.lj_sigma + O_SIGMA);
            let ε = (atom.lj_eps * O_EPS).sqrt();
            // One-sided by Li–Merz design: the ION's pair constant acts on
            // the water oxygen; the water carries none of its own.
            water_std.push((σ, ε, atom.lj_c4));
        }

        Self {
            std,
            water_std,
            n_std,
        }
    }

    /// Get (σ, ε, c4) — c4 follows the pair conventions documented on
    /// [`LjTables`]; 0 disables the 12-6-4 term for the pair (bit-identical
    /// to plain 12-6 in both the scalar and SIMD kernels).
    pub fn lookup(&self, i: &LjTableIndices) -> (f32, f32, f32) {
        match i {
            LjTableIndices::StdStd((i_0, i_1)) => {
                // Map to (i<j), then index into the packed upper triangle (row-major).
                let (i, j) = if i_0 < i_1 {
                    (*i_0, *i_1)
                } else {
                    (*i_1, *i_0)
                };

                if i >= self.n_std {
                    println!("I > i: {i} std: {}", self.n_std);
                }

                if j >= self.n_std {
                    println!("J > J: {j} std: {}", self.n_std);
                }

                // Elements before row i: sum_{r=0}^{i-1} (N-1-r) = i*(2N - i - 1)/2
                // Offset within row i: (j - i - 1)
                let idx = i * (2 * self.n_std - i - 1) / 2 + (j - i - 1);

                self.std[idx]
            }
            LjTableIndices::StdWater(ix) => self.water_std[*ix],
            LjTableIndices::WaterWater => (O_SIGMA, O_EPS, 0.0),
        }
    }
}

impl AddAssign<Self> for ForcesOnWaterMol {
    fn add_assign(&mut self, rhs: Self) {
        self.f_o += rhs.f_o;
        self.f_h0 += rhs.f_h0;
        self.f_h1 += rhs.f_h1;
        self.f_m += rhs.f_m;
    }
}

#[derive(Copy, Clone)]
pub enum BodyRef {
    NonWater(usize),
    // Static(usize),
    Water { mol: usize, site: WaterSite },
}

impl BodyRef {
    pub(crate) fn get<'a>(
        &self,
        non_waters: &'a [AtomDynamics],
        waters: &'a [WaterMolOpc],
    ) -> &'a AtomDynamics {
        match *self {
            BodyRef::NonWater(i) => &non_waters[i],
            BodyRef::Water { mol, site } => match site {
                WaterSite::O => &waters[mol].o,
                WaterSite::M => &waters[mol].m,
                WaterSite::H0 => &waters[mol].h0,
                WaterSite::H1 => &waters[mol].h1,
            },
        }
    }
}

#[derive(Clone)]
pub struct NonBondedPair {
    pub tgt: BodyRef,
    pub src: BodyRef,
    pub scale_14: bool,
    pub lj_indices: LjTableIndices,
    pub calc_lj: bool,
    pub calc_coulomb: bool,
    pub symmetric: bool,
    /// True when this pair is a cross interaction between the alchemical molecule
    /// and the rest of the system, and therefore should use alchemical LJ/Coulomb
    /// handling.
    /// False unless using an alchemical free-energy computation.
    pub alch_interaction: bool,
    /// True for the COMPACT rigid-water pairs: one entry per water–water molecule
    /// pair (or per std–water pair) whose dedicated kernel computes all the
    /// site–site interactions (O-O LJ + 3×3 charged-site Coulomb) in a single
    /// call, sharing one minimum-image displacement. The per-site expansion is
    /// used instead when `alch_interaction` requires soft-core handling.
    pub water_full: bool,
}

/// Compact CPU scalar pair record. It intentionally mirrors only the fields
/// consumed by the scalar kernel and is independent from the legacy GPU pair
/// representation stored in `MdState::cpu_pairs`.
#[derive(Clone)]
pub(crate) struct CompactNonBondedPair {
    pub tgt: BodyRef,
    pub src: BodyRef,
    pub scale_14: bool,
    pub lj_indices: LjTableIndices,
    pub calc_lj: bool,
    pub calc_coulomb: bool,
    pub symmetric: bool,
    pub alch_interaction: bool,
    pub water_full: bool,
}

impl From<&NonBondedPair> for CompactNonBondedPair {
    fn from(p: &NonBondedPair) -> Self {
        Self {
            tgt: p.tgt,
            src: p.src,
            scale_14: p.scale_14,
            lj_indices: p.lj_indices.clone(),
            calc_lj: p.calc_lj,
            calc_coulomb: p.calc_coulomb,
            symmetric: p.symmetric,
            alch_interaction: p.alch_interaction,
            water_full: p.water_full,
        }
    }
}

/// Continuous worker-local standard-atom force accumulator.
#[derive(Clone, Default)]
pub(super) struct StdForceSoA {
    /// Sparse target-owned forces. Workers only allocate entries they touch.
    values: IndexMap<usize, Vec3F64>,
}
impl StdForceSoA {
    fn resize_zero(&mut self, _n: usize) {
        self.values.clear();
    }
    #[inline]
    fn add(&mut self, i: usize, f: Vec3F64) {
        *self.values.entry(i).or_insert_with(Vec3F64::new_zero) += f;
    }
    #[inline]
    fn get(&self, i: usize) -> Vec3F64 {
        self.values
            .get(&i)
            .copied()
            .unwrap_or_else(Vec3F64::new_zero)
    }
    #[inline]
    fn set(&mut self, i: usize, f: Vec3F64) {
        self.values.insert(i, f);
    }
    fn into_vec(self, n: usize) -> Vec<Vec3F64> {
        let mut out = vec![Vec3F64::new_zero(); n];
        for (i, f) in self.values {
            if i < n {
                out[i] = f;
            }
        }
        out
    }
}

/// Add a force into the right accumulator (std or solvent). Static never accumulates.
pub(super) fn add_to_sink(
    sink_non_water: &mut StdForceSoA,
    sink_wat: &mut [ForcesOnWaterMol],
    body_type: BodyRef,
    f: Vec3F64,
) {
    match body_type {
        BodyRef::NonWater(i) => sink_non_water.add(i, f),
        BodyRef::Water { mol, site } => add_water_site_force(&mut sink_wat[mol], site, f),
        // BodyRef::Static(_) => (),
    }
}

/// Add a force to a single site of a water molecule's accumulator.
#[inline]
pub(super) fn add_water_site_force(f: &mut ForcesOnWaterMol, site: WaterSite, v: Vec3F64) {
    match site {
        WaterSite::O => f.f_o += v,
        WaterSite::M => f.f_m += v,
        WaterSite::H0 => f.f_h0 += v,
        WaterSite::H1 => f.f_h1 += v,
    }
}

use std::cell::RefCell;

thread_local! {
    static CACHED_STD_FORCES: RefCell<StdForceSoA> = RefCell::new(StdForceSoA::default());
    static CACHED_WAT_FORCES: RefCell<Vec<ForcesOnWaterMol>> = RefCell::new(Vec::new());
    static CACHED_MOL_FORCES: RefCell<Vec<f64>> = RefCell::new(Vec::new());
}

/// Applies non-bonded force in parallel (CPU thread-pool) over a set of atoms, with indices assigned
/// upstream.
///
/// Returns (forces on non-solvent atoms, forces on solvent molecules, virial, potential energy total,
/// potential energy between molecule pairs. (kcal/mol)
pub(super) fn calc_force_cpu(
    pairs: &[CompactNonBondedPair],
    atoms_std: &[AtomDynamics],
    water: &[WaterMolOpc],
    cell: &SimBox,
    lj_tables: &LjTables,
    overrides: &MdOverrides,
    mol_start_indices: &[usize],
    // For alchemical free-energy computation. Ignored unless a pair has
    // alch_interaction = true.
    lambda_alch: f64,
    spme_alpha: f32,
    coulomb_cutoff: f32,
    lj_cutoff: f32,
) -> (Vec<Vec3F64>, Vec<ForcesOnWaterMol>, f64, f64, Vec<f64>, f64) {
    let n_std = atoms_std.len();
    let n_wat = water.len();
    let n_mol = mol_start_indices.len();
    // Dense molecule-pair energy is opt-in: it is O(n_mol²) in every Rayon
    // worker and is not part of the production force contract.
    let track_molecule_energy =
        n_mol <= 256 && std::env::var_os("SPICE_DENSE_MOLECULE_ENERGY").is_some_and(|v| v == "1");
    let atom_to_mol = atom_to_mol_indices(n_std, mol_start_indices);

    let result = pairs
        .par_iter()
        .fold(
            || {
                let mut f_std = CACHED_STD_FORCES.with(|c| std::mem::take(&mut *c.borrow_mut()));
                let mut f_wat = CACHED_WAT_FORCES.with(|c| std::mem::take(&mut *c.borrow_mut()));
                let mut f_mol = CACHED_MOL_FORCES.with(|c| std::mem::take(&mut *c.borrow_mut()));

                f_std.resize_zero(n_std);

                f_wat.truncate(0);
                f_wat.resize(n_wat, ForcesOnWaterMol::default());

                f_mol.truncate(0);
                if track_molecule_energy {
                    f_mol.resize(n_mol * n_mol, 0.0_f64);
                }

                (
                    f_std, f_wat, 0.0_f64, // Virial sum
                    0.0_f64, // Energy sum
                    f_mol,   // Per-pair
                    0.0_f64, // Alchemical dH/dlambda
                )
            },
            |(
                mut f_std,
                mut f_wat,
                mut virial,
                mut energy,
                mut energy_between_mols,
                mut alch_dh_dl,
            ),
             p| {
                let mut e_pair = 0.0f32;
                let mut dh_dl_pair = 0.0f32;

                if p.water_full {
                    // Compact rigid-water pairs: the dedicated kernels compute
                    // all site-site interactions and accumulate directly into
                    // the per-thread force sinks (no per-site add_to_sink).
                    match (p.tgt, p.src) {
                        (BodyRef::Water { mol: mi, .. }, BodyRef::Water { mol: mj, .. }) => {
                            // Disjoint mutable borrows of f_wat at two indices.
                            let (f_wa, f_wb) = if mi < mj {
                                let (lo, hi) = f_wat.split_at_mut(mj);
                                (&mut lo[mi], &mut hi[0])
                            } else {
                                let (lo, hi) = f_wat.split_at_mut(mi);
                                (&mut hi[0], &mut lo[mj])
                            };
                            // v1.3.8 pressure-audit closure: the scalar tail
                            // must book water-water direct energy exactly like
                            // the SIMD lanes do (eval_water_batches always
                            // folded it). Discarding it here made U depend on
                            // batch parity — same system, different energy per
                            // 8-lane tail alignment.
                            e_pair = f_water_water_cpu(
                                &mut virial,
                                f_wa,
                                f_wb,
                                &water[mi],
                                &water[mj],
                                cell,
                                lj_tables,
                                overrides,
                                spme_alpha,
                                coulomb_cutoff,
                                lj_cutoff,
                            ) as f32;
                        }
                        (BodyRef::NonWater(i), BodyRef::Water { mol, .. }) => {
                            let mut f_i = f_std.get(i);
                            e_pair = f_water_std_cpu(
                                &mut virial,
                                &mut f_i,
                                &mut f_wat[mol],
                                &atoms_std[i],
                                &water[mol],
                                cell,
                                lj_tables,
                                overrides,
                                spme_alpha,
                                coulomb_cutoff,
                                lj_cutoff,
                                i,
                            ) as f32;
                            f_std.set(i, f_i);
                        }
                        (BodyRef::Water { mol, .. }, BodyRef::NonWater(i)) => {
                            // Not emitted (std is always tgt), but symmetric-safe.
                            let mut f_i = f_std.get(i);
                            e_pair = f_water_std_cpu(
                                &mut virial,
                                &mut f_i,
                                &mut f_wat[mol],
                                &atoms_std[i],
                                &water[mol],
                                cell,
                                lj_tables,
                                overrides,
                                spme_alpha,
                                coulomb_cutoff,
                                lj_cutoff,
                                i,
                            ) as f32;
                            f_std.set(i, f_i);
                        }
                        _ => unreachable!("water_full pair without any water"),
                    }
                } else {
                    let a_t = p.tgt.get(atoms_std, water);
                    let a_s = p.src.get(atoms_std, water);

                    let alchemical_lambda = p
                        .alch_interaction
                        .then_some(lambda_alch.clamp(0.0, 1.0) as f32);

                    let (f, ee, dd) = f_nonbonded_cpu(
                        &mut virial,
                        a_t,
                        a_s,
                        cell,
                        p.scale_14,
                        &p.lj_indices,
                        lj_tables,
                        p.calc_lj,
                        p.calc_coulomb,
                        overrides,
                        spme_alpha,
                        coulomb_cutoff,
                        lj_cutoff,
                        alchemical_lambda,
                    );
                    e_pair = ee;
                    dh_dl_pair = dd;

                    // Convert to f64 prior to summing.
                    let f: Vec3F64 = f.into();
                    add_to_sink(&mut f_std, &mut f_wat, p.tgt, f);
                    if p.symmetric {
                        add_to_sink(&mut f_std, &mut f_wat, p.src, -f);
                    }
                }

                // v1.3.8 pressure-audit closure: solvent–solvent direct energy
                // is part of the total now. The old skip was paired with the
                // scalar-tail discard above; with the SIMD fold always having
                // counted water-water, excluding it here double-standard'd U
                // by batch parity. std–std, std–solvent, solvent–solvent all
                // fold into the same bucket.
                energy += e_pair as f64;

                if p.alch_interaction {
                    alch_dh_dl += dh_dl_pair as f64;
                }

                // Optional dense molecule-pair energy analysis.
                if track_molecule_energy
                    && let (BodyRef::NonWater(i_tgt), BodyRef::NonWater(i_src)) = (p.tgt, p.src)
                {
                    let m_t = atom_to_mol[i_tgt];
                    let m_s = atom_to_mol[i_src];
                    let idx_ts = m_t * n_mol + m_s;
                    energy_between_mols[idx_ts] += e_pair as f64;

                    // make it symmetric so callers don't have to
                    if m_t != m_s {
                        let idx_st = m_s * n_mol + m_t;
                        energy_between_mols[idx_st] += e_pair as f64;
                    }
                }

                (
                    f_std,
                    f_wat,
                    virial,
                    energy,
                    energy_between_mols,
                    alch_dh_dl,
                )
            },
        )
        .reduce(
            || {
                (
                    StdForceSoA::default(),
                    vec![ForcesOnWaterMol::default(); n_wat],
                    0.0_f64,
                    0.0_f64,
                    if track_molecule_energy {
                        vec![0.0_f64; mol_start_indices.len() * mol_start_indices.len()]
                    } else {
                        Vec::new()
                    },
                    0.0_f64,
                )
            },
            |(mut f_on_std, mut f_on_water, virial_a, e_a, mut em_a, dhdl_a),
             (db, wb, virial_b, e_b, em_b, dhdl_b)| {
                for (i, force) in db.values.iter() {
                    f_on_std.add(*i, *force);
                }
                for i in 0..n_wat {
                    f_on_water[i].f_o += wb[i].f_o;
                    f_on_water[i].f_m += wb[i].f_m;
                    f_on_water[i].f_h0 += wb[i].f_h0;
                    f_on_water[i].f_h1 += wb[i].f_h1;
                }

                // Merge per-molecule energy
                for i in 0..em_a.len() {
                    em_a[i] += em_b[i];
                }

                // Recycle the db, wb, em_b vectors into thread-local caches
                CACHED_STD_FORCES.with(|c| {
                    let mut b = c.borrow_mut();
                    if b.values.capacity() < db.values.capacity() {
                        *b = db;
                    }
                });
                CACHED_WAT_FORCES.with(|c| {
                    let mut b = c.borrow_mut();
                    if b.capacity() < wb.capacity() {
                        *b = wb;
                    }
                });
                CACHED_MOL_FORCES.with(|c| {
                    let mut b = c.borrow_mut();
                    if b.capacity() < em_b.capacity() {
                        *b = em_b;
                    }
                });

                // (f_on_std, f_on_water, virial_a + virial_b, e_a + e_b)
                (
                    f_on_std,
                    f_on_water,
                    virial_a + virial_b,
                    e_a + e_b,
                    em_a,
                    dhdl_a + dhdl_b,
                )
            },
        );
    let (f_std, f_wat, virial, energy, energy_between_mols, dhdl) = result;
    (
        f_std.into_vec(n_std),
        f_wat,
        virial,
        energy,
        energy_between_mols,
        dhdl,
    )
}

// #[cfg(target_arch = "x86_64")]
// fn calc_force_x8(
//     pairs: &[NonBondedPair],
//     atoms_std: &[AtomDynamicsx8],
//     solvent: &[WaterMolx8],
//     cell: &SimBox,
//     lj_tables: &LjTablesx8,
// ) -> (Vec<Vec3x8>, Vec<ForcesOnWaterMol>, f64, f64) {
// }
//
// #[cfg(target_arch = "x86_64")]
// fn calc_force_x16(
//     pairs: &[NonBondedPair],
//     atoms_std: &[AtomDynamicsx16],
//     solvent: &[WaterMolx16],
//     cell: &SimBox,
//     lj_tables: &LjTablesx16,
// ) -> (Vec<Vec3x16>, Vec<ForcesOnWaterMol>, f64, f64) {
// }
