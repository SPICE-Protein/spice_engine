use super::*;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NumericSimdPair {
    pub(super) tgt: u32,
    pub(super) src: u32,
    pub(super) sigma: f32,
    pub(super) epsilon: f32,
    pub(super) charge_product: f32,
}

/// Preclassified compact standard-atom/water pair. The water molecule is
/// represented by its index (not a BodyRef/site enum); the fixed OPC site
/// expansion happens inside the dedicated kernel. Parameters that are
/// invariant for the pair are captured at neighbor rebuild time.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(super) struct NumericWaterPair {
    pub(crate) std: u32,
    pub(crate) water: u32,
    pub(crate) sigma: f32,
    pub(crate) epsilon: f32,
    pub(crate) charge_product_m: f32,
    pub(crate) charge_product_h0: f32,
    pub(crate) charge_product_h1: f32,
}

/// Eight-lane compact water pair input.  Indices and parameters are prepared
/// during neighbour-list construction; the kernel itself performs no table or
/// enum lookup.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy, Debug)]
pub(super) struct WaterPairBatch8 {
    pub std: [u32; 8],
    pub water: [u32; 8],
    pub sigma: WideF32x8,
    pub epsilon: WideF32x8,
    pub q_m: WideF32x8,
    pub q_h0: WideF32x8,
    pub q_h1: WideF32x8,
}

#[cfg(target_arch = "aarch64")]
impl WaterPairBatch8 {
    fn from_pairs(pairs: &[NumericWaterPair]) -> Option<Self> {
        if pairs.len() < 8 {
            return None;
        }
        let mut std = [0u32; 8];
        let mut water = [0u32; 8];
        let mut sigma = [0.0; 8];
        let mut epsilon = [0.0; 8];
        let mut q_m = [0.0; 8];
        let mut q_h0 = [0.0; 8];
        let mut q_h1 = [0.0; 8];
        for (lane, p) in pairs[..8].iter().enumerate() {
            std[lane] = p.std;
            water[lane] = p.water;
            sigma[lane] = p.sigma;
            epsilon[lane] = p.epsilon;
            q_m[lane] = p.charge_product_m;
            q_h0[lane] = p.charge_product_h0;
            q_h1[lane] = p.charge_product_h1;
        }
        Some(Self {
            std,
            water,
            sigma: WideF32x8::new(sigma),
            epsilon: WideF32x8::new(epsilon),
            q_m: WideF32x8::new(q_m),
            q_h0: WideF32x8::new(q_h0),
            q_h1: WideF32x8::new(q_h1),
        })
    }
}

#[cfg(feature = "cuda")]
use crate::engine::md_core::gpu_interface::force_nonbonded_gpu;
use crate::engine::md_core::{
    AtomDynamics, ComputationDevice, MdOverrides, MdState,
    alchemical::{
        SOFT_CORE_ALPHA, SOFT_CORE_POWER, SOFT_CORE_SIGMA_MIN, staged_decoupling_schedule,
    },
    barostat::SimBox,
    forces::force_e_lj,
    solvent::{ForcesOnWaterMol, O_EPS, O_H_R, O_SIGMA, WaterMolOpc, WaterSite},
    validate_mol_start_indices,
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
    /// Non-solvent, non-solvent interactions. Upper triangle.
    pub std: Vec<(f32, f32)>,
    /// Water, non-solvent interactions.
    pub water_std: Vec<(f32, f32)>,
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
                let (σ, ε) = combine_lj_params(atom_0, atom_1);
                std.push((σ, ε));
            }
        }

        // One LJ pair per dynamic atom vs solvent O:
        let mut water_std = Vec::with_capacity(n_std);
        for atom in atoms {
            let σ = 0.5 * (atom.lj_sigma + O_SIGMA);
            let ε = (atom.lj_eps * O_EPS).sqrt();
            water_std.push((σ, ε));
        }

        Self {
            std,
            water_std,
            n_std,
        }
    }

    /// Get (σ, ε)
    pub fn lookup(&self, i: &LjTableIndices) -> (f32, f32) {
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
            LjTableIndices::WaterWater => (O_SIGMA, O_EPS),
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

/// Continuous worker-local standard-atom force accumulator.
#[derive(Clone, Default)]
pub(super) struct StdForceSoA {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
}
impl StdForceSoA {
    fn resize_zero(&mut self, n: usize) {
        self.x.clear();
        self.x.resize(n, 0.0);
        self.y.clear();
        self.y.resize(n, 0.0);
        self.z.clear();
        self.z.resize(n, 0.0);
    }
    #[inline]
    fn add(&mut self, i: usize, f: Vec3F64) {
        self.x[i] += f.x;
        self.y[i] += f.y;
        self.z[i] += f.z;
    }
    #[inline]
    fn get(&self, i: usize) -> Vec3F64 {
        Vec3F64::new(self.x[i], self.y[i], self.z[i])
    }
    #[inline]
    fn set(&mut self, i: usize, f: Vec3F64) {
        self.x[i] = f.x;
        self.y[i] = f.y;
        self.z[i] = f.z;
    }
    fn into_vec(self) -> Vec<Vec3F64> {
        self.x
            .into_iter()
            .zip(self.y)
            .zip(self.z)
            .map(|((x, y), z)| Vec3F64::new(x, y, z))
            .collect()
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
    pairs: &[NonBondedPair],
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
    let track_molecule_energy = n_mol <= 256;
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
                            // Water-water energy is intentionally discarded
                            // (solvent-only energy isn't part of the total).
                            let _ = f_water_water_cpu(
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
                            );
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

                // We are not interested, in this point, at potential energy that only involves solvent atoms.
                // We skip solvent-solvent.
                let involves_std =
                    matches!(p.tgt, BodyRef::NonWater(_)) || matches!(p.src, BodyRef::NonWater(_));

                if involves_std {
                    energy += e_pair as f64;
                }

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
                    StdForceSoA {
                        x: vec![0.0; n_std],
                        y: vec![0.0; n_std],
                        z: vec![0.0; n_std],
                    },
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
                for i in 0..n_std {
                    f_on_std.x[i] += db.x[i];
                    f_on_std.y[i] += db.y[i];
                    f_on_std.z[i] += db.z[i];
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
                    if b.x.capacity() < db.x.capacity() {
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
        f_std.into_vec(),
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
