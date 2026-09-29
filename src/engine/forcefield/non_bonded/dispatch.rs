use std::cell::RefCell;

use super::*;

/// Abramowitz-Stegun erfc approximation used by the x86_64 SIMD path.
///
/// NOTE: never `wide`'s `.recip()` here — on SSE/AVX it is the raw
/// `_mm*_*_rcp_ps` (relative error ~1.5e-4) while on NEON/simd128 it is an
/// exact division. A force kernel must not branch its physics on the ISA:
/// the production x86 wheels must reproduce the arm64-validated numbers
/// (caught by the CI x86 numeric gate, v1.3.10). Use IEEE division.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(super) fn erfc_approx_x8(x: WideF32x8) -> WideF32x8 {
    let t = WideF32x8::splat(1.0) / (WideF32x8::splat(1.0) + WideF32x8::splat(0.3275911) * x);
    let poly = ((((WideF32x8::splat(1.061405429) * t + WideF32x8::splat(-1.453152027)) * t
        + WideF32x8::splat(1.421413741))
        * t
        + WideF32x8::splat(-0.284496736))
        * t
        + WideF32x8::splat(0.254829592))
        * t;
    poly * (-(x * x)).exp()
}

/// Per-lane outcome of the vectorized std–std chunk kernels: the force on the
/// target atom, the pair energy, and the r·F virial contribution. Masked
/// (skin-only) lanes return zeros, so applying every lane is equivalent to the
/// old per-lane `continue`.
#[derive(Clone, Copy, Default)]
pub(super) struct StdLaneOut {
    pub tgt: u32,
    pub src: u32,
    pub f: Vec3,
    pub e: f32,
    pub w: f64,
}

/// Worker-local accumulation state for the std–std SIMD stream.
pub(super) struct StdSimdAccum {
    pub f_std: Vec<Vec3F64>,
    pub energy_between_mols: Vec<f64>,
    pub virial: f64,
    pub energy: f64,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
thread_local! {
    /// Recycled per-worker dense solute force scratch for the SIMD chunk fold,
    /// mirroring the scalar path's `CACHED_STD_FORCES` policy (keep the larger
    /// capacity).
    static STD_CHUNK_SCRATCH: RefCell<Vec<Vec3F64>> = const { RefCell::new(Vec::new()) };
}

/// Run the std–std 8-lane kernel over the whole pair stream in parallel.
///
/// Pairs are independent; worker-local dense accumulators are merged by a
/// fixed-order reduce — the same determinism contract `calc_force_cpu`'s fold
/// gives. (This loop used to run serially on every step, which dominated the
/// evaluation on all cores of an M-series machine.)
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(super) fn eval_std_simd_chunks<K>(
    simd_pairs: &[NumericSimdPair],
    n_std: usize,
    n_mol: usize,
    atom_to_mol: &[usize],
    track_molecule_energy: bool,
    lanes_kernel: K,
) -> StdSimdAccum
where
    K: Fn(&[NumericSimdPair]) -> [StdLaneOut; 8] + Sync,
{
    let init = || {
        let mut f_std = STD_CHUNK_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
        f_std.clear();
        f_std.resize(n_std, Vec3F64::new_zero());
        StdSimdAccum {
            f_std,
            energy_between_mols: if track_molecule_energy {
                vec![0.0; n_mol * n_mol]
            } else {
                Vec::new()
            },
            virial: 0.0,
            energy: 0.0,
        }
    };
    simd_pairs
        .par_chunks_exact(8)
        .fold(init, |mut acc, chunk| {
            for lane in lanes_kernel(chunk) {
                let i = lane.tgt as usize;
                let j = lane.src as usize;
                acc.f_std[i] += Vec3F64::from(lane.f);
                acc.f_std[j] -= Vec3F64::from(lane.f);
                acc.energy += lane.e as f64;
                acc.virial += lane.w;
                if !acc.energy_between_mols.is_empty() {
                    let mi = atom_to_mol[i];
                    let mj = atom_to_mol[j];
                    acc.energy_between_mols[mi * n_mol + mj] += lane.e as f64;
                    if mi != mj {
                        acc.energy_between_mols[mj * n_mol + mi] += lane.e as f64;
                    }
                }
            }
            acc
        })
        .reduce(init, |mut a, mut b| {
            for (x, y) in a.f_std.iter_mut().zip(b.f_std.iter()) {
                *x += *y;
            }
            for (x, y) in a
                .energy_between_mols
                .iter_mut()
                .zip(b.energy_between_mols.iter())
            {
                *x += *y;
            }
            a.virial += b.virial;
            a.energy += b.energy;
            STD_CHUNK_SCRATCH.with(|c| {
                let mut s = c.borrow_mut();
                if s.capacity() < b.f_std.capacity() {
                    std::mem::swap(&mut *s, &mut b.f_std);
                }
            });
            a
        })
}

/// x86_64 fused LJ + short-range-Coulomb 8-lane kernel for one chunk. The
/// arithmetic is unchanged from the previous in-loop body; only the
/// accumulation moved into the parallel driver.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn std_simd_lanes_x86(
    chunk: &[NumericSimdPair],
    atoms_std: &[AtomDynamics],
    cell: &SimBox,
    spme_alpha: f32,
    coulomb_cutoff: f32,
    lj_cutoff: f32,
    max_cutoff_sq: f32,
    soa_x: Option<&[f32]>,
    soa_y: Option<&[f32]>,
    soa_z: Option<&[f32]>,
) -> [StdLaneOut; 8] {
    let mut diffs = [Vec3::new_zero(); 8];
    let mut sigmas = [0.0f32; 8];
    let mut epsilons = [0.0f32; 8];
    let mut c4s = [0.0f32; 8];
    let mut q_products = [0.0f32; 8];
    let mut tgt_idx = [0u32; 8];
    let mut src_idx = [0u32; 8];
    for (lane, p) in chunk.iter().enumerate() {
        let tgt = p.tgt as usize;
        let src = p.src as usize;
        let diff = match (soa_x, soa_y, soa_z) {
            (Some(x), Some(y), Some(z)) if tgt < x.len() && src < x.len() => {
                cell.min_image(Vec3::new(x[tgt] - x[src], y[tgt] - y[src], z[tgt] - z[src]))
            }
            _ => cell.min_image(atoms_std[tgt].posit - atoms_std[src].posit),
        };
        diffs[lane] = diff;
        sigmas[lane] = p.sigma;
        epsilons[lane] = p.epsilon;
        c4s[lane] = p.c4;
        q_products[lane] = p.charge_product;
        tgt_idx[lane] = p.tgt;
        src_idx[lane] = p.src;
    }
    // Neighbor lists include the skin shell. A whole batch outside the
    // largest active cutoff contributes exactly zero.
    if diffs.iter().all(|d| d.magnitude_squared() > max_cutoff_sq) {
        return [StdLaneOut::default(); 8];
    }
    let dx = WideF32x8::new(diffs.map(|v| v.x));
    let dy = WideF32x8::new(diffs.map(|v| v.y));
    let dz = WideF32x8::new(diffs.map(|v| v.z));
    // IEEE division (see erfc_approx_x8's ISA note): NOT `.recip()`.
    let inv_dist = WideF32x8::splat(1.0) / (dx * dx + dy * dy + dz * dz).sqrt();
    let sigma = WideF32x8::new(sigmas);
    let epsilon = WideF32x8::new(epsilons);
    let c4 = WideF32x8::new(c4s);
    let sr = sigma * inv_dist;
    let sr2 = sr * sr;
    let sr6 = sr2 * sr2 * sr2;
    let sr12 = sr6 * sr6;
    let inv2 = inv_dist * inv_dist;
    let inv4 = inv2 * inv2;
    let lj_mag = WideF32x8::splat(24.0) * epsilon * (WideF32x8::splat(2.0) * sr12 - sr6) * inv_dist
        - WideF32x8::splat(4.0) * c4 * inv4 * inv_dist;
    let lj_energy = WideF32x8::splat(4.0) * epsilon * (sr12 - sr6) - c4 * inv4;
    let dist = WideF32x8::splat(1.0) / inv_dist;
    let alpha_r = dist * WideF32x8::splat(spme_alpha);
    let erfc = erfc_approx_x8(alpha_r);
    let qprod = WideF32x8::new(q_products);
    let exp_term = (-(alpha_r * alpha_r)).exp();
    let coul_energy = qprod * inv_dist * erfc;
    let coul_mag = qprod
        * (erfc * inv_dist * inv_dist
            + WideF32x8::splat(1.1283791670955126 * spme_alpha) * exp_term * inv_dist);
    let total_mag = lj_mag + coul_mag;
    let fx = (dx * inv_dist * total_mag).to_array();
    let fy = (dy * inv_dist * total_mag).to_array();
    let fz = (dz * inv_dist * total_mag).to_array();
    let distances = dist.to_array();
    let lj_e = lj_energy.to_array();
    let c_e = coul_energy.to_array();
    let c_fx = (dx * inv_dist * coul_mag).to_array();
    let c_fy = (dy * inv_dist * coul_mag).to_array();
    let c_fz = (dz * inv_dist * coul_mag).to_array();
    let mut out = [StdLaneOut::default(); 8];
    for lane in 0..8 {
        let r = distances[lane];
        // The neighbor list contains a skin shell. This lane mask removes
        // shell-only pairs before force and energy accumulation.
        if diffs[lane].magnitude_squared() > max_cutoff_sq {
            continue;
        }
        let mut f_lj = Vec3::new(
            fx[lane] - c_fx[lane],
            fy[lane] - c_fy[lane],
            fz[lane] - c_fz[lane],
        );
        let mut e_lj = lj_e[lane].clamp(-1.0e6, 1.0e6);
        if r > lj_cutoff {
            f_lj = Vec3::new_zero();
            e_lj = 0.0;
        } else {
            let fm = f_lj.magnitude();
            if fm > 1.0e4 {
                f_lj *= 1.0e4 / fm;
            }
        }
        let mut f_coul = Vec3::new(c_fx[lane], c_fy[lane], c_fz[lane]);
        let mut e_coul = c_e[lane];
        if r > coulomb_cutoff {
            f_coul = Vec3::new_zero();
            e_coul = 0.0;
        }
        let f = f_lj + f_coul;
        out[lane] = StdLaneOut {
            tgt: tgt_idx[lane],
            src: src_idx[lane],
            f,
            e: e_lj + e_coul,
            w: diffs[lane].dot(f) as f64,
        };
    }
    out
}

/// aarch64 conservative kernel: SIMD LJ with scalar short-range Coulomb per
/// lane (the established arm64 semantics — vectorized erfc/exp there is
/// slower than the accurate scalar ewald call).
#[cfg(target_arch = "aarch64")]
fn std_simd_lanes_arm(
    chunk: &[NumericSimdPair],
    atoms_std: &[AtomDynamics],
    cell: &SimBox,
    spme_alpha: f32,
    coulomb_cutoff: f32,
    lj_cutoff: f32,
) -> [StdLaneOut; 8] {
    let mut diffs = [Vec3::new_zero(); 8];
    let mut sigmas = [0.0f32; 8];
    let mut epsilons = [0.0f32; 8];
    let mut c4s = [0.0f32; 8];
    let mut tgt_idx = [0u32; 8];
    let mut src_idx = [0u32; 8];
    for (lane, p) in chunk.iter().enumerate() {
        let diff =
            cell.min_image(atoms_std[p.tgt as usize].posit - atoms_std[p.src as usize].posit);
        diffs[lane] = diff;
        sigmas[lane] = p.sigma;
        epsilons[lane] = p.epsilon;
        c4s[lane] = p.c4;
        tgt_idx[lane] = p.tgt;
        src_idx[lane] = p.src;
    }
    let dx = WideF32x8::new(diffs.map(|v| v.x));
    let dy = WideF32x8::new(diffs.map(|v| v.y));
    let dz = WideF32x8::new(diffs.map(|v| v.z));
    // IEEE division (see erfc_approx_x8's ISA note): NOT `.recip()`.
    let inv_dist = WideF32x8::splat(1.0) / (dx * dx + dy * dy + dz * dz).sqrt();
    let sigma = WideF32x8::new(sigmas);
    let epsilon = WideF32x8::new(epsilons);
    let c4 = WideF32x8::new(c4s);
    let sr = sigma * inv_dist;
    let sr2 = sr * sr;
    let sr6 = sr2 * sr2 * sr2;
    let sr12 = sr6 * sr6;
    let inv2 = inv_dist * inv_dist;
    let inv4 = inv2 * inv2;
    let lj_mag = WideF32x8::splat(24.0) * epsilon * (WideF32x8::splat(2.0) * sr12 - sr6) * inv_dist
        - WideF32x8::splat(4.0) * c4 * inv4 * inv_dist;
    let lj_energy = WideF32x8::splat(4.0) * epsilon * (sr12 - sr6) - c4 * inv4;
    let fx = (dx * inv_dist * lj_mag).to_array();
    let fy = (dy * inv_dist * lj_mag).to_array();
    let fz = (dz * inv_dist * lj_mag).to_array();
    let distances = (WideF32x8::splat(1.0) / inv_dist).to_array();
    let lj_e = lj_energy.to_array();
    let mut out = [StdLaneOut::default(); 8];
    for lane in 0..8 {
        let i = tgt_idx[lane] as usize;
        let r = distances[lane];
        let mut f_lj = Vec3::new(fx[lane], fy[lane], fz[lane]);
        let mut e_lj = lj_e[lane].clamp(-1.0e6, 1.0e6);
        if r > lj_cutoff {
            f_lj = Vec3::new_zero();
            e_lj = 0.0;
        } else {
            let fm = f_lj.magnitude();
            if fm > 1.0e4 {
                f_lj *= 1.0e4 / fm;
            }
        }
        let inv = 1.0 / r;
        let dir = diffs[lane] * inv;
        let (mut f_coul, mut e_coul) = force_coulomb_short_range(
            dir,
            r,
            inv,
            atoms_std[i].partial_charge,
            atoms_std[src_idx[lane] as usize].partial_charge,
            coulomb_cutoff,
            spme_alpha,
        );
        if r > coulomb_cutoff {
            f_coul = Vec3::new_zero();
            e_coul = 0.0;
        }
        let f = f_lj + f_coul;
        out[lane] = StdLaneOut {
            tgt: tgt_idx[lane],
            src: src_idx[lane],
            f,
            e: e_lj + e_coul,
            w: diffs[lane].dot(f) as f64,
        };
    }
    out
}

#[cfg(target_arch = "x86_64")]
/// SIMD dispatch for ordinary standard-atom pairs plus the compact water
/// batches. Complex pair classes (1-4 scaling, alchemical terms, and the
/// SIMD tails) keep the scalar reference path so they retain their existing
/// semantics.
pub(super) fn calc_force_cpu_dispatch(
    pairs: &[CompactNonBondedPair],
    preclassified_simd: &[NumericSimdPair],
    preclassified_scalar: &[CompactNonBondedPair],
    water_simd_pairs: &[WaterSoluteBatch8],
    water_water_simd_pairs: &[WaterWaterBatch8],
    atoms_std: &[AtomDynamics],
    water: &[WaterMolOpc],
    cell: &SimBox,
    lj_tables: &LjTables,
    overrides: &MdOverrides,
    mol_start_indices: &[usize],
    lambda_alch: f64,
    spme_alpha: f32,
    coulomb_cutoff: f32,
    lj_cutoff: f32,
    soa_x: Option<&[f32]>,
    soa_y: Option<&[f32]>,
    soa_z: Option<&[f32]>,
) -> (Vec<Vec3F64>, Vec<ForcesOnWaterMol>, f64, f64, Vec<f64>, f64) {
    if overrides.lj_disabled
        || overrides.coulomb_disabled
        || std::env::var_os("SPICE_NONBONDED_REFERENCE").is_some_and(|v| v == "1")
    {
        return calc_force_cpu(
            pairs,
            atoms_std,
            water,
            cell,
            lj_tables,
            overrides,
            mol_start_indices,
            lambda_alch,
            spme_alpha,
            coulomb_cutoff,
            lj_cutoff,
        );
    }
    let n_std = atoms_std.len();
    let n_mol = mol_start_indices.len();
    let atom_to_mol = atom_to_mol_indices(n_std, mol_start_indices);
    let track_molecule_energy =
        n_mol <= 256 && std::env::var_os("SPICE_DENSE_MOLECULE_ENERGY").is_some_and(|v| v == "1");
    // Pair classes are prepared during neighbor-list rebuild. Reuse the
    // contiguous lists on every force evaluation instead of reclassifying and
    // cloning the complete pair list in the MD hot loop.
    let simd_pairs = preclassified_simd;
    let remaining = preclassified_scalar;
    // Water batches were built in the same rebuild as these pair streams, so
    // `remaining` contains exactly their scalar tails.
    let (ws_f_std, water_simd_force, ws_virial, ws_energy) = eval_water_batches(
        water_simd_pairs,
        water_water_simd_pairs,
        atoms_std,
        water,
        cell,
        spme_alpha,
        coulomb_cutoff,
        lj_cutoff,
    );
    let max_cutoff_sq = coulomb_cutoff.max(lj_cutoff).powi(2);
    let accum = eval_std_simd_chunks(
        simd_pairs,
        n_std,
        n_mol,
        &atom_to_mol,
        track_molecule_energy,
        |chunk| {
            std_simd_lanes_x86(
                chunk,
                atoms_std,
                cell,
                spme_alpha,
                coulomb_cutoff,
                lj_cutoff,
                max_cutoff_sq,
                soa_x,
                soa_y,
                soa_z,
            )
        },
    );

    // `setup_pairs` places incomplete SIMD tails into the scalar cache, so
    // there is no per-step temporary pair vector here.
    let (mut rf, mut rw, rv, re, mut rem, rd) = calc_force_cpu(
        remaining,
        atoms_std,
        water,
        cell,
        lj_tables,
        overrides,
        mol_start_indices,
        lambda_alch,
        spme_alpha,
        coulomb_cutoff,
        lj_cutoff,
    );
    merge_water_force_map(&mut rw, water_simd_force);
    for (a, b) in rf.iter_mut().zip(accum.f_std) {
        *a += b;
    }
    for (a, b) in rf.iter_mut().zip(ws_f_std) {
        *a += b;
    }
    for (a, b) in rem.iter_mut().zip(accum.energy_between_mols) {
        *a += b;
    }
    (
        rf,
        rw,
        rv + accum.virial + ws_virial,
        re + accum.energy + ws_energy,
        rem,
        rd,
    )
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(super) fn calc_force_cpu_dispatch_arm_lj(
    pairs: &[CompactNonBondedPair],
    preclassified_simd: &[NumericSimdPair],
    preclassified_scalar: &[CompactNonBondedPair],
    water_simd_pairs: &[WaterSoluteBatch8],
    water_water_simd_pairs: &[WaterWaterBatch8],
    atoms_std: &[AtomDynamics],
    water: &[WaterMolOpc],
    cell: &SimBox,
    lj_tables: &LjTables,
    overrides: &MdOverrides,
    mol_start_indices: &[usize],
    lambda_alch: f64,
    spme_alpha: f32,
    coulomb_cutoff: f32,
    lj_cutoff: f32,
) -> (Vec<Vec3F64>, Vec<ForcesOnWaterMol>, f64, f64, Vec<f64>, f64) {
    let n_std = atoms_std.len();
    let n_mol = mol_start_indices.len();
    let atom_to_mol = atom_to_mol_indices(n_std, mol_start_indices);
    let track_molecule_energy =
        n_mol <= 256 && std::env::var_os("SPICE_DENSE_MOLECULE_ENERGY").is_some_and(|v| v == "1");
    // Pair classes are prepared during neighbor-list rebuild. Reuse the
    // contiguous lists on every force evaluation instead of reclassifying and
    // cloning the complete pair list in the MD hot loop.
    if overrides.lj_disabled
        || overrides.coulomb_disabled
        || std::env::var_os("SPICE_NONBONDED_REFERENCE").is_some_and(|v| v == "1")
    {
        return calc_force_cpu(
            pairs,
            atoms_std,
            water,
            cell,
            lj_tables,
            overrides,
            mol_start_indices,
            lambda_alch,
            spme_alpha,
            coulomb_cutoff,
            lj_cutoff,
        );
    }
    let simd_pairs = preclassified_simd;
    let remaining = preclassified_scalar;
    // The portable 8-lane water kernels run on NEON via `wide`; setup paired
    // the batch coverage with this consumer (see `WATER_SIMD_ACTIVE`).
    let (ws_f_std, water_simd_force, ws_virial, ws_energy) = eval_water_batches(
        water_simd_pairs,
        water_water_simd_pairs,
        atoms_std,
        water,
        cell,
        spme_alpha,
        coulomb_cutoff,
        lj_cutoff,
    );
    let accum = eval_std_simd_chunks(
        simd_pairs,
        n_std,
        n_mol,
        &atom_to_mol,
        track_molecule_energy,
        |chunk| {
            std_simd_lanes_arm(
                chunk,
                atoms_std,
                cell,
                spme_alpha,
                coulomb_cutoff,
                lj_cutoff,
            )
        },
    );

    // `setup_pairs` places incomplete SIMD tails into the scalar cache, so
    // there is no per-step temporary pair vector here.
    let (mut rf, mut rw, rv, re, mut rem, rd) = calc_force_cpu(
        remaining,
        atoms_std,
        water,
        cell,
        lj_tables,
        overrides,
        mol_start_indices,
        lambda_alch,
        spme_alpha,
        coulomb_cutoff,
        lj_cutoff,
    );
    merge_water_force_map(&mut rw, water_simd_force);
    for (a, b) in rf.iter_mut().zip(accum.f_std) {
        *a += b;
    }
    for (a, b) in rf.iter_mut().zip(ws_f_std) {
        *a += b;
    }
    for (a, b) in rem.iter_mut().zip(accum.energy_between_mols) {
        *a += b;
    }
    (
        rf,
        rw,
        rv + accum.virial + ws_virial,
        re + accum.energy + ws_energy,
        rem,
        rd,
    )
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(super) fn calc_force_cpu_dispatch(
    pairs: &[CompactNonBondedPair],
    preclassified_simd: &[NumericSimdPair],
    preclassified_scalar: &[CompactNonBondedPair],
    water_simd_pairs: &[WaterSoluteBatch8],
    water_water_simd_pairs: &[WaterWaterBatch8],
    atoms_std: &[AtomDynamics],
    water: &[WaterMolOpc],
    cell: &SimBox,
    lj_tables: &LjTables,
    overrides: &MdOverrides,
    mol_start_indices: &[usize],
    lambda_alch: f64,
    spme_alpha: f32,
    coulomb_cutoff: f32,
    lj_cutoff: f32,
    _soa_x: Option<&[f32]>,
    _soa_y: Option<&[f32]>,
    _soa_z: Option<&[f32]>,
) -> (Vec<Vec3F64>, Vec<ForcesOnWaterMol>, f64, f64, Vec<f64>, f64) {
    calc_force_cpu_dispatch_arm_lj(
        pairs,
        preclassified_simd,
        preclassified_scalar,
        water_simd_pairs,
        water_water_simd_pairs,
        atoms_std,
        water,
        cell,
        lj_tables,
        overrides,
        mol_start_indices,
        lambda_alch,
        spme_alpha,
        coulomb_cutoff,
        lj_cutoff,
    )
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(super) fn calc_force_cpu_dispatch(
    pairs: &[CompactNonBondedPair],
    _preclassified_simd: &[NumericSimdPair],
    _preclassified_scalar: &[CompactNonBondedPair],
    _water_simd_pairs: &[WaterSoluteBatch8],
    _water_water_simd_pairs: &[WaterWaterBatch8],
    atoms_std: &[AtomDynamics],
    water: &[WaterMolOpc],
    cell: &SimBox,
    lj_tables: &LjTables,
    overrides: &MdOverrides,
    mol_start_indices: &[usize],
    lambda_alch: f64,
    spme_alpha: f32,
    coulomb_cutoff: f32,
    lj_cutoff: f32,
    _soa_x: Option<&[f32]>,
    _soa_y: Option<&[f32]>,
    _soa_z: Option<&[f32]>,
) -> (Vec<Vec3F64>, Vec<ForcesOnWaterMol>, f64, f64, Vec<f64>, f64) {
    calc_force_cpu(
        pairs,
        atoms_std,
        water,
        cell,
        lj_tables,
        overrides,
        mol_start_indices,
        lambda_alch,
        spme_alpha,
        coulomb_cutoff,
        lj_cutoff,
    )
}
