//! Portable 8-lane SIMD kernels for ordinary OPC water interactions.
//!
//! Lanes are independent compact molecule pairs.  This module deliberately
//! keeps scatter/reduction outside the vector arithmetic: callers receive one
//! result per lane and apply it to the owning solute/water accumulators.

use std::cell::RefCell;

use super::*;
use lin_alg::f32::Vec3;
use wide::f32x8;

thread_local! {
    /// Recycled per-evaluation SoA water-site snapshot (see `WaterSites`).
    static WATER_SITES_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Per-evaluation snapshot of the water-site geometry the SIMD batches
/// consume. The batch kernels used to gather O positions, site offsets, and
/// charges out of four ~150-byte `AtomDynamics` per lane — roughly 5 KB
/// touched per batch across two mol lookups, essentially all cache misses on
/// a 6k-water box. One linear pass copies each quantity exactly as the batch
/// loop computed it (`w.m.posit - w.o.posit`, `w.m.partial_charge`, …), so
/// every lane input is bit-identical to the AoS gather it replaces; a batch
/// then reads one contiguous 60-byte row per molecule instead.
///
/// Row layout (`FIELDS` per molecule): `0..3` O position xyz; `3..6` M−O
/// offset; `6..9` H0−O; `9..12` H1−O; `12..15` charges of M, H0, H1 (O is
/// neutral in OPC).
pub(super) struct WaterSites {
    n: usize,
    data: Vec<f32>,
}

impl WaterSites {
    pub(super) const FIELDS: usize = 15;
    /// Field bases within a molecule row.
    // Offsets into the row for the charged-site series only; O's xyz lives at
    // `0..3` and the H−O offsets at `6..9`/`9..12`, read via fixed ranges.
    pub(super) const M: usize = 3;
    pub(super) const Q_M: usize = 12;
    #[inline]
    pub(super) fn row(&self, k: u32) -> &[f32] {
        let start = k as usize * Self::FIELDS;
        &self.data[start..start + Self::FIELDS]
    }
}

fn build_water_sites(water: &[WaterMolOpc]) -> WaterSites {
    let n = water.len();
    let mut data = WATER_SITES_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    data.clear();
    data.resize(n * WaterSites::FIELDS, 0.0);
    data.par_chunks_mut(WaterSites::FIELDS)
        .zip(water.par_iter())
        .for_each(|(row, w)| {
            row[0] = w.o.posit.x;
            row[1] = w.o.posit.y;
            row[2] = w.o.posit.z;
            row[3] = w.m.posit.x - w.o.posit.x;
            row[4] = w.m.posit.y - w.o.posit.y;
            row[5] = w.m.posit.z - w.o.posit.z;
            row[6] = w.h0.posit.x - w.o.posit.x;
            row[7] = w.h0.posit.y - w.o.posit.y;
            row[8] = w.h0.posit.z - w.o.posit.z;
            row[9] = w.h1.posit.x - w.o.posit.x;
            row[10] = w.h1.posit.y - w.o.posit.y;
            row[11] = w.h1.posit.z - w.o.posit.z;
            row[12] = w.m.partial_charge;
            row[13] = w.h0.partial_charge;
            row[14] = w.h1.partial_charge;
        });
    WaterSites { n, data }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
thread_local! {
    /// Recycled per-worker dense accumulators for the water-batch fold (same
    /// keep-the-larger-capacity policy as `types.rs` `CACHED_*`). Allocating
    /// an `n_std` + `n_wat` vector per rayon chain every step dominated the
    /// water-SIMD evaluation cost.
    static WATER_BATCH_SCRATCH: RefCell<(Vec<Vec3F64>, Vec<ForcesOnWaterMol>)> =
        const { RefCell::new((Vec::new(), Vec::new())) };
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct WaterLaneResult {
    pub solute: Vec3,
    pub a_o: Vec3,
    pub a_m: Vec3,
    pub a_h0: Vec3,
    pub a_h1: Vec3,
    pub b_o: Vec3,
    pub b_m: Vec3,
    pub b_h0: Vec3,
    pub b_h1: Vec3,
    pub energy: f32,
    pub virial: f32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct WaterSoluteBatch8 {
    pub std: [u32; 8],
    pub water: [u32; 8],
    pub sigma: [f32; 8],
    pub epsilon: [f32; 8],
    /// Per-lane solute 12-6-4 induction constant (Li-Merz pair C4, kcal/mol·Å⁴).
    pub c4: [f32; 8],
}

/// Compact water-water SIMD input. Complete batches are evaluated by the CPU
/// dispatch on every architecture with `WATER_SIMD_ACTIVE`; only the batch
/// tail remains on the scalar reference kernel.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WaterWaterBatch8 {
    pub a: [u32; 8],
    pub b: [u32; 8],
}

#[inline]
fn erfc8(x: f32x8) -> f32x8 {
    let t = (f32x8::splat(1.0) + f32x8::splat(0.3275911) * x).recip();
    let p = ((((f32x8::splat(1.061405429) * t - f32x8::splat(1.453152027)) * t
        + f32x8::splat(1.421413741))
        * t
        - f32x8::splat(0.284496736))
        * t
        + f32x8::splat(0.254829592))
        * t;
    p * (-(x * x)).exp()
}

#[inline]
fn sanitize_delta(delta: &mut [Vec3; 8]) {
    for d in delta.iter_mut() {
        if !d.x.is_finite()
            || !d.y.is_finite()
            || !d.z.is_finite()
            || d.magnitude_squared() < 1.0e-12
        {
            *d = Vec3::new(1.0, 0.0, 0.0);
        }
    }
}

#[inline]
fn coulomb8(
    mut delta: [Vec3; 8],
    qprod: [f32; 8],
    alpha: f32,
    cutoff: f32,
) -> ([Vec3; 8], [f32; 8]) {
    let mut valid = [true; 8];
    for (i, d) in delta.iter().enumerate() {
        let r2 = d.magnitude_squared();
        valid[i] = r2.is_finite() && r2 >= 1.0e-12 && r2 < cutoff * cutoff;
    }
    sanitize_delta(&mut delta);
    let dx = f32x8::new(delta.map(|v| v.x));
    let dy = f32x8::new(delta.map(|v| v.y));
    let dz = f32x8::new(delta.map(|v| v.z));
    let r2 = dx * dx + dy * dy + dz * dz;
    let inv = r2.sqrt().recip();
    let r = inv.recip();
    let ar = r * f32x8::splat(alpha);
    let e = (-(ar * ar)).exp();
    let erfc = erfc8(ar);
    let q = f32x8::new(qprod);
    let mag = q * (erfc * inv * inv + f32x8::splat(2.0 * alpha * 0.5641895835477563) * e * inv);
    let fx = (dx * inv * mag).to_array();
    let fy = (dy * inv * mag).to_array();
    let fz = (dz * inv * mag).to_array();
    let energy = (q * inv * erfc).to_array();
    let mut force = [Vec3::new_zero(); 8];
    let mut out_energy = [0.0; 8];
    for i in 0..8 {
        if valid[i] {
            force[i] = Vec3::new(fx[i], fy[i], fz[i]);
            out_energy[i] = energy[i];
        }
    }
    (force, out_energy)
}

#[inline]
fn lj8(
    mut delta: [Vec3; 8],
    sigma: [f32; 8],
    epsilon: [f32; 8],
    c4: [f32; 8],
    cutoff: f32,
) -> ([Vec3; 8], [f32; 8]) {
    let mut valid = [true; 8];
    for (i, d) in delta.iter().enumerate() {
        let r2 = d.magnitude_squared();
        valid[i] = r2.is_finite() && r2 >= 1.0e-12 && r2 < cutoff * cutoff;
    }
    sanitize_delta(&mut delta);
    let dx = f32x8::new(delta.map(|v| v.x));
    let dy = f32x8::new(delta.map(|v| v.y));
    let dz = f32x8::new(delta.map(|v| v.z));
    let r2 = dx * dx + dy * dy + dz * dz;
    let inv = r2.sqrt().recip();
    let sr = f32x8::new(sigma) * inv;
    let sr2 = sr * sr;
    let sr6 = sr2 * sr2 * sr2;
    let sr12 = sr6 * sr6;
    let ep = f32x8::new(epsilon);
    // Fold 12-6-4 in before the cap (see scalar `force_e_lj_c4`); c4=0 lanes
    // are bitwise no-ops on finite values.
    let c4v = f32x8::new(c4);
    let inv2 = inv * inv;
    let inv4 = inv2 * inv2;
    let mag = (f32x8::splat(24.0) * ep * (f32x8::splat(2.0) * sr12 - sr6) * inv
        - f32x8::splat(4.0) * c4v * inv4 * inv)
        .to_array();
    let en = (f32x8::splat(4.0) * ep * (sr12 - sr6) - c4v * inv4).to_array();
    let inva = inv.to_array();
    let mut force = [Vec3::new_zero(); 8];
    let mut energy = [0.0; 8];
    for i in 0..8 {
        if valid[i] {
            force[i] = delta[i] * (inva[i] * mag[i]);
            // Match the scalar `force_e_lj` 1e4 per-pair force cap so a hard
            // clash yields the identical, path-independent magnitude that the
            // crash guards rely on.
            let fm = force[i].magnitude();
            if fm > 1.0e4 {
                force[i] *= 1.0e4 / fm;
            }
            energy[i] = en[i].clamp(-1.0e6, 1.0e6);
        }
    }
    (force, energy)
}

impl WaterSoluteBatch8 {
    pub(super) fn eval(
        self,
        atoms: &[AtomDynamics],
        sites: &WaterSites,
        cell: &SimBox,
        alpha: f32,
        coulomb_cutoff: f32,
        lj_cutoff: f32,
    ) -> [WaterLaneResult; 8] {
        let mut d = [Vec3::new_zero(); 8];
        let mut offsets = [[Vec3::new_zero(); 8]; 3];
        let mut q = [[0.0; 8]; 3];
        let mut q_std = [0.0; 8];
        for i in 0..8 {
            let a = &atoms[self.std[i] as usize];
            let w = sites.row(self.water[i]);
            d[i] = cell.min_image(Vec3::new(
                w[0] - a.posit.x,
                w[1] - a.posit.y,
                w[2] - a.posit.z,
            ));
            for site in 0..3 {
                let b = WaterSites::M + site * 3;
                offsets[site][i] = Vec3::new(w[b], w[b + 1], w[b + 2]);
                q[site][i] = w[WaterSites::Q_M + site];
            }
            q_std[i] = a.partial_charge;
        }
        let (lj_force, lj_energy) = lj8(d, self.sigma, self.epsilon, self.c4, lj_cutoff);
        let mut out = [WaterLaneResult::default(); 8];
        for i in 0..8 {
            out[i].b_o += lj_force[i];
            out[i].solute -= lj_force[i];
            out[i].energy += lj_energy[i];
            out[i].virial += d[i].dot(lj_force[i]);
        }
        for site in 0..3 {
            let delta = std::array::from_fn(|i| d[i] + offsets[site][i]);
            let (force, energy) = coulomb8(
                delta,
                std::array::from_fn(|i| q[site][i] * q_std[i]),
                alpha,
                coulomb_cutoff,
            );
            for i in 0..8 {
                out[i].solute -= force[i];
                out[i].energy += energy[i];
                out[i].virial += delta[i].dot(force[i]);
                match site {
                    0 => out[i].b_m += force[i],
                    1 => out[i].b_h0 += force[i],
                    _ => out[i].b_h1 += force[i],
                }
            }
        }
        out
    }
}

/// Evaluate every complete water SIMD batch and return the solute force
/// contributions, a dense per-molecule water force vector, plus virial and the
/// full direct-space energy — solute-water AND water-water (v1.3.8: the water
/// -water lanes always computed and folded their virial; the energy fold was
/// the remaining half of that bookkeeping gap). Shared by the x86_64 and
/// aarch64 CPU dispatch paths; the kernels themselves are portable `wide`.
///
/// Batches are independent, so evaluation runs through Rayon with
/// worker-local dense accumulators merged element-wise. The summation-order
/// profile matches the scalar `calc_force_cpu` fold exactly (same f64
/// ulp-level sensitivity to the scheduler's split tree); unlike a per-lane
/// sparse-map merge, no result ever depends on HashMap iteration order.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn eval_water_batches(
    water_simd_pairs: &[WaterSoluteBatch8],
    water_water_simd_pairs: &[WaterWaterBatch8],
    atoms_std: &[AtomDynamics],
    water: &[WaterMolOpc],
    cell: &SimBox,
    spme_alpha: f32,
    coulomb_cutoff: f32,
    lj_cutoff: f32,
) -> (Vec<Vec3F64>, Vec<ForcesOnWaterMol>, f64, f64) {
    let n_std = atoms_std.len();
    let n_wat = water.len();
    let n_ww = water_water_simd_pairs.len();
    let total_jobs = n_ww + water_simd_pairs.len();
    if total_jobs == 0 {
        // Empty streams (unsolvated systems, or an architecture where
        // `WATER_SIMD_ACTIVE` is off) must not pay the dense zero-init below;
        // callers `zip` these vectors, so empties are a no-op merge.
        return (Vec::new(), Vec::new(), 0.0, 0.0);
    }

    // Measurement-only phase timing (`SPICE_WATER_SIMD_TIME=1`).
    let timed = std::env::var_os("SPICE_WATER_SIMD_TIME").is_some_and(|v| v == "1");
    let t_sites0 = crate::engine::md_core::clock::Mono::now();
    // One linear AoS→rows pass replaces the per-batch gather (see
    // `WaterSites`). Bit-identical to what the batches used to read.
    let mut sites = build_water_sites(water);
    let t_sites = t_sites0.elapsed();
    let t_fold0 = crate::engine::md_core::clock::Mono::now();

    let zeroed = || {
        let (mut f_std, mut f_wat) =
            WATER_BATCH_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
        f_std.clear();
        f_std.resize(n_std, Vec3F64::new_zero());
        f_wat.clear();
        f_wat.resize(n_wat, ForcesOnWaterMol::default());
        (f_std, f_wat, 0.0f64, 0.0f64)
    };

    // Fold-state cost is dominated by the dense per-state accumulator
    // (re)initialization, so split into ~4 states per worker instead of
    // rayon's fine default: the M-series ablation showed 64-batch states
    // costing 2x the wall time of ~2k-batch states, while states so coarse
    // that threads idle reverse the gain. `SPICE_WATER_SIMD_MINLEN` overrides
    // for A/B measurement. Splitting depends on the pool size exactly like
    // the scalar fold's scheduler-dependent summation order.
    let min_len: usize = std::env::var("SPICE_WATER_SIMD_MINLEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| (total_jobs / (rayon::current_num_threads() * 4)).clamp(256, 4096));
    let result = (0..total_jobs)
        .into_par_iter()
        .with_min_len(min_len)
        .fold(
            zeroed,
            |(mut f_std, mut f_wat, mut virial, mut energy), job| {
                if job < n_ww {
                    let batch = &water_water_simd_pairs[job];
                    let lanes = batch.eval(
                        &sites,
                        cell,
                        spme_alpha,
                        coulomb_cutoff,
                        lj_cutoff,
                        O_SIGMA,
                        O_EPS,
                    );
                    for (lane, (&a, &b)) in lanes.iter().zip(batch.a.iter().zip(batch.b.iter())) {
                        let fa = &mut f_wat[a as usize];
                        fa.f_o += Vec3F64::from(lane.a_o);
                        fa.f_m += Vec3F64::from(lane.a_m);
                        fa.f_h0 += Vec3F64::from(lane.a_h0);
                        fa.f_h1 += Vec3F64::from(lane.a_h1);
                        let fb = &mut f_wat[b as usize];
                        fb.f_o += Vec3F64::from(lane.b_o);
                        fb.f_m += Vec3F64::from(lane.b_m);
                        fb.f_h0 += Vec3F64::from(lane.b_h0);
                        fb.f_h1 += Vec3F64::from(lane.b_h1);
                        virial += lane.virial as f64;
                        energy += lane.energy as f64;
                    }
                } else {
                    let batch = &water_simd_pairs[job - n_ww];
                    let lanes = batch.eval(
                        atoms_std,
                        &sites,
                        cell,
                        spme_alpha,
                        coulomb_cutoff,
                        lj_cutoff,
                    );
                    for (lane, (&s, &mol)) in
                        lanes.iter().zip(batch.std.iter().zip(batch.water.iter()))
                    {
                        f_std[s as usize] += Vec3F64::from(lane.solute);
                        let wf = &mut f_wat[mol as usize];
                        wf.f_o += Vec3F64::from(lane.b_o);
                        wf.f_m += Vec3F64::from(lane.b_m);
                        wf.f_h0 += Vec3F64::from(lane.b_h0);
                        wf.f_h1 += Vec3F64::from(lane.b_h1);
                        energy += lane.energy as f64;
                        virial += lane.virial as f64;
                    }
                }
                (f_std, f_wat, virial, energy)
            },
        )
        .reduce(
            zeroed,
            |(mut a_std, mut a_wat, a_v, a_e), (mut b_std, mut b_wat, b_v, b_e)| {
                for (x, y) in a_std.iter_mut().zip(b_std.iter()) {
                    *x += *y;
                }
                for (x, y) in a_wat.iter_mut().zip(b_wat.iter()) {
                    x.f_o += y.f_o;
                    x.f_m += y.f_m;
                    x.f_h0 += y.f_h0;
                    x.f_h1 += y.f_h1;
                }
                WATER_BATCH_SCRATCH.with(|c| {
                    let mut s = c.borrow_mut();
                    if s.0.capacity() < b_std.capacity() {
                        std::mem::swap(&mut s.0, &mut b_std);
                    }
                    if s.1.capacity() < b_wat.capacity() {
                        std::mem::swap(&mut s.1, &mut b_wat);
                    }
                });
                (a_std, a_wat, a_v + b_v, a_e + b_e)
            },
        );
    let t_fold = t_fold0.elapsed();
    WATER_SITES_SCRATCH.with(|c| *c.borrow_mut() = std::mem::take(&mut sites.data));
    if timed {
        eprintln!(
            "water_batches ws_batches={} ww_batches={} sites_build_ms={:.3}              fold_reduce_ms={:.3}",
            water_simd_pairs.len(),
            n_ww,
            t_sites.as_secs_f64() * 1e3,
            t_fold.as_secs_f64() * 1e3,
        );
    }
    result
}

/// Fold the dense per-molecule SIMD water-force vector into the scalar water
/// accumulator produced by `calc_force_cpu`.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(super) fn merge_water_force_map(dst: &mut [ForcesOnWaterMol], src: Vec<ForcesOnWaterMol>) {
    for (d, s) in dst.iter_mut().zip(src) {
        d.f_o += s.f_o;
        d.f_m += s.f_m;
        d.f_h0 += s.f_h0;
        d.f_h1 += s.f_h1;
    }
}

impl WaterWaterBatch8 {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn eval(
        self,
        sites: &WaterSites,
        cell: &SimBox,
        alpha: f32,
        coulomb_cutoff: f32,
        lj_cutoff: f32,
        sigma: f32,
        epsilon: f32,
    ) -> [WaterLaneResult; 8] {
        let mut d = [Vec3::new_zero(); 8];
        let mut aoff = [[Vec3::new_zero(); 8]; 3];
        let mut boff = [[Vec3::new_zero(); 8]; 3];
        let mut aq = [[0.0; 8]; 3];
        let mut bq = [[0.0; 8]; 3];
        for i in 0..8 {
            let a = sites.row(self.a[i]);
            let b = sites.row(self.b[i]);
            d[i] = cell.min_image(Vec3::new(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
            for site in 0..3 {
                let base = WaterSites::M + site * 3;
                aoff[site][i] = Vec3::new(a[base], a[base + 1], a[base + 2]);
                boff[site][i] = Vec3::new(b[base], b[base + 1], b[base + 2]);
                aq[site][i] = a[WaterSites::Q_M + site];
                bq[site][i] = b[WaterSites::Q_M + site];
            }
        }
        let (lj_force, lj_energy) = lj8(d, [sigma; 8], [epsilon; 8], [0.0; 8], lj_cutoff);
        let mut out = [WaterLaneResult::default(); 8];
        for i in 0..8 {
            out[i].a_o += lj_force[i];
            out[i].b_o -= lj_force[i];
            out[i].energy += lj_energy[i];
            out[i].virial += d[i].dot(lj_force[i]);
        }
        for sa in 0..3 {
            for sb in 0..3 {
                let delta = std::array::from_fn(|i| d[i] + aoff[sa][i] - boff[sb][i]);
                let qp = std::array::from_fn(|i| aq[sa][i] * bq[sb][i]);
                let (force, energy) = coulomb8(delta, qp, alpha, coulomb_cutoff);
                for i in 0..8 {
                    out[i].energy += energy[i];
                    out[i].virial += delta[i].dot(force[i]);
                    match sa {
                        0 => out[i].a_m += force[i],
                        1 => out[i].a_h0 += force[i],
                        _ => out[i].a_h1 += force[i],
                    }
                    match sb {
                        0 => out[i].b_m -= force[i],
                        1 => out[i].b_h0 -= force[i],
                        _ => out[i].b_h1 -= force[i],
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use lin_alg::f32::Quaternion;

    const ALPHA: f32 = 0.35;
    const COULOMB_CUTOFF: f32 = 10.0;
    const LJ_CUTOFF: f32 = 9.0;

    fn cell() -> SimBox {
        SimBox::new(Vec3::new_zero(), Vec3::new(40.0, 40.0, 40.0))
    }

    fn solute(posit: [f32; 3], charge: f32) -> AtomDynamics {
        AtomDynamics {
            posit: Vec3::new(posit[0], posit[1], posit[2]),
            partial_charge: charge,
            lj_sigma: 3.15,
            lj_eps: 0.12,
            ..Default::default()
        }
    }

    fn water(o_posit: [f32; 3]) -> WaterMolOpc {
        let o = Vec3::new(o_posit[0], o_posit[1], o_posit[2]);
        WaterMolOpc::new(o, Vec3::new_zero(), Quaternion::new_identity())
    }

    /// Combined absolute+relative tolerance. Tight enough that a sign flip,
    /// a missing interaction, or a double count (off by ≥50%) fails.
    fn close(ref_v: f64, simd_v: f32, what: &str, lane: usize) {
        let r = ref_v;
        let s = simd_v as f64;
        let diff = (r - s).abs();
        let scale = r.abs().max(s.abs());
        assert!(
            diff <= 1.0e-4 + scale * 5.0e-3,
            "lane {lane} {what}: scalar={r} simd={s} diff={diff}"
        );
    }

    fn close_vec3(ref_v: Vec3F64, simd_v: Vec3, what: &str, lane: usize) {
        let s: Vec3F64 = simd_v.into();
        close(ref_v.x, s.x as f32, &format!("{what}.x"), lane);
        close(ref_v.y, s.y as f32, &format!("{what}.y"), lane);
        close(ref_v.z, s.z as f32, &format!("{what}.z"), lane);
    }

    #[test]
    fn water_solute_batch_matches_scalar_kernel() {
        let cell = cell();
        let overrides = MdOverrides::default();
        let lj_tables = LjTables::default();

        let mut atoms = [
            solute([0.5, 1.0, 1.0], 5.0),
            solute([20.0, 20.0, 20.0], -3.0),
        ];
        // Lane-relevant induction constants: solute 0 is a 12-6-4 style ion
        // (Mg's Li-Merz pair C4), solute 1 stays plain 12-6 — the batch must
        // match the scalar kernel on *both* behaviours.
        atoms[0].lj_c4 = 127.0;
        let water = [
            water([2.5, 1.0, 1.0]),    // 0: LJ + Coulomb active
            water([9.2, 1.0, 1.0]),    // 1: center inside LJ, outer sites past cutoff
            water([38.0, 1.0, 1.0]),   // 2: wraps the periodic boundary toward atom 0
            water([12.0, 12.0, 12.0]), // 3: outside every cutoff for both solutes
            water([0.5, 1.0, 1.0]),    // 4: O exactly on the solute (zero O–O delta)
        ];

        let std_idx = [0u32, 0, 1, 1, 0, 1, 0, 0];
        let wat_idx = [0u32, 1, 3, 3, 2, 2, 4, 0];
        let mut sigma = [0.0f32; 8];
        let mut epsilon = [0.0f32; 8];
        for (i, s) in sigma.iter_mut().enumerate() {
            let a = &atoms[std_idx[i] as usize];
            *s = 0.5 * (a.lj_sigma + O_SIGMA);
            epsilon[i] = (a.lj_eps * O_EPS).sqrt();
        }
        let batch = WaterSoluteBatch8 {
            std: std_idx,
            water: wat_idx,
            sigma,
            epsilon,
            c4: std::array::from_fn(|i| atoms[std_idx[i] as usize].lj_c4),
        };
        let sites = build_water_sites(&water);
        let lanes = batch.eval(&atoms, &sites, &cell, ALPHA, COULOMB_CUTOFF, LJ_CUTOFF);

        for i in 0..8 {
            let mut f_std_ref = Vec3F64::new_zero();
            let mut f_wat_ref = ForcesOnWaterMol::default();
            let mut virial_ref = 0.0f64;
            let energy_ref = f_water_std_cpu(
                &mut virial_ref,
                &mut f_std_ref,
                &mut f_wat_ref,
                &atoms[std_idx[i] as usize],
                &water[wat_idx[i] as usize],
                &cell,
                &lj_tables,
                &overrides,
                ALPHA,
                COULOMB_CUTOFF,
                LJ_CUTOFF,
                std_idx[i] as usize,
            );
            let lane = lanes[i];
            close_vec3(f_std_ref, lane.solute, "f_std", i);
            close_vec3(f_wat_ref.f_o, lane.b_o, "f_o", i);
            close_vec3(f_wat_ref.f_m, lane.b_m, "f_m", i);
            close_vec3(f_wat_ref.f_h0, lane.b_h0, "f_h0", i);
            close_vec3(f_wat_ref.f_h1, lane.b_h1, "f_h1", i);
            close(energy_ref, lane.energy, "energy", i);
            close(virial_ref, lane.virial, "virial", i);
        }

        // Out-of-range lanes must be *exactly* zero — the mask, not a
        // tolerance accident.
        for i in [2, 3, 5] {
            assert_eq!(lanes[i].solute, Vec3::new_zero());
            assert_eq!(lanes[i].b_o, Vec3::new_zero());
            assert_eq!(lanes[i].b_m, Vec3::new_zero());
            assert_eq!(lanes[i].b_h0, Vec3::new_zero());
            assert_eq!(lanes[i].b_h1, Vec3::new_zero());
            assert_eq!(lanes[i].energy, 0.0);
            assert_eq!(lanes[i].virial, 0.0);
        }
        // Duplicated lane (7 repeats 0): lanes are fully independent.
        assert_eq!(lanes[7].solute, lanes[0].solute);
        assert_eq!(lanes[7].energy, lanes[0].energy);
    }

    #[test]
    fn water_water_batch_matches_scalar_kernel() {
        let cell = cell();
        let overrides = MdOverrides::default();
        let lj_tables = LjTables::default();

        let water = [
            water([2.0, 2.0, 2.0]),
            water([5.0, 2.5, 2.0]),
            water([11.5, 2.0, 2.0]), // O–O past LJ cutoff, H sites inside Coulomb
            water([39.0, 2.0, 2.0]), // wraps the boundary toward mol 0/1
            water([30.0, 30.0, 20.0]), // far from the pairs used below
        ];
        let a = [0u32, 0, 0, 1, 1, 0, 2, 0];
        let b = [1u32, 2, 3, 3, 4, 4, 3, 1];
        let batch = WaterWaterBatch8 { a, b };
        let sites = build_water_sites(&water);
        let lanes = batch.eval(
            &sites,
            &cell,
            ALPHA,
            COULOMB_CUTOFF,
            LJ_CUTOFF,
            O_SIGMA,
            O_EPS,
        );

        for i in 0..8 {
            let mut fwa = ForcesOnWaterMol::default();
            let mut fwb = ForcesOnWaterMol::default();
            let mut virial_ref = 0.0f64;
            let energy_ref = f_water_water_cpu(
                &mut virial_ref,
                &mut fwa,
                &mut fwb,
                &water[a[i] as usize],
                &water[b[i] as usize],
                &cell,
                &lj_tables,
                &overrides,
                ALPHA,
                COULOMB_CUTOFF,
                LJ_CUTOFF,
            );
            let lane = lanes[i];
            close_vec3(fwa.f_o, lane.a_o, "a_o", i);
            close_vec3(fwa.f_m, lane.a_m, "a_m", i);
            close_vec3(fwa.f_h0, lane.a_h0, "a_h0", i);
            close_vec3(fwa.f_h1, lane.a_h1, "a_h1", i);
            close_vec3(fwb.f_o, lane.b_o, "b_o", i);
            close_vec3(fwb.f_m, lane.b_m, "b_m", i);
            close_vec3(fwb.f_h0, lane.b_h0, "b_h0", i);
            close_vec3(fwb.f_h1, lane.b_h1, "b_h1", i);
            close(energy_ref, lane.energy, "energy", i);
            close(virial_ref, lane.virial, "virial", i);
        }

        // PBC lane (0,3) must actually interact through the wrapped image.
        assert!(lanes[2].energy != 0.0);
        // Far lane (0,4) is exactly zero.
        assert_eq!(lanes[5].a_o, Vec3::new_zero());
        assert_eq!(lanes[5].b_m, Vec3::new_zero());
        assert_eq!(lanes[5].energy, 0.0);
        // Lane 7 repeats lane 0 bit-for-bit.
        assert_eq!(lanes[7].a_m, lanes[0].a_m);
        assert_eq!(lanes[7].energy, lanes[0].energy);
    }

    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self, lo: f32, hi: f32) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((self.0 >> 40) as f32) / ((1u64 << 24) as f32);
            lo + u * (hi - lo)
        }
    }

    fn adiff(a: f64, b: f32) -> f64 {
        (a - b as f64).abs()
    }

    /// Per-pair worst ABSOLUTE deviation across many randomized 8-lane
    /// batches. This is the fine-grained companion to the whole-system
    /// reference-vs-optimized check: it pins each SIMD lane against the exact
    /// scalar kernel over random geometry (including sub-cutoff clashes that
    /// exercise the LJ force/energy caps, where both paths clamp identically).
    /// f32 vector-transcendental noise on a single pair is <1e-2 in absolute
    /// units; a sign flip or dropped interaction moves a capped/coulombic term
    /// by O(100–1e6), far past this floor. Relative metrics are useless here
    /// because weakly interacting near-cutoff lanes cancel to tiny totals.
    const MAX_PAIR_ABS: f64 = 2.0;

    #[test]
    fn water_solute_random_pair_scan() {
        let cell = cell();
        let overrides = MdOverrides::default();
        let lj_tables = LjTables::default();
        let mut rng = Lcg(0x9E3779B97F4A7C15);
        let mut worst = (0.0f64, 0.0f64, 0.0f64); // force, energy, virial rel
        let mut worst_detail = String::new();
        for _ in 0..400 {
            let mut atoms = Vec::new();
            let mut water_mols = Vec::new();
            let mut std_idx = [0u32; 8];
            let mut wat_idx = [0u32; 8];
            let mut sigma = [0.0f32; 8];
            let mut epsilon = [0.0f32; 8];
            for lane in 0..8 {
                let ax = rng.f(2.0, 38.0);
                let ay = rng.f(2.0, 38.0);
                let az = rng.f(2.0, 38.0);
                atoms.push(AtomDynamics {
                    posit: Vec3::new(ax, ay, az),
                    partial_charge: rng.f(-18.0, 18.0),
                    lj_sigma: rng.f(2.4, 4.0),
                    lj_eps: rng.f(0.02, 0.3),
                    ..Default::default()
                });
                let dist = rng.f(0.3, 12.0);
                let dir =
                    Vec3::new(rng.f(-1.0, 1.0), rng.f(-1.0, 1.0), rng.f(-1.0, 1.0)).to_normalized();
                let o_pos = Vec3::new(ax, ay, az) + dir * dist;
                water_mols.push(water([o_pos.x, o_pos.y, o_pos.z]));
                let a = &mut atoms[lane];
                if lane % 3 == 0 {
                    a.lj_c4 = 127.0; // Li-Merz Mg2+ induction on 1/3 of lanes
                }
                std_idx[lane] = lane as u32;
                wat_idx[lane] = lane as u32;
                sigma[lane] = 0.5 * (a.lj_sigma + O_SIGMA);
                epsilon[lane] = (a.lj_eps * O_EPS).sqrt();
            }
            let batch = WaterSoluteBatch8 {
                std: std_idx,
                water: wat_idx,
                sigma,
                epsilon,
                c4: std::array::from_fn(|i| atoms[std_idx[i] as usize].lj_c4),
            };
            let sites = build_water_sites(&water_mols);
            let lanes = batch.eval(&atoms, &sites, &cell, ALPHA, COULOMB_CUTOFF, LJ_CUTOFF);
            for i in 0..8 {
                let mut f_ref = Vec3F64::new_zero();
                let mut fw_ref = ForcesOnWaterMol::default();
                let mut v_ref = 0.0f64;
                let e_ref = f_water_std_cpu(
                    &mut v_ref,
                    &mut f_ref,
                    &mut fw_ref,
                    &atoms[i],
                    &water_mols[i],
                    &cell,
                    &lj_tables,
                    &overrides,
                    ALPHA,
                    COULOMB_CUTOFF,
                    LJ_CUTOFF,
                    i,
                );
                let lane = lanes[i];
                let comps = [
                    ("f_std.x", f_ref.x, lane.solute.x),
                    ("f_std.y", f_ref.y, lane.solute.y),
                    ("f_std.z", f_ref.z, lane.solute.z),
                    ("b_o.x", fw_ref.f_o.x, lane.b_o.x),
                    ("b_o.y", fw_ref.f_o.y, lane.b_o.y),
                    ("b_o.z", fw_ref.f_o.z, lane.b_o.z),
                    ("b_m.x", fw_ref.f_m.x, lane.b_m.x),
                    ("b_m.y", fw_ref.f_m.y, lane.b_m.y),
                    ("b_m.z", fw_ref.f_m.z, lane.b_m.z),
                    ("b_h0.x", fw_ref.f_h0.x, lane.b_h0.x),
                    ("b_h0.y", fw_ref.f_h0.y, lane.b_h0.y),
                    ("b_h0.z", fw_ref.f_h0.z, lane.b_h0.z),
                    ("b_h1.x", fw_ref.f_h1.x, lane.b_h1.x),
                    ("b_h1.y", fw_ref.f_h1.y, lane.b_h1.y),
                    ("b_h1.z", fw_ref.f_h1.z, lane.b_h1.z),
                ];
                for (name, r, s) in comps {
                    let v = adiff(r, s);
                    if v > worst.0 {
                        worst.0 = v;
                        worst_detail = format!("{name} ref={r} simd={s}");
                    }
                }
                worst.1 = worst.1.max(adiff(e_ref, lane.energy));
                worst.2 = worst.2.max(adiff(v_ref, lane.virial));
            }
        }
        assert!(
            worst.0 < MAX_PAIR_ABS && worst.1 < MAX_PAIR_ABS && worst.2 < MAX_PAIR_ABS,
            "water-solute per-pair abs-diff force={:.3e} [{}] energy={:.3e} virial={:.3e}",
            worst.0,
            worst_detail,
            worst.1,
            worst.2
        );
    }

    #[test]
    fn water_water_random_pair_scan() {
        let cell = cell();
        let overrides = MdOverrides::default();
        let lj_tables = LjTables::default();
        let mut rng = Lcg(0xDEADBEEF12345678);
        let mut worst = (0.0f64, 0.0f64, 0.0f64);
        let mut worst_detail = String::new();
        for _ in 0..400 {
            let mut water_mols = Vec::new();
            let mut a = [0u32; 8];
            let mut b = [0u32; 8];
            for lane in 0..8 {
                let ox = rng.f(2.0, 38.0);
                let oy = rng.f(2.0, 38.0);
                let oz = rng.f(2.0, 38.0);
                water_mols.push(water([ox, oy, oz]));
                let dist = rng.f(0.5, 12.0);
                let dir =
                    Vec3::new(rng.f(-1.0, 1.0), rng.f(-1.0, 1.0), rng.f(-1.0, 1.0)).to_normalized();
                let o2 = Vec3::new(ox, oy, oz) + dir * dist;
                water_mols.push(water([o2.x, o2.y, o2.z]));
                a[lane] = (2 * lane) as u32;
                b[lane] = (2 * lane + 1) as u32;
            }
            let batch = WaterWaterBatch8 { a, b };
            let sites = build_water_sites(&water_mols);
            let lanes = batch.eval(
                &sites,
                &cell,
                ALPHA,
                COULOMB_CUTOFF,
                LJ_CUTOFF,
                O_SIGMA,
                O_EPS,
            );
            for i in 0..8 {
                let mut fwa = ForcesOnWaterMol::default();
                let mut fwb = ForcesOnWaterMol::default();
                let mut v_ref = 0.0f64;
                let e_ref = f_water_water_cpu(
                    &mut v_ref,
                    &mut fwa,
                    &mut fwb,
                    &water_mols[a[i] as usize],
                    &water_mols[b[i] as usize],
                    &cell,
                    &lj_tables,
                    &overrides,
                    ALPHA,
                    COULOMB_CUTOFF,
                    LJ_CUTOFF,
                );
                let lane = lanes[i];
                let comps = [
                    ("a_o.x", fwa.f_o.x, lane.a_o.x),
                    ("a_o.y", fwa.f_o.y, lane.a_o.y),
                    ("a_o.z", fwa.f_o.z, lane.a_o.z),
                    ("a_m.x", fwa.f_m.x, lane.a_m.x),
                    ("a_m.y", fwa.f_m.y, lane.a_m.y),
                    ("a_m.z", fwa.f_m.z, lane.a_m.z),
                    ("a_h0.x", fwa.f_h0.x, lane.a_h0.x),
                    ("a_h0.y", fwa.f_h0.y, lane.a_h0.y),
                    ("a_h0.z", fwa.f_h0.z, lane.a_h0.z),
                    ("a_h1.x", fwa.f_h1.x, lane.a_h1.x),
                    ("a_h1.y", fwa.f_h1.y, lane.a_h1.y),
                    ("a_h1.z", fwa.f_h1.z, lane.a_h1.z),
                    ("b_o.x", fwb.f_o.x, lane.b_o.x),
                    ("b_m.y", fwb.f_m.y, lane.b_m.y),
                    ("b_h0.z", fwb.f_h0.z, lane.b_h0.z),
                    ("b_h1.x", fwb.f_h1.x, lane.b_h1.x),
                ];
                for (name, r, s) in comps {
                    let v = adiff(r, s);
                    if v > worst.0 {
                        worst.0 = v;
                        worst_detail = format!("{name} ref={r} simd={s}");
                    }
                }
                worst.1 = worst.1.max(adiff(e_ref, lane.energy));
                worst.2 = worst.2.max(adiff(v_ref, lane.virial));
            }
        }
        assert!(
            worst.0 < MAX_PAIR_ABS && worst.1 < MAX_PAIR_ABS && worst.2 < MAX_PAIR_ABS,
            "water-water per-pair abs-diff force={:.3e} [{}] energy={:.3e} virial={:.3e}",
            worst.0,
            worst_detail,
            worst.1,
            worst.2
        );
    }
}
