use lin_alg::f32::Vec3;
#[allow(unused)]
#[cfg(target_arch = "x86_64")]
use lin_alg::f32::{Vec3x8, Vec3x16, f32x8, f32x16};

/// CPU LJ force. See notes on `V_lj()`. We set up the inv_dist param to share computation
/// with short-range Coulomb.
/// This assumes diff (and dir) is in order tgt - src.
/// This variant also computes energy.
/// 12-6-4 variant: folds the induction term `U4 = −c4/r⁴` into the *same*
/// magnitude/energy as the LJ before the 1e4 force cap, so a hard clash
/// yields one path-independent capped force (the cap must cover C4 too —
/// 4·c4·r⁻⁵ explodes at r→0). `c4 = 0` reduces bit-exactly to
/// [`force_e_lj`] (subtracting `0.0` from finite terms).
///
/// Pair constants come from Li & Merz et al., J. Chem. Theory Comput. 2020
/// (doi:10.1021/acs.jctc.0c00194), Table 5 *OPC column* — the published
/// ion–water C4 pair values (kcal/mol·Å⁴), applied to ion–water-oxygen
/// pairs only (their Eq. 5 zeroes ion–H C4 and gives no non-water partners
/// a value).
pub fn force_e_lj_c4(dir: Vec3, inv_dist: f32, sigma: f32, eps: f32, c4: f32) -> (Vec3, f32) {
    let sr = sigma * inv_dist;
    let sr2 = sr * sr;
    let sr6 = sr2 * sr2 * sr2;
    let sr12 = sr6 * sr6;

    let inv2 = inv_dist * inv_dist;
    let inv4 = inv2 * inv2;

    let mag = 24. * eps * 2.0f32.mul_add(sr12, -sr6) * inv_dist - 4. * c4 * inv4 * inv_dist;

    let mut f = dir * mag;
    const FORCE_CAP: f32 = 1e4;
    let mag_f = f.magnitude();
    if mag_f > FORCE_CAP {
        f = f * (FORCE_CAP / mag_f);
    }
    let energy = (4. * eps * (sr12 - sr6) - c4 * inv4).clamp(-1e6, 1e6);
    (f, energy)
}

pub fn force_e_lj(dir: Vec3, inv_dist: f32, sigma: f32, eps: f32) -> (Vec3, f32) {
    let sr = sigma * inv_dist;
    // Expand fixed powers explicitly; this is in the innermost pair kernel.
    let sr2 = sr * sr;
    let sr6 = sr2 * sr2 * sr2;
    let sr12 = sr6 * sr6;

    let mag = 24. * eps * 2.0f32.mul_add(sr12, -sr6) * inv_dist;

    let mut f = dir * mag;
    const FORCE_CAP: f32 = 1e4;
    let mag_f = f.magnitude();
    if mag_f > FORCE_CAP {
        f = f * (FORCE_CAP / mag_f);
    }
    // energy 同步限幅，防止累加爆炸
    let energy = (4. * eps * (sr12 - sr6)).clamp(-1e6, 1e6);
    (f, energy)
}

/// SIMD variant
#[allow(unused)]
#[cfg(target_arch = "x86_64")]
pub fn force_e_lj_x8(dir: Vec3x8, inv_dist: f32x8, sigma: f32x8, eps: f32x8) -> (Vec3x8, f32x8) {
    let sr = sigma * inv_dist;
    let sr2 = sr * sr;
    let sr6 = sr2 * sr2 * sr2;
    let sr12 = sr6 * sr6;

    let mag = f32x8::splat(24.) * eps * (f32x8::splat(2.) * sr12 - sr6) * inv_dist;

    let energy = f32x8::splat(4.) * eps * (sr12 - sr6);
    (dir * mag, energy)
}

/// SIMD variant. Note: Having this code compiled, then run on an AVX-512 system is fine;
/// just don't run it.
#[allow(unused)]
#[cfg(target_arch = "x86_64")]
pub fn force_e_lj_x16(
    dir: Vec3x16,
    inv_dist: f32x16,
    sigma: f32x16,
    eps: f32x16,
) -> (Vec3x16, f32x16) {
    let sr = sigma * inv_dist;
    let sr2 = sr * sr;
    let sr6 = sr2 * sr2 * sr2;
    let sr12 = sr6 * sr6;

    let mag = f32x16::splat(24.) * eps * (f32x16::splat(2.) * sr12 - sr6) * inv_dist;

    let energy = f32x16::splat(4.) * eps * (sr12 - sr6);
    (dir * mag, energy)
}
