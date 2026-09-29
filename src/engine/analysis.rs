//! Diagnostic probes over a built system: electrostatic potential at arbitrary
//! points, per-atom solvent-accessible surface area (SASA), and subset geometry
//! queries. Pure functions over borrowed data, deliberately far from the hot
//! path; the `SpiceEngine` methods in `crate::engine` just wire state in.
//!
//! The electrostatic evaluator MIRRORS the engine's own PME parameterization
//! (same alpha, same mesh, same cubic-B-spline assignment, same influence
//! function as the patched `ewald` fork's `PmeRecip::forces`) so probe values
//! are consistent with the forces the atoms actually feel. The unit test
//! `esp_recip_matches_pme_energy` pins this: feeding the probe potential back
//! through 1/2 * sum q_i phi(r_i) must reproduce the fork's reciprocal energy.
//!
//! Units: potentials are reported in kcal/mol per elementary charge
//! (internal charges divided out via CHARGE_UNIT_SCALER). The absolute value of
//! a PME potential carries a position-independent constant (self +
//! neutralizing-background terms); every useful quantity here is a DIFFERENCE
//! between probe points, where the constant cancels. We omit the constant and
//! document it rather than invent a reference.

use lin_alg::f32::Vec3;
use rustfft::num_complex::Complex;

/// Same value as `CHARGE_UNIT_SCALER` in `forcefield/non_bonded/types.rs:82`
/// (private to that module); the `esp_units_are_elementary` test below pins
/// the numeric contract.
const CUS: f64 = 18.2223;

const TAU: f64 = std::f64::consts::TAU;
const SPLINE_ORDER: usize = 4;

/// Mirror of the ewald fork's `make_k_array`: signed frequency bins as
/// angular wave numbers (rad per length unit).
fn make_k_array(n: usize, l: f64) -> Vec<f64> {
    let tau_div_l = TAU / l;
    let n_half = n / 2;
    (0..n)
        .map(|i| {
            let fi = if i <= n_half {
                i as isize
            } else {
                (i as isize) - (n as isize)
            };
            tau_div_l * fi as f64
        })
        .collect()
}

/// Mirror of the fork's `spline_bmod_sq_inv_1d`: 1/|b_4(k)|^2, the PME
/// deconvolution factor for cardinal cubic B-splines.
fn spline_bmod_sq_inv_1d(n: usize) -> Vec<f64> {
    let bspline_ints: [f64; 3] = [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0];
    (0..n)
        .map(|k| {
            let theta = TAU * k as f64 / n as f64;
            let mut re = 0.0;
            let mut im = 0.0;
            for (j, &val) in bspline_ints.iter().enumerate() {
                let phase = theta * j as f64;
                re += val * phase.cos();
                im += val * phase.sin();
            }
            let b2 = (re * re + im * im).max(1e-12);
            1.0 / b2
        })
        .collect()
}

/// Mirror of the fork's `bspline4_weights`: cubic B-spline 4-point stencil.
/// `s` is position in grid units; returns the leftmost support index and the
/// four weights (same convention as the fork: i0 = floor(s) - 1).
fn bspline4_weights(s: f64) -> (isize, [f64; SPLINE_ORDER]) {
    let sfloor = s.floor();
    let u = s - sfloor;
    let i0 = sfloor as isize - 1;
    let u2 = u * u;
    let u3 = u2 * u;
    let w0 = (1.0 - u).powi(3) / 6.0;
    let w1 = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0;
    let w2 = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0;
    let w3 = u3 / 6.0;
    (i0, [w0, w1, w2, w3])
}

fn wrap(i: isize, n: usize) -> usize {
    ((i.rem_euclid(n as isize)) as usize).min(n - 1)
}

/// Electrostatic potential at probe points, in kcal/mol/e, using the same PME
/// split as the engine: real-space `erfc(alpha r)/r` within `rcut` (minimum
/// image), plus the reciprocal mesh sum over the full half-spectrum with the
/// exact B-spline deconvolution. Positions/charges/box are the engine's own
/// (internal charge units, Angstrom).
pub fn electrostatic_potential(
    points: &[[f64; 3]],
    pos: &[Vec3],
    q: &[f32],
    extent: Vec3,
    alpha: f64,
    rcut: f64,
    mesh_spacing: f64,
) -> Vec<f64> {
    let (nx, ny, nz) = ewald::get_grid_n((extent.x, extent.y, extent.z), mesh_spacing as f32);
    let (lx, ly, lz) = (extent.x as f64, extent.y as f64, extent.z as f64);
    let vol = lx * ly * lz;
    let nzc = nz / 2 + 1;

    // 1. Spread charges onto the mesh (positions folded into [0, L)).
    let mut rho_real = vec![0.0f32; nx * ny * nz];
    for (r, &qi) in pos.iter().zip(q.iter()) {
        if qi == 0.0 {
            continue;
        }
        let sx = (r.x as f64).rem_euclid(lx) / lx * nx as f64;
        let sy = (r.y as f64).rem_euclid(ly) / ly * ny as f64;
        let sz = (r.z as f64).rem_euclid(lz) / lz * nz as f64;
        let (ix0, wx) = bspline4_weights(sx);
        let (iy0, wy) = bspline4_weights(sy);
        let (iz0, wz) = bspline4_weights(sz);
        for a in 0..SPLINE_ORDER {
            let ix = wrap(ix0 + a as isize, nx);
            for b in 0..SPLINE_ORDER {
                let iy = wrap(iy0 + b as isize, ny);
                let wxy = wx[a] * wy[b];
                let base = ix * ny * nz + iy * nz;
                for c in 0..SPLINE_ORDER {
                    let iz = wrap(iz0 + c as isize, nz);
                    rho_real[base + iz] += qi * wxy as f32 * wz[c] as f32;
                }
            }
        }
    }

    // 2. Forward r2c FFT, z-fast layout (mirror of the fork's fft3d_r2c).
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let rho = fft3d_r2c(&mut rho_real, (nx, ny, nz), &mut planner);

    // 3. phi_k = G(k) * rho_k with the fork's exact influence function.
    let kx = make_k_array(nx, lx);
    let ky = make_k_array(ny, ly);
    let kz = make_k_array(nz, lz);
    let bx = spline_bmod_sq_inv_1d(nx);
    let by = spline_bmod_sq_inv_1d(ny);
    let bz = spline_bmod_sq_inv_1d(nz);
    let mut phi_k: Vec<Complex<f32>> = Vec::with_capacity(rho.len());
    for (idx, &rho_val) in rho.iter().enumerate() {
        let izc = idx % nzc;
        let iy = (idx / nzc) % ny;
        let ix = idx / (nzc * ny);
        let k2 = kx[ix] * kx[ix] + ky[iy] * ky[iy] + kz[izc] * kz[izc];
        if k2 == 0.0 {
            phi_k.push(Complex::new(0.0, 0.0));
            continue;
        }
        // Force-consistent PME potential at arbitrary points: the fork's
        // phi_k = (2pi/V) exp(-k^2/4a^2)/k^2 * |b|^-2 * rho_k, times ONE more
        // assignment window b_hat(k) (evaluating the grid potential off-node
        // interpolates with the same B-spline, and rho_k already carries two
        // powers of b_hat, so the net influence is sqrt(|b|^-2) = 1/|b|).
        // This makes 1/2 sum q_i phi(r_i) reproduce the fork's reciprocal
        // energy exactly - pinned by esp_recip_matches_pme_energy.
        let inv_b_hat = (bx[ix] * by[iy] * bz[izc]).sqrt(); // 1/|b|^2 -> 1/|b|
        let ghat = (2.0 * TAU / vol) * (-(k2) / (4.0 * alpha * alpha)).exp() / k2 * inv_b_hat;
        phi_k.push(Complex::new(
            (rho_val.re as f64 * ghat) as f32,
            (rho_val.im as f64 * ghat) as f32,
        ));
    }

    // 4. Per-point evaluation: real space + reciprocal k-sum.
    points
        .iter()
        .map(|p| {
            let (px, py, pz) = (p[0], p[1], p[2]);
            // real space (minimum image, cutoff)
            let mut phi = 0.0f64;
            for (r, &qi) in pos.iter().zip(q.iter()) {
                if qi == 0.0 {
                    continue;
                }
                let mut dx = (r.x as f64) - px;
                let mut dy = (r.y as f64) - py;
                let mut dz = (r.z as f64) - pz;
                dx -= lx * (dx / lx).round();
                dy -= ly * (dy / ly).round();
                dz -= lz * (dz / lz).round();
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 >= rcut * rcut || r2 < 1e-12 {
                    continue;
                }
                let r = r2.sqrt();
                phi += qi as f64 * libm::erfc(alpha * r) / r;
            }
            // reciprocal space: sum over half spectrum, interior izc doubled
            let mut recip = 0.0f64;
            for (idx, &phi) in phi_k.iter().enumerate() {
                let izc = idx % nzc;
                let iy = (idx / nzc) % ny;
                let ix = idx / (nzc * ny);
                let theta = kx[ix] * px + ky[iy] * py + kz[izc] * pz;
                let (s, c) = libm::sincos(theta);
                let mut term = phi.re as f64 * c - phi.im as f64 * s;
                if izc > 0 && izc < nzc - 1 {
                    term *= 2.0;
                }
                recip += term;
            }
            phi += recip;
            // internal charge units -> kcal/mol per elementary charge
            phi * CUS
        })
        .collect()
}

/// Electrostatic FIELD **E = −∇φ** (kcal/mol·e⁻¹·Å⁻¹) at probe points, using the
/// SAME PME split as [`electrostatic_potential`] (real-space erfc gradient + the
/// exact reciprocal k-gradient), so the field is consistent with the forces the
/// atoms actually feel. This is the B2 functional-probe primitive: the site's
/// catalytic electric field, not its gauge-arbitrary absolute potential.
///
/// WHY ANALYTIC, not a finite difference of `electrostatic_potential`: φ is a
/// large, slowly-drifting number (tens of kcal/mol/e from the PME background as
/// the box relaxes), so `−(φ(p+ħ)−φ(p−ħ))/2ħ` *amplifies* that drift (difference
/// of two wandering large numbers). The closed-form gradient differentiates
/// `erfc(αr)/r` and `e^{ik·r}` directly, with NO subtractive cancellation — the
/// noise that made the FD field unusable as a gate simply does not appear.
///
/// `positions` are the charges' coordinates; the caller may feed a
/// time-averaged structure (see `SpiceEngine::pme_positions`) to average out
/// thermal motion. The reciprocal influence function `ghat` and half-spectrum
/// doubling match the potential exactly (pinned against its finite difference by
/// `esp_field_matches_fd_potential`).
pub fn electrostatic_field(
    points: &[[f64; 3]],
    pos: &[Vec3],
    q: &[f32],
    extent: Vec3,
    alpha: f64,
    rcut: f64,
    mesh_spacing: f64,
) -> Vec<[f64; 3]> {
    let (nx, ny, nz) = ewald::get_grid_n((extent.x, extent.y, extent.z), mesh_spacing as f32);
    let (lx, ly, lz) = (extent.x as f64, extent.y as f64, extent.z as f64);
    let vol = lx * ly * lz;
    let nzc = nz / 2 + 1;

    // 1. Spread charges onto the mesh (identical to electrostatic_potential).
    let mut rho_real = vec![0.0f32; nx * ny * nz];
    for (r, &qi) in pos.iter().zip(q.iter()) {
        if qi == 0.0 {
            continue;
        }
        let sx = (r.x as f64).rem_euclid(lx) / lx * nx as f64;
        let sy = (r.y as f64).rem_euclid(ly) / ly * ny as f64;
        let sz = (r.z as f64).rem_euclid(lz) / lz * nz as f64;
        let (ix0, wx) = bspline4_weights(sx);
        let (iy0, wy) = bspline4_weights(sy);
        let (iz0, wz) = bspline4_weights(sz);
        for a in 0..SPLINE_ORDER {
            let ix = wrap(ix0 + a as isize, nx);
            for b in 0..SPLINE_ORDER {
                let iy = wrap(iy0 + b as isize, ny);
                let wxy = wx[a] * wy[b];
                let base = ix * ny * nz + iy * nz;
                for c in 0..SPLINE_ORDER {
                    let iz = wrap(iz0 + c as isize, nz);
                    rho_real[base + iz] += qi * wxy as f32 * wz[c] as f32;
                }
            }
        }
    }
    // 2. Forward r2c FFT.
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let rho = fft3d_r2c(&mut rho_real, (nx, ny, nz), &mut planner);
    // 3. phi_k = G(k) * rho_k (same influence function as the potential).
    let kx = make_k_array(nx, lx);
    let ky = make_k_array(ny, ly);
    let kz = make_k_array(nz, lz);
    let bx = spline_bmod_sq_inv_1d(nx);
    let by = spline_bmod_sq_inv_1d(ny);
    let bz = spline_bmod_sq_inv_1d(nz);
    let mut phi_k: Vec<Complex<f32>> = Vec::with_capacity(rho.len());
    for (idx, &rho_val) in rho.iter().enumerate() {
        let izc = idx % nzc;
        let iy = (idx / nzc) % ny;
        let ix = idx / (nzc * ny);
        let k2 = kx[ix] * kx[ix] + ky[iy] * ky[iy] + kz[izc] * kz[izc];
        if k2 == 0.0 {
            phi_k.push(Complex::new(0.0, 0.0));
            continue;
        }
        let inv_b_hat = (bx[ix] * by[iy] * bz[izc]).sqrt();
        let ghat = (2.0 * TAU / vol) * (-(k2) / (4.0 * alpha * alpha)).exp() / k2 * inv_b_hat;
        phi_k.push(Complex::new(
            (rho_val.re as f64 * ghat) as f32,
            (rho_val.im as f64 * ghat) as f32,
        ));
    }

    // 4. Per-point E = −∇φ: real-space gradient + reciprocal k-gradient.
    let two_alpha_over_sqrt_pi = 2.0 * alpha / f64::sqrt(std::f64::consts::PI);
    points
        .iter()
        .map(|p| {
            let (px, py, pz) = (p[0], p[1], p[2]);
            let (mut ex, mut ey, mut ez) = (0.0f64, 0.0f64, 0.0f64);
            // real space: g(r)=erfc(a r)/r, g'(r) = -(2a/sqrtpi) e^{-a^2 r^2}/r - erfc(a r)/r^2
            //   E_real = CUS * sum qi g'(r) * (r_i - p)/r   (points away from + charge)
            for (r, &qi) in pos.iter().zip(q.iter()) {
                if qi == 0.0 {
                    continue;
                }
                let mut dx = (r.x as f64) - px;
                let mut dy = (r.y as f64) - py;
                let mut dz = (r.z as f64) - pz;
                dx -= lx * (dx / lx).round();
                dy -= ly * (dy / ly).round();
                dz -= lz * (dz / lz).round();
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 >= rcut * rcut || r2 < 1e-12 {
                    continue;
                }
                let rr = r2.sqrt();
                let gp = -two_alpha_over_sqrt_pi * (-(alpha * alpha) * r2).exp() / rr
                    - libm::erfc(alpha * rr) / r2; // = g'(r)
                let coef = qi as f64 * gp / rr; // × (r_i - p) below; CUS at end
                ex += coef * dx;
                ey += coef * dy;
                ez += coef * dz;
            }
            // reciprocal: recip(p) = sum mult Re[phi_k e^{i k·p}];
            //   E_recip = CUS * sum mult (phi.im*cos + phi.re*sin) * k
            for (idx, &phi) in phi_k.iter().enumerate() {
                let izc = idx % nzc;
                let iy = (idx / nzc) % ny;
                let ix = idx / (nzc * ny);
                let theta = kx[ix] * px + ky[iy] * py + kz[izc] * pz;
                let (s, c) = libm::sincos(theta);
                let mut m = phi.im as f64 * c + phi.re as f64 * s;
                if izc > 0 && izc < nzc - 1 {
                    m *= 2.0;
                }
                ex += m * kx[ix];
                ey += m * ky[iy];
                ez += m * kz[izc];
            }
            [ex * CUS, ey * CUS, ez * CUS]
        })
        .collect()
}

/// rustfft's r2c 3D forward, copied layout-for-layout from the fork's
/// `fft3d_r2c` (z contiguous rows, then y, then x columns; unnormalized).
fn fft3d_r2c(
    data_r: &mut [f32],
    dims: (usize, usize, usize),
    planner: &mut rustfft::FftPlanner<f32>,
) -> Vec<Complex<f32>> {
    let (nx, ny, nz) = dims;
    let nzc = nz / 2 + 1;
    let mut real_planner = realfft::RealFftPlanner::<f32>::new();
    let r2c_z = real_planner.plan_fft_forward(nz);
    let fft_y = planner.plan_fft_forward(ny);
    let fft_x = planner.plan_fft_forward(nx);

    let mut out = vec![Complex::new(0.0f32, 0.0); nx * ny * nzc];
    for ix in 0..nx {
        for iy in 0..ny {
            let row_r = ix * (ny * nz) + iy * nz;
            let row_k = ix * (ny * nzc) + iy * nzc;
            r2c_z
                .process(&mut data_r[row_r..row_r + nz], &mut out[row_k..row_k + nzc])
                .unwrap();
        }
    }
    {
        let mut tmp = vec![Complex::new(0.0f32, 0.0); ny];
        for ix in 0..nx {
            for izc in 0..nzc {
                for (j, iy) in (0..ny).enumerate() {
                    tmp[j] = out[ix * (ny * nzc) + iy * nzc + izc];
                }
                fft_y.process(&mut tmp);
                for (j, iy) in (0..ny).enumerate() {
                    out[ix * (ny * nzc) + iy * nzc + izc] = tmp[j];
                }
            }
        }
    }
    {
        let mut tmp = vec![Complex::new(0.0f32, 0.0); nx];
        for iy in 0..ny {
            for izc in 0..nzc {
                for (k, ix) in (0..nx).enumerate() {
                    tmp[k] = out[ix * (ny * nzc) + iy * nzc + izc];
                }
                fft_x.process(&mut tmp);
                for (k, ix) in (0..nx).enumerate() {
                    out[ix * (ny * nzc) + iy * nzc + izc] = tmp[k];
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// SASA (Shrake-Rupley)
// ---------------------------------------------------------------------------

/// Bondi-style vdW radii (Angstrom) for the elements SPICE builds; unknown
/// elements (and ions, whose LJ sigma is a well radius, not a vdW contact
/// radius) fall back to `sigma / 2^(1/6) * ...` via the caller-provided
/// `lj_sigma`. H gets the larger 1.20 Bondi value (standard for SASA).
pub fn vdw_radius(element: &na_seq::Element, lj_sigma: f64) -> f64 {
    match element {
        na_seq::Element::Hydrogen => 1.20,
        na_seq::Element::Carbon => 1.70,
        na_seq::Element::Nitrogen => 1.55,
        na_seq::Element::Oxygen => 1.52,
        na_seq::Element::Sulfur => 1.80,
        na_seq::Element::Phosphorus => 1.80,
        _ => lj_sigma / 1.122_462, // R_min/2: true-sigma -> contact radius
    }
}

/// Deterministic Fibonacci-lattice directions on the unit sphere (n points).
fn sphere_directions(n: usize) -> Vec<[f64; 3]> {
    let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    (0..n)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = golden * i as f64;
            [phi.cos() * r, phi.sin() * r, z]
        })
        .collect()
}

/// Per-atom solvent-accessible surface area (Angstrom^2) by Shrake-Rupley:
/// each atom gets a probe sphere of radius `r_i + probe`; a lattice point is
/// buried when another atom's probe sphere covers it. Neighbor lookup via a
/// uniform grid, minimum image. `radii` are vdW radii (see `vdw_radius`).
pub fn atom_sasa(
    pos: &[Vec3],
    radii: &[f64],
    extent: Vec3,
    probe: f64,
    n_sphere: usize,
) -> Vec<f64> {
    let n = pos.len();
    let dirs = sphere_directions(n_sphere);
    let cell = 2.0 * (radii.iter().cloned().fold(0.0f64, f64::max) + probe) + 1e-6;
    let (lx, ly, lz) = (extent.x as f64, extent.y as f64, extent.z as f64);
    let gx = (lx / cell).ceil() as usize;
    let gy = (ly / cell).ceil() as usize;
    let gz = (lz / cell).ceil() as usize;
    let mut grid: Vec<Vec<usize>> = vec![Vec::new(); gx * gy * gz];
    for (i, r) in pos.iter().enumerate() {
        let cx = ((r.x as f64).rem_euclid(lx) / lx * gx as f64) as usize;
        let cy = ((r.y as f64).rem_euclid(ly) / ly * gy as f64) as usize;
        let cz = ((r.z as f64).rem_euclid(lz) / lz * gz as f64) as usize;
        grid[(cx * gy + cy) * gz + cz].push(i);
    }
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        let ri = radii[i] + probe;
        let cx = ((pos[i].x as f64).rem_euclid(lx) / lx * gx as f64) as usize;
        let cy = ((pos[i].y as f64).rem_euclid(ly) / ly * gy as f64) as usize;
        let cz = ((pos[i].z as f64).rem_euclid(lz) / lz * gz as f64) as usize;
        // collect candidate neighbors within one cell ring
        let mut cand: Vec<usize> = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let nx = (cx as isize + dx).rem_euclid(gx as isize) as usize;
                    let ny = (cy as isize + dy).rem_euclid(gy as isize) as usize;
                    let nz = (cz as isize + dz).rem_euclid(gz as isize) as usize;
                    cand.extend_from_slice(&grid[(nx * gy + ny) * gz + nz]);
                }
            }
        }
        let mut exposed = 0usize;
        'sphere: for d in &dirs {
            let px = pos[i].x as f64 + ri * d[0];
            let py = pos[i].y as f64 + ri * d[1];
            let pz = pos[i].z as f64 + ri * d[2];
            for &j in &cand {
                if j == i {
                    continue;
                }
                let mut ddx = px - pos[j].x as f64;
                let mut ddy = py - pos[j].y as f64;
                let mut ddz = pz - pos[j].z as f64;
                ddx -= lx * (ddx / lx).round();
                ddy -= ly * (ddy / ly).round();
                ddz -= lz * (ddz / lz).round();
                let rr = radii[j] + probe;
                if ddx * ddx + ddy * ddy + ddz * ddz < rr * rr {
                    continue 'sphere; // buried
                }
            }
            exposed += 1;
        }
        out[i] = 4.0 * std::f64::consts::PI * ri * ri * exposed as f64 / n_sphere as f64;
    }
    out
}

// ---------------------------------------------------------------------------
// Subset geometry
// ---------------------------------------------------------------------------

/// Count pairs (a_i, b_j) whose minimum-image distance is below `cutoff`.
/// When the two sets are the same (a == b by content is not checked; pass the
/// same slice for both), self-pairs are skipped and each pair counts once.
pub fn contact_count(a: &[usize], b: &[usize], pos: &[Vec3], extent: Vec3, cutoff: f64) -> usize {
    let same = std::ptr::eq(a, b) || (a.len() == b.len() && a == b);
    let (lx, ly, lz) = (extent.x as f64, extent.y as f64, extent.z as f64);
    let c2 = cutoff * cutoff;
    let mut count = 0;
    for (ai, &i) in a.iter().enumerate() {
        let start = if same { ai + 1 } else { 0 };
        for &j in &b[start..] {
            if i == j {
                continue;
            }
            let mut dx = pos[i].x as f64 - pos[j].x as f64;
            let mut dy = pos[i].y as f64 - pos[j].y as f64;
            let mut dz = pos[i].z as f64 - pos[j].z as f64;
            dx -= lx * (dx / lx).round();
            dy -= ly * (dy / ly).round();
            dz -= lz * (dz / lz).round();
            if dx * dx + dy * dy + dz * dz < c2 {
                count += 1;
            }
        }
    }
    count
}

/// Clearance profile along a polyline: resampled every `spacing` Angstrom, the
/// distance from the sample point to the nearest atom SURFACE (minimum image,
/// vdW radius subtracted). The bottleneck is the minimum of the profile:
/// a channel passes a probe of radius `r` iff min clearance >= r.
/// `exclude` skips atoms (e.g. the substrate whose channel you are measuring).
pub fn bottleneck_profile(
    path: &[[f64; 3]],
    spacing: f64,
    pos: &[Vec3],
    radii: &[f64],
    extent: Vec3,
    exclude: &[usize],
) -> Vec<f64> {
    let (lx, ly, lz) = (extent.x as f64, extent.y as f64, extent.z as f64);
    let mut samples: Vec<[f64; 3]> = Vec::new();
    for seg in path.windows(2) {
        let (a, b) = (seg[0], seg[1]);
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let n = (len / spacing).max(1.0) as usize;
        for s in 0..n {
            let t = s as f64 / n as f64;
            samples.push([a[0] + d[0] * t, a[1] + d[1] * t, a[2] + d[2] * t]);
        }
    }
    if let Some(last) = path.last() {
        samples.push(*last);
    }
    let skip = |i: usize| exclude.contains(&i);
    samples
        .iter()
        .map(|p| {
            let mut best = f64::INFINITY;
            for (i, r) in pos.iter().enumerate() {
                if skip(i) {
                    continue;
                }
                let mut dx = p[0] - r.x as f64;
                let mut dy = p[1] - r.y as f64;
                let mut dz = p[2] - r.z as f64;
                dx -= lx * (dx / lx).round();
                dy -= ly * (dy / ly).round();
                dz -= lz * (dz / lz).round();
                let d = (dx * dx + dy * dy + dz * dz).sqrt() - radii[i];
                if d < best {
                    best = d;
                }
            }
            best
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3::new(x as f32, y as f32, z as f32)
    }

    /// The reciprocal part of the probe potential, fed back through
    /// 1/2 sum q_i phi_recip(r_i), must equal the ewald fork's reciprocal
    /// energy (forces() total minus the self term). This pins every
    /// convention: spread, FFT layout, G(k), half-spectrum weights.
    #[test]
    fn esp_recip_matches_pme_energy() {
        let extent = v(30.0, 30.0, 30.0);
        let alpha = 0.26f64;
        let rcut = 10.0f64;
        let mesh = 1.0f64;
        // deterministic pseudo-random charges/positions (no rand dep here)
        let n = 24usize;
        let mut pos = Vec::with_capacity(n);
        let mut q = Vec::with_capacity(n);
        let mut seed = 0x1234_5678u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f64 / (1u64 << 24) as f64
        };
        for i in 0..n {
            pos.push(Vec3::new(
                (next() * 30.0) as f32,
                (next() * 30.0) as f32,
                (next() * 30.0) as f32,
            ));
            q.push(if i % 2 == 0 { 18.2223 } else { -18.2223 });
        }
        let (nx, ny, nz) = ewald::get_grid_n((30.0, 30.0, 30.0), 1.0);
        let mut pme = ewald::PmeRecip::new((nx, ny, nz), (30.0, 30.0, 30.0), alpha as f32);
        let (_f, e_total) = pme.forces(&pos, &q);
        let self_e: f64 = -alpha / f64::sqrt(std::f64::consts::PI)
            * q.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
        let e_recip = e_total as f64 - self_e;

        // probe reciprocal-only energy: evaluate phi at the atom positions and
        // subtract the real-space part we also compute (do it by difference:
        // phi_total - phi_real_atoms = phi_recip).
        let pts: Vec<[f64; 3]> = pos
            .iter()
            .map(|p| [p.x as f64, p.y as f64, p.z as f64])
            .collect();
        let phi_total = electrostatic_potential(&pts, &pos, &q, extent, alpha, rcut, mesh);
        // real-space-only via huge alpha (reciprocal weight exp(-k^2/4a^2) -> 0
        // for k>0; k=0 excluded anyway) is fragile; instead compute real part
        // directly here (same loop as production, minus the mesh sum):
        let phi_real: Vec<f64> = pts
            .iter()
            .map(|p| {
                let mut acc = 0.0;
                for (r, &qi) in pos.iter().zip(q.iter()) {
                    let mut dx = r.x as f64 - p[0];
                    let mut dy = r.y as f64 - p[1];
                    let mut dz = r.z as f64 - p[2];
                    dx -= 30.0 * (dx / 30.0).round();
                    dy -= 30.0 * (dy / 30.0).round();
                    dz -= 30.0 * (dz / 30.0).round();
                    let r2 = dx * dx + dy * dy + dz * dz;
                    if r2 >= rcut * rcut || r2 < 1e-12 {
                        continue;
                    }
                    let rr = r2.sqrt();
                    acc += qi as f64 * libm::erfc(alpha * rr) / rr;
                }
                acc * CUS
            })
            .collect();
        // phi is reported per elementary charge; divide back by CUS to get
        // per-internal-charge, then 1/2 sum q_int phi_int = energy.
        let e_probe: f64 = 0.5
            * q.iter()
                .enumerate()
                .map(|(i, &qi)| qi as f64 * (phi_total[i] - phi_real[i]) / CUS)
                .sum::<f64>();
        let rel = ((e_probe - e_recip) / e_recip.abs()).abs();
        assert!(
            rel < 5e-4,
            "reciprocal mismatch: probe {e_probe} vs pme {e_recip} (rel {rel})"
        );
    }

    #[test]
    fn esp_units_are_elementary() {
        // one +1e charge (internal 18.2223): potential difference at 3 vs 6 A
        // ~ 332.05 * (1/3 - 1/6) = 55.34 kcal/mol/e (real space dominates;
        // periodic corrections are <1% at these radii in an 80 A box).
        let extent = v(80.0, 80.0, 80.0);
        let pos = vec![v(40.0, 40.0, 40.0)];
        let q = vec![18.2223f32];
        let pts = vec![[43.0, 40.0, 40.0], [46.0, 40.0, 40.0]];
        let phi = electrostatic_potential(&pts, &pos, &q, extent, 0.26, 10.0, 1.0);
        let d = phi[0] - phi[1];
        let expected = 332.05 * (1.0 / 3.0 - 1.0 / 6.0);
        assert!(
            (d - expected).abs() / expected < 0.03,
            "single-ion dphi {d} vs coulomb {expected}"
        );
    }

    #[test]
    fn esp_field_single_charge_is_coulomb() {
        // one +1e at origin: E at (3,0,0) points +x with |E| ~ 332.05/9 (real
        // space dominates in a big box); reciprocal/image corrections are small.
        let extent = v(80.0, 80.0, 80.0);
        let pos = vec![v(40.0, 40.0, 40.0)];
        let q = vec![18.2223f32];
        let e = electrostatic_field(&[[43.0, 40.0, 40.0]], &pos, &q, extent, 0.26, 10.0, 1.0);
        let [ex, ey, ez] = e[0];
        let expected = 332.05 / 9.0; // kcal/mol/e/Angstrom
        assert!(ex > 0.0, "E must point away from a + charge, got ex={ex}");
        assert!(
            (ex - expected).abs() / expected < 0.03 && ey.abs() < 1.0 && ez.abs() < 1.0,
            "single-ion E [{ex},{ey},{ez}] vs coulomb {expected} along x"
        );
    }

    /// The analytic field must equal the central difference of the (pinned)
    /// potential. This validates sign + units + the reciprocal k-gradient in one
    /// shot on a real multi-charge PME system: if the FD agrees, both the erfc
    /// gradient and the mesh-sum gradient are right (they are independent code).
    #[test]
    fn esp_field_matches_fd_potential() {
        let extent = v(34.0, 30.0, 28.0); // triclinic-ish box to exercise wrap
        let alpha = 0.26f64;
        let rcut = 10.0f64;
        let mesh = 1.2f64;
        let n = 18usize;
        let mut pos = Vec::with_capacity(n);
        let mut q = Vec::with_capacity(n);
        let mut seed = 0x0bad_c0deu32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f64 / (1u64 << 24) as f64
        };
        for i in 0..n {
            pos.push(Vec3::new(
                (next() * 34.0) as f32,
                (next() * 30.0) as f32,
                (next() * 28.0) as f32,
            ));
            q.push(if i % 2 == 0 { 18.2223 } else { -18.2223 });
        }
        // probe points in open space (>= ~3 A from every charge is not guaranteed
        // for random placement, so pick the box centre region and skip a charge
        // coincidence by testing the FD-vs-analytic agreement is self-consistent;
        // a probe near a charge would make FD wrong, not the analytic field).
        let probes = [[17.0, 15.0, 14.0], [9.0, 22.0, 5.0], [25.0, 7.0, 20.0]];
        let h = 1.0e-3;
        let e_an = electrostatic_field(&probes, &pos, &q, extent, alpha, rcut, mesh);
        for (pi, &p) in probes.iter().enumerate() {
            // nearest-charge distance guard: FD is invalid within ~1.5 A of a charge
            let dmin = pos
                .iter()
                .map(|r| {
                    ((r.x as f64 - p[0]).powi(2)
                        + (r.y as f64 - p[1]).powi(2)
                        + (r.z as f64 - p[2]).powi(2))
                    .sqrt()
                })
                .fold(f64::INFINITY, f64::min);
            assert!(
                dmin > 1.5,
                "probe {pi} too close to a charge (dmin={dmin}); pick another"
            );
            for ax in 0..3 {
                let mut pp = p;
                let mut pm = p;
                pp[ax] += h;
                pm[ax] -= h;
                let fd = -(electrostatic_potential(&[pp], &pos, &q, extent, alpha, rcut, mesh)[0]
                    - electrostatic_potential(&[pm], &pos, &q, extent, alpha, rcut, mesh)[0])
                    / (2.0 * h);
                let ea = e_an[pi][ax];
                let tol = 1.0e-2 + 0.01 * ea.abs(); // abs + 1% rel
                assert!(
                    (fd - ea).abs() < tol,
                    "probe {pi} axis {ax}: analytic E {ea} vs finite-diff {fd} (|d|={})",
                    (fd - ea).abs()
                );
            }
        }
    }

    #[test]
    fn sasa_isolated_atom_is_full_sphere() {
        let extent = v(40.0, 40.0, 40.0);
        let pos = vec![v(20.0, 20.0, 20.0)];
        let radii = vec![1.70f64];
        let out = atom_sasa(&pos, &radii, extent, 1.4, 200);
        let full = 4.0 * std::f64::consts::PI * (3.1f64).powi(2);
        assert!(
            (out[0] - full) / full < 0.02,
            "isolated SASA {} vs {}",
            out[0],
            full
        );
    }

    #[test]
    fn sasa_buried_atom_shrinks() {
        // two atoms touching: each loses a cap to the other
        let extent = v(40.0, 40.0, 40.0);
        let pos = vec![v(20.0, 20.0, 20.0), v(23.4, 20.0, 20.0)];
        let radii = vec![1.7f64; 2];
        let out = atom_sasa(&pos, &radii, extent, 1.4, 200);
        let full = 4.0 * std::f64::consts::PI * 3.1f64.powi(2);
        assert!(out[0] < full * 0.95 && out[0] > full * 0.5);
        assert!((out[0] - out[1]).abs() < 0.15 * full);
    }

    #[test]
    fn contacts_and_bottleneck_basic() {
        let extent = v(40.0, 40.0, 40.0);
        let pos = vec![
            v(10.0, 10.0, 10.0),
            v(11.0, 10.0, 10.0),
            v(30.0, 30.0, 30.0),
        ];
        let a = [0usize, 1, 2];
        assert_eq!(contact_count(&a, &a, &pos, extent, 1.5), 1);
        assert_eq!(contact_count(&a, &a, &pos, extent, 5.0), 1);
        // minimum image: atom 0 and 2 are 20 A apart directly, 20 A wrapped - no contact
        // bottleneck: path through open space has large clearance
        let radii = vec![1.7f64; 3];
        let prof = bottleneck_profile(
            &[[20.0, 20.0, 20.0], [25.0, 20.0, 20.0]],
            1.0,
            &pos,
            &radii,
            extent,
            &[],
        );
        assert!(
            prof.iter().all(|&c| c > 5.0),
            "open-space clearance {prof:?}"
        );
    }
}
