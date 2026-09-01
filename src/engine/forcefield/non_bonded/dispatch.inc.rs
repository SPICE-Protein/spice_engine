/// Abramowitz-Stegun erfc approximation used by the x86_64 SIMD path.
#[cfg(target_arch = "x86_64")]
#[inline]
fn erfc_approx_x8(x: WideF32x8) -> WideF32x8 {
    let t = (WideF32x8::splat(1.0) + WideF32x8::splat(0.3275911) * x).recip();
    let poly = ((((WideF32x8::splat(1.061405429) * t + WideF32x8::splat(-1.453152027)) * t
        + WideF32x8::splat(1.421413741))
        * t
        + WideF32x8::splat(-0.284496736))
        * t
        + WideF32x8::splat(0.254829592))
        * t;
    poly * (-(x * x)).exp()
}

#[cfg(target_arch = "x86_64")]
/// SIMD dispatch for ordinary standard-atom pairs. Complex pair classes keep
/// the scalar reference path so water, 1-4 scaling, alchemical terms, and
/// tails retain their existing semantics.
fn calc_force_cpu_dispatch(
    pairs: &[NonBondedPair],
    preclassified_simd: &[NumericSimdPair],
    preclassified_scalar: &[NonBondedPair],
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
    let mut f_std = vec![Vec3F64::new_zero(); n_std];
    let mut virial = 0.0f64;
    let mut energy = 0.0f64;
    let mut energy_between_mols = vec![0.0f64; n_mol * n_mol];
    // Pair classes are prepared during neighbor-list rebuild. Reuse the
    // contiguous lists on every force evaluation instead of reclassifying and
    // cloning the complete pair list in the MD hot loop.
    if overrides.lj_disabled || overrides.coulomb_disabled {
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

    for chunk in simd_pairs.chunks_exact(8) {
        let mut diffs = [Vec3::new_zero(); 8];
        let mut sigmas = [0.0f32; 8];
        let mut epsilons = [0.0f32; 8];
        let mut q_products = [0.0f32; 8];
        let mut tgt_idx = [0usize; 8];
        let mut src_idx = [0usize; 8];
        for (lane, p) in chunk.iter().enumerate() {
            let diff =
                cell.min_image(atoms_std[p.tgt as usize].posit - atoms_std[p.src as usize].posit);
            diffs[lane] = diff;
            sigmas[lane] = p.sigma;
            epsilons[lane] = p.epsilon;
            q_products[lane] = p.charge_product;
            tgt_idx[lane] = p.tgt as usize;
            src_idx[lane] = p.src as usize;
        }
        let dx = WideF32x8::new(diffs.map(|v| v.x));
        let dy = WideF32x8::new(diffs.map(|v| v.y));
        let dz = WideF32x8::new(diffs.map(|v| v.z));
        let inv_dist = (dx * dx + dy * dy + dz * dz).sqrt().recip();
        let sigma = WideF32x8::new(sigmas);
        let epsilon = WideF32x8::new(epsilons);
        let sr = sigma * inv_dist;
        let sr2 = sr * sr;
        let sr6 = sr2 * sr2 * sr2;
        let sr12 = sr6 * sr6;
        let lj_mag =
            WideF32x8::splat(24.0) * epsilon * (WideF32x8::splat(2.0) * sr12 - sr6) * inv_dist;
        let lj_energy = WideF32x8::splat(4.0) * epsilon * (sr12 - sr6);
        let dist = inv_dist.recip();
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
        for lane in 0..8 {
            let i = tgt_idx[lane];
            let j = src_idx[lane];
            let r = distances[lane];
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
            let f64_force: Vec3F64 = f.into();
            f_std[i] += f64_force;
            f_std[j] -= f64_force;
            let e = e_lj + e_coul;
            energy += e as f64;
            virial += diffs[lane].dot(f) as f64;
            if !energy_between_mols.is_empty() {
                let mi = atom_to_mol[i];
                let mj = atom_to_mol[j];
                energy_between_mols[mi * n_mol + mj] += e as f64;
                if mi != mj {
                    energy_between_mols[mj * n_mol + mi] += e as f64;
                }
            }
        }
    }

    // `setup_pairs` places incomplete SIMD tails into the scalar cache, so
    // there is no per-step temporary pair vector here.
    let (mut rf, rw, rv, re, mut rem, rd) = calc_force_cpu(
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
    for (a, b) in rf.iter_mut().zip(f_std) {
        *a += b;
    }
    for (a, b) in rem.iter_mut().zip(energy_between_mols) {
        *a += b;
    }
    (rf, rw, rv + virial, re + energy, rem, rd)
}

#[cfg(target_arch = "aarch64")]
fn calc_force_cpu_dispatch_arm_lj(
    pairs: &[NonBondedPair],
    preclassified_simd: &[NumericSimdPair],
    preclassified_scalar: &[NonBondedPair],
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
    let mut f_std = vec![Vec3F64::new_zero(); n_std];
    let mut virial = 0.0f64;
    let mut energy = 0.0f64;
    let mut energy_between_mols = vec![0.0f64; n_mol * n_mol];
    // Pair classes are prepared during neighbor-list rebuild. Reuse the
    // contiguous lists on every force evaluation instead of reclassifying and
    // cloning the complete pair list in the MD hot loop.
    if overrides.lj_disabled || overrides.coulomb_disabled {
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

    for chunk in simd_pairs.chunks_exact(8) {
        let mut diffs = [Vec3::new_zero(); 8];
        let mut sigmas = [0.0f32; 8];
        let mut epsilons = [0.0f32; 8];
        let mut q_products = [0.0f32; 8];
        let mut tgt_idx = [0usize; 8];
        let mut src_idx = [0usize; 8];
        for (lane, p) in chunk.iter().enumerate() {
            let diff =
                cell.min_image(atoms_std[p.tgt as usize].posit - atoms_std[p.src as usize].posit);
            diffs[lane] = diff;
            sigmas[lane] = p.sigma;
            epsilons[lane] = p.epsilon;
            q_products[lane] = p.charge_product;
            tgt_idx[lane] = p.tgt as usize;
            src_idx[lane] = p.src as usize;
        }
        let dx = WideF32x8::new(diffs.map(|v| v.x));
        let dy = WideF32x8::new(diffs.map(|v| v.y));
        let dz = WideF32x8::new(diffs.map(|v| v.z));
        let inv_dist = (dx * dx + dy * dy + dz * dz).sqrt().recip();
        let sigma = WideF32x8::new(sigmas);
        let epsilon = WideF32x8::new(epsilons);
        let sr = sigma * inv_dist;
        let sr2 = sr * sr;
        let sr6 = sr2 * sr2 * sr2;
        let sr12 = sr6 * sr6;
        let lj_mag =
            WideF32x8::splat(24.0) * epsilon * (WideF32x8::splat(2.0) * sr12 - sr6) * inv_dist;
        let lj_energy = WideF32x8::splat(4.0) * epsilon * (sr12 - sr6);
        let fx = (dx * inv_dist * lj_mag).to_array();
        let fy = (dy * inv_dist * lj_mag).to_array();
        let fz = (dz * inv_dist * lj_mag).to_array();
        let distances = inv_dist.recip().to_array();
        let lj_e = lj_energy.to_array();
        for lane in 0..8 {
            let i = tgt_idx[lane];
            let j = src_idx[lane];
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
                atoms_std[j].partial_charge,
                coulomb_cutoff,
                spme_alpha,
            );
            if r > coulomb_cutoff {
                f_coul = Vec3::new_zero();
                e_coul = 0.0;
            }
            let f = f_lj + f_coul;
            let f64_force: Vec3F64 = f.into();
            f_std[i] += f64_force;
            f_std[j] -= f64_force;
            let e = e_lj + e_coul;
            energy += e as f64;
            virial += diffs[lane].dot(f) as f64;
            if !energy_between_mols.is_empty() {
                let mi = atom_to_mol[i];
                let mj = atom_to_mol[j];
                energy_between_mols[mi * n_mol + mj] += e as f64;
                if mi != mj {
                    energy_between_mols[mj * n_mol + mi] += e as f64;
                }
            }
        }
    }

    // `setup_pairs` places incomplete SIMD tails into the scalar cache, so
    // there is no per-step temporary pair vector here.
    let (mut rf, rw, rv, re, mut rem, rd) = calc_force_cpu(
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
    for (a, b) in rf.iter_mut().zip(f_std) {
        *a += b;
    }
    for (a, b) in rem.iter_mut().zip(energy_between_mols) {
        *a += b;
    }
    (rf, rw, rv + virial, re + energy, rem, rd)
}

#[cfg(target_arch = "aarch64")]
fn calc_force_cpu_dispatch(
    pairs: &[NonBondedPair],
    preclassified_simd: &[NumericSimdPair],
    preclassified_scalar: &[NonBondedPair],
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
    calc_force_cpu_dispatch_arm_lj(
        pairs,
        preclassified_simd,
        preclassified_scalar,
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
fn calc_force_cpu_dispatch(
    pairs: &[NonBondedPair],
    _preclassified_simd: &[NumericSimdPair],
    _preclassified_scalar: &[NonBondedPair],
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

