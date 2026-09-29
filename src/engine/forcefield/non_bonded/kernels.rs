use super::*;

/// Beutler/GROMACS-style soft-core LJ decoupling for a pair with B-state LJ set
/// to zero.
///
/// Returns `(force, energy, dH/dlambda)` for
/// `V_sc(r, lambda) = (1 - lambda) * V_LJ(r_sc)`.
pub(crate) fn alchemical_lj_soft_core_decouple(
    dir: Vec3,
    dist_sq: f32,
    sigma: f32,
    eps: f32,
    lambda: f32,
) -> (Vec3, f32, f32) {
    if eps == 0.0 || !eps.is_finite() || !sigma.is_finite() || dist_sq < 0.0 {
        return (Vec3::new_zero(), 0.0, 0.0);
    }

    let lambda = lambda.clamp(0.0, 1.0);
    let scale = 1.0 - lambda;

    if SOFT_CORE_ALPHA <= 0.0 {
        let dist = dist_sq.sqrt();
        if dist <= 0.0 {
            return (Vec3::new_zero(), 0.0, 0.0);
        }
        let (force, energy) = force_e_lj(dir, 1.0 / dist, sigma, eps);
        return (force * scale, energy * scale, -energy);
    }

    let soft_sigma = sigma.max(SOFT_CORE_SIGMA_MIN);
    let soft_sigma2 = soft_sigma * soft_sigma;
    let soft_sigma6 = soft_sigma2 * soft_sigma2 * soft_sigma2;
    let dist6 = dist_sq * dist_sq * dist_sq;
    let lambda_power = lambda.powi(SOFT_CORE_POWER);
    let r_sc6 = dist6 + SOFT_CORE_ALPHA * soft_sigma6 * lambda_power;

    if r_sc6 <= 0.0 || !r_sc6.is_finite() {
        return (Vec3::new_zero(), 0.0, 0.0);
    }

    let r_sc = r_sc6.powf(1.0 / 6.0);
    let inv_r_sc = 1.0 / r_sc;
    let sr = sigma * inv_r_sc;
    let sr2 = sr * sr;
    let sr6 = sr2 * sr2 * sr2;
    let sr12 = sr6 * sr6;
    let hard_force_mag = 24.0 * eps * 2.0f32.mul_add(sr12, -sr6) * inv_r_sc;
    let hard_energy = 4.0 * eps * (sr12 - sr6);
    let hard_force = dir * hard_force_mag;

    let dist = dist_sq.sqrt();
    let soft_ratio = dist * inv_r_sc;
    let soft_ratio2 = soft_ratio * soft_ratio;
    let force_softening = soft_ratio2 * soft_ratio2 * soft_ratio;
    let force = hard_force * (scale * force_softening);
    let energy = hard_energy * scale;

    let lambda_power_deriv = if SOFT_CORE_POWER == 1 {
        1.0
    } else {
        lambda.powi(SOFT_CORE_POWER - 1)
    };
    let dr_sc_dlambda =
        (SOFT_CORE_POWER as f32) * SOFT_CORE_ALPHA * soft_sigma6 * lambda_power_deriv
            / (6.0 * {
                let r_sc2 = r_sc * r_sc;
                r_sc2 * r_sc2 * r_sc
            });
    let dh_dl = -hard_energy - scale * hard_force_mag * dr_sc_dlambda;

    (force, energy, dh_dl)
}

#[allow(clippy::too_many_arguments)]
/// Lennard Jones and (short-range) Coulomb forces. Used by solvent and non-solvent.
/// We run long-range SPME Coulomb force separately.
///
/// We use a hard distance cutoff for Vdw, enabled by its ^-7 falloff.
/// Returns force, potential energy, and this pair's alchemical dH/dlambda
/// contribution. The derivative is zero for ordinary pairs.
pub fn f_nonbonded_cpu(
    virial_w: &mut f64,
    tgt: &AtomDynamics,
    src: &AtomDynamics,
    cell: &SimBox,
    scale14: bool, // See notes earlier in this module.
    lj_indices: &LjTableIndices,
    lj_tables: &LjTables,
    // These flags are for use with forces on solvent.
    calc_lj: bool,
    calc_coulomb: bool,
    overrides: &MdOverrides,
    spme_alpha: f32,
    coulomb_cutoff: f32,
    lj_cutoff: f32,
    alchemical_lambda: Option<f32>,
) -> (Vec3, f32, f32) {
    let diff = cell.min_image(tgt.posit - src.posit);

    // We compute these dist-related values once, and share them between
    // LJ and Coulomb.
    let dist_sq = diff.magnitude_squared();
    let alchemical_lambda = alchemical_lambda.map(|lambda| lambda.clamp(0.0, 1.0));

    if dist_sq < 1e-12 {
        if let Some(lambda) = alchemical_lambda
            && calc_lj
            && !overrides.lj_disabled
        {
            let schedule = staged_decoupling_schedule(lambda as f64);
            let (σ, ε, _) = lj_tables.lookup(lj_indices);
            let (mut f, mut e, mut dh_dl) =
                alchemical_lj_soft_core_decouple(Vec3::new_zero(), 0.0, σ, ε, schedule.lj_lambda);
            dh_dl *= schedule.lj_dlambda_dlambda;

            if scale14 {
                f *= SCALE_LJ_14;
                e *= SCALE_LJ_14;
                dh_dl *= SCALE_LJ_14;
            }
            return (f, e, dh_dl);
        }
        return (Vec3::new_zero(), 0., 0.);
    }

    // LAMMPS-style early exit on dist² BEFORE the sqrt: the neighbor list is
    // built out to cutoff+skin, so a big fraction of the checked pairs lie in
    // the skin shell and contribute nothing. Skip the sqrt/div/dir work for
    // them (unless an alchemical lambda path needs the soft-core handling).
    if alchemical_lambda.is_none() {
        let lj_active = calc_lj && !overrides.lj_disabled;
        let coul_active = calc_coulomb && !overrides.coulomb_disabled;
        let in_lj = !lj_active || dist_sq < lj_cutoff * lj_cutoff;
        let in_coul = !coul_active || dist_sq < coulomb_cutoff * coulomb_cutoff;
        if !in_lj && !in_coul {
            return (Vec3::new_zero(), 0., 0.);
        }
    }

    let dist = dist_sq.sqrt();
    let inv_dist = 1.0 / dist;
    let dir = diff * inv_dist;

    let schedule = alchemical_lambda.map(|lambda| staged_decoupling_schedule(lambda as f64));

    let (f_lj, energy_lj, dh_dl_lj) = if !calc_lj || dist > lj_cutoff || overrides.lj_disabled {
        (Vec3::new_zero(), 0., 0.)
    } else {
        let (σ, ε, c4) = lj_tables.lookup(lj_indices);

        let (mut f, mut e, mut dh_dl) = if let Some(schedule) = schedule {
            let (f, e, dh_dl) =
                alchemical_lj_soft_core_decouple(dir, dist_sq, σ, ε, schedule.lj_lambda);
            (f, e, dh_dl * schedule.lj_dlambda_dlambda)
        } else {
            let (f, e) = force_e_lj_c4(dir, inv_dist, σ, ε, c4);
            (f, e, 0.)
        };
        if scale14 {
            f *= SCALE_LJ_14;
            e *= SCALE_LJ_14;
            dh_dl *= SCALE_LJ_14;
        }
        (f, e, dh_dl)
    };

    // We assume that in the AtomDynamics structs, charges are already scaled to Amber units.
    // (No longer in elementary charge)
    let (mut f_coulomb, mut energy_coulomb) = if !calc_coulomb || overrides.coulomb_disabled {
        (Vec3::new_zero(), 0.)
    } else {
        force_coulomb_short_range(
            dir,
            dist,
            inv_dist,
            tgt.partial_charge,
            src.partial_charge,
            coulomb_cutoff,
            spme_alpha,
        )
    };

    // See Amber RM, section 15, "1-4 Non-Bonded Interaction Scaling"
    if scale14 {
        f_coulomb *= SCALE_COUL_14;
        energy_coulomb *= SCALE_COUL_14;
    }

    let (force, energy, dh_dl) = if let Some(schedule) = schedule {
        (
            f_lj + f_coulomb * schedule.coulomb_scale,
            energy_lj + energy_coulomb * schedule.coulomb_scale,
            dh_dl_lj + energy_coulomb * schedule.coulomb_dscale_dlambda,
        )
    } else {
        (f_lj + f_coulomb, energy_lj + energy_coulomb, 0.)
    };

    *virial_w += diff.dot(force) as f64;

    (force, energy, dh_dl)
}

/// Specialized OPC water–water kernel (rigid, LAMMPS tip4p-style).
///
/// One call per water–water molecule pair instead of ~10 generic site-pair
/// invocations. The O–O minimum-image displacement is computed ONCE and the
/// other site–site displacements are derived by adding the small rigid
/// intramolecular offsets (box >> molecule, so this stays in the correct
/// periodic image — the standard rigid-water trick in GROMACS/OpenMM/LAMMPS).
/// Forces accumulate directly into both molecules' per-site accumulators.
/// Water–water energy is returned but the caller excludes it from the reported
/// total (matches the generic path, which ignores solvent-only energy).
///
/// `wa` is the target water, `wb` the source: forces on `wa` come out positive
/// (repulsion/attraction along the O→O direction), `wb` receives the opposite.
#[allow(clippy::too_many_arguments)]
pub(super) fn f_water_water_cpu(
    virial_w: &mut f64,
    f_wa: &mut ForcesOnWaterMol,
    f_wb: &mut ForcesOnWaterMol,
    wa: &WaterMolOpc,
    wb: &WaterMolOpc,
    cell: &SimBox,
    lj_tables: &LjTables,
    overrides: &MdOverrides,
    spme_alpha: f32,
    coulomb_cutoff: f32,
    lj_cutoff: f32,
) -> f64 {
    let o_a = wa.o.posit;
    let o_b = wb.o.posit;
    let d_oo = cell.min_image(o_a - o_b);
    let mut energy = 0.0f64;

    // Cluster-style broad phase: every charged water site is within O_H_R
    // of its oxygen. If the oxygen centers are farther than cutoff plus the
    // rigid-body diameter, no site-site interaction can be active.
    let max_cutoff = coulomb_cutoff.max(lj_cutoff);
    let water_extent = 2.0 * O_H_R;
    let broad_cutoff = max_cutoff + water_extent;
    if d_oo.magnitude_squared() >= broad_cutoff * broad_cutoff {
        return 0.0;
    }

    // --- O–O Lennard-Jones (only O carries LJ params in OPC) ---
    if !overrides.lj_disabled {
        let r2 = d_oo.magnitude_squared();
        if r2 > 1e-12 && r2 < lj_cutoff * lj_cutoff {
            let r = r2.sqrt();
            let inv = 1.0 / r;
            let dir = d_oo * inv;
            let (sigma, eps, _c4) = lj_tables.lookup(&LjTableIndices::WaterWater);
            let (f, e) = force_e_lj(dir, inv, sigma, eps);
            let f64v: Vec3F64 = f.into();
            f_wa.f_o += f64v;
            f_wb.f_o -= f64v;
            energy += e as f64;
            *virial_w += d_oo.dot(f) as f64;
        }
    }

    // --- Coulomb between charged sites {M, H0, H1} × {M, H0, H1} ---
    // Explicit 3x3 expansion avoids constructing two arrays and repeatedly
    // dispatching WaterSite in the inner loop.  This is the fixed OPC analogue
    // of a LAMMPS pair style with preclassified site types.
    if !overrides.coulomb_disabled {
        let cutoff_sq = coulomb_cutoff * coulomb_cutoff;
        let om = wa.m.posit - o_a;
        let oh0 = wa.h0.posit - o_a;
        let oh1 = wa.h1.posit - o_a;
        let pm = wb.m.posit - o_b;
        let ph0 = wb.h0.posit - o_b;
        let ph1 = wb.h1.posit - o_b;
        macro_rules! site_pair {
            ($da:expr, $db:expr, $qa:expr, $qb:expr, $fa:ident, $fb:ident) => {{
                let delta = d_oo + $da - $db;
                let r2 = delta.magnitude_squared();
                if r2 >= 1e-12 && r2 < cutoff_sq {
                    let r = r2.sqrt();
                    let inv = 1.0 / r;
                    let f = force_coulomb_short_range(
                        delta * inv,
                        r,
                        inv,
                        $qa,
                        $qb,
                        coulomb_cutoff,
                        spme_alpha,
                    );
                    let f64v: Vec3F64 = f.0.into();
                    f_wa.$fa += f64v;
                    f_wb.$fb -= f64v;
                    energy += f.1 as f64;
                    *virial_w += delta.dot(f.0) as f64;
                }
            }};
        }
        site_pair!(om, pm, wa.m.partial_charge, wb.m.partial_charge, f_m, f_m);
        site_pair!(
            om,
            ph0,
            wa.m.partial_charge,
            wb.h0.partial_charge,
            f_m,
            f_h0
        );
        site_pair!(
            om,
            ph1,
            wa.m.partial_charge,
            wb.h1.partial_charge,
            f_m,
            f_h1
        );
        site_pair!(
            oh0,
            pm,
            wa.h0.partial_charge,
            wb.m.partial_charge,
            f_h0,
            f_m
        );
        site_pair!(
            oh0,
            ph0,
            wa.h0.partial_charge,
            wb.h0.partial_charge,
            f_h0,
            f_h0
        );
        site_pair!(
            oh0,
            ph1,
            wa.h0.partial_charge,
            wb.h1.partial_charge,
            f_h0,
            f_h1
        );
        site_pair!(
            oh1,
            pm,
            wa.h1.partial_charge,
            wb.m.partial_charge,
            f_h1,
            f_m
        );
        site_pair!(
            oh1,
            ph0,
            wa.h1.partial_charge,
            wb.h0.partial_charge,
            f_h1,
            f_h0
        );
        site_pair!(
            oh1,
            ph1,
            wa.h1.partial_charge,
            wb.h1.partial_charge,
            f_h1,
            f_h1
        );
    }

    energy
}

/// Specialized OPC water–solute kernel: O-solute LJ + (M,H0,H1)-solute Coulomb,
/// one minimum-image displacement per O-solute pair (rigid-offset trick).
/// `atom_std` is the solute atom; forces on it accumulate into `f_std`, and the
/// water's per-site forces into `f_wat`. Returns the pair energy (reported — it
/// IS part of the total, since it involves the solute).
#[allow(clippy::too_many_arguments)]
pub(super) fn f_water_std_cpu(
    virial_w: &mut f64,
    f_std: &mut Vec3F64,
    f_wat: &mut ForcesOnWaterMol,
    atom_std: &AtomDynamics,
    w: &WaterMolOpc,
    cell: &SimBox,
    _lj_tables: &LjTables,
    overrides: &MdOverrides,
    spme_alpha: f32,
    coulomb_cutoff: f32,
    lj_cutoff: f32,
    _std_idx: usize,
) -> f64 {
    let p_std = atom_std.posit;
    let o_w = w.o.posit;
    let d_o = cell.min_image(o_w - p_std); // water O relative to solute
    let mut energy = 0.0f64;

    // Broad phase for the compact water-solute cluster. All water sites are
    // within O_H_R of O, so a center distance beyond cutoff + O_H_R cannot
    // produce any LJ or Coulomb interaction.
    let max_cutoff = coulomb_cutoff.max(lj_cutoff);
    let broad_cutoff = max_cutoff + O_H_R;
    if d_o.magnitude_squared() >= broad_cutoff * broad_cutoff {
        return 0.0;
    }

    // --- O(std)–O(water) LJ ---
    if !overrides.lj_disabled {
        let r2 = d_o.magnitude_squared();
        if r2 > 1e-12 && r2 < lj_cutoff * lj_cutoff {
            let r = r2.sqrt();
            let inv = 1.0 / r;
            let dir = d_o * inv;
            // Water oxygen has one fixed LJ type; avoid the per-pair table
            // lookup in this hot path.
            let sigma = 0.5 * (atom_std.lj_sigma + O_SIGMA);
            let eps = (atom_std.lj_eps * O_EPS).sqrt();
            // `lj_c4` is nonzero only for the Li-Merz 12-6-4 divalent ions;
            // every other solute gets the bitwise-identical plain-LJ path.
            let (f, e) = force_e_lj_c4(dir, inv, sigma, eps, atom_std.lj_c4);
            let f64v: Vec3F64 = f.into();
            // `f` is the force on the water O (tgt of d_o); solute gets the opposite.
            *f_std -= f64v;
            f_wat.f_o += f64v;
            energy += e as f64;
            *virial_w += d_o.dot(f) as f64;
        }
    }

    // --- Coulomb: (M, H0, H1) of water vs solute ---
    // Keep the three fixed OPC sites explicit.  Besides avoiding a stack array
    // and enum dispatch in this hottest water-solute loop, this lets LLVM keep
    // the solute charge/cutoff values in registers.  The operation order is
    // intentionally M, H0, H1, matching the previous iterator.
    if !overrides.coulomb_disabled {
        let q_std = atom_std.partial_charge;
        let cutoff_sq = coulomb_cutoff * coulomb_cutoff;

        let delta_m = d_o + (w.m.posit - o_w);
        let r2_m = delta_m.magnitude_squared();
        if r2_m >= 1e-12 && r2_m < cutoff_sq {
            let r = r2_m.sqrt();
            let inv = 1.0 / r;
            let f = force_coulomb_short_range(
                delta_m * inv,
                r,
                inv,
                w.m.partial_charge,
                q_std,
                coulomb_cutoff,
                spme_alpha,
            );
            let f64v: Vec3F64 = f.0.into();
            *f_std -= f64v;
            f_wat.f_m += f64v;
            energy += f.1 as f64;
            *virial_w += delta_m.dot(f.0) as f64;
        }

        let delta_h0 = d_o + (w.h0.posit - o_w);
        let r2_h0 = delta_h0.magnitude_squared();
        if r2_h0 >= 1e-12 && r2_h0 < cutoff_sq {
            let r = r2_h0.sqrt();
            let inv = 1.0 / r;
            let f = force_coulomb_short_range(
                delta_h0 * inv,
                r,
                inv,
                w.h0.partial_charge,
                q_std,
                coulomb_cutoff,
                spme_alpha,
            );
            let f64v: Vec3F64 = f.0.into();
            *f_std -= f64v;
            f_wat.f_h0 += f64v;
            energy += f.1 as f64;
            *virial_w += delta_h0.dot(f.0) as f64;
        }

        let delta_h1 = d_o + (w.h1.posit - o_w);
        let r2_h1 = delta_h1.magnitude_squared();
        if r2_h1 >= 1e-12 && r2_h1 < cutoff_sq {
            let r = r2_h1.sqrt();
            let inv = 1.0 / r;
            let f = force_coulomb_short_range(
                delta_h1 * inv,
                r,
                inv,
                w.h1.partial_charge,
                q_std,
                coulomb_cutoff,
                spme_alpha,
            );
            let f64v: Vec3F64 = f.0.into();
            *f_std -= f64v;
            f_wat.f_h1 += f64v;
            energy += f.1 as f64;
            *virial_w += delta_h1.dot(f.0) as f64;
        }
    }

    energy
}

pub(super) fn atom_to_mol_indices(n_atoms: usize, mol_start_indices: &[usize]) -> Vec<usize> {
    validate_mol_start_indices(n_atoms, mol_start_indices).expect("invalid molecule start indices");

    let mut atom_to_mol = vec![0; n_atoms];

    for (mol_idx, &start) in mol_start_indices.iter().enumerate() {
        let end = mol_start_indices
            .get(mol_idx + 1)
            .copied()
            .unwrap_or(n_atoms);

        for atom_idx in start..end {
            atom_to_mol[atom_idx] = mol_idx;
        }
    }

    atom_to_mol
}

/// Helper. Returns σ, ε between an atom pair. Atom order passed as params doesn't matter.
/// Note that this uses the traditional algorithm; not the Amber-specific version: We pre-set
/// atom-specific σ and ε to traditional versions on ingest, and when building solvent.
pub(crate) fn combine_lj_params(atom_0: &AtomDynamics, atom_1: &AtomDynamics) -> (f32, f32, f32) {
    let σ = 0.5 * (atom_0.lj_sigma + atom_1.lj_sigma);
    let ε = (atom_0.lj_eps * atom_1.lj_eps).sqrt();
    // 12-6-4 pair rule for solute–solute (std–std) contacts. Amber's own
    // 12-6-4 machinery combines per-type C4 coefficients with the same
    // geometric mean as ε (Panteva–Giambasu–York site corrections rely on
    // exactly this: a site carrying an effective C4 gets sqrt(c4_site ·
    // c4_ion) against the metal). A single-sided C4 stays 0 here: ion–water
    // induction is the water_std table's one-sided channel, and ordinary
    // protein/salt atoms (c4 = 0) are untouched — bit-for-bit for everything
    // except cation–cation pairs in multi-divalent systems, which now carry
    // the Amber-consistent mutual term — sub-kcal at contact, and it is a
    // correction Amber itself prescribes, so enabling it is the conservative
    // choice, not a new interaction.
    // Opposite-sign pairs (future anion C4) would be sqrt of
    // negative: undefined → 0, never NaN.
    let c4 = if atom_0.lj_c4 > 0.0 && atom_1.lj_c4 > 0.0 {
        (atom_0.lj_c4 * atom_1.lj_c4).sqrt()
    } else {
        0.0
    };

    (σ, ε, c4)
}
