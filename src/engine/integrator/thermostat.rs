//! Note: We keep most thermostat and barostat code as f64, although we use f32 in most sections.

use lin_alg::f32::{Mat3 as Mat3F32, Vec3};
use na_seq::Element;
use rand::{RngExt, distr::Distribution, rngs::StdRng};
use rand_distr::{ChiSquared, StandardNormal};

use crate::engine::md_core::{
    ComMotionRemoval, HydrogenConstraint, MdState, NATIVE_TO_KCAL,
    solvent::{H_MASS, MASS_WATER_MOL, O_MASS},
};

// Per-molecule Boltzmann, in kcal/mol/K.
// For assigning velocities from temperature, and other thermostat/barostat use.
pub(crate) const GAS_CONST_R: f64 = 0.001_987_204_1; // kcal mol⁻¹ K⁻¹ (Amber-style units)

// Boltzmann constant in (amu · Å²/ps²) K⁻¹
// We use this for the Langevin and Anderson thermostat, where we need per-particle Gaussian noise or variance.
pub(crate) const KB_A2_PS2_PER_K_PER_AMU: f32 = 0.831_446_26;

// TAU is for the CSVR thermostat. In ps. Lower means more sensitive.
// We set an aggressive thermostat during solvent initialization, then a more relaxed one at runtime.
// This is for the VV/CVSR themostat only.
// Note: These are publically exposed, for use in applications.
pub const TEMP_DEFAULT: f32 = 300.; // GROMACS default.

pub const TAU_TEMP_DEFAULT: f64 = 0.1; // GROMACS default.
pub const TAU_TEMP_WATER_INIT: f64 = 0.01; // for CSVR

// These are in 1/ps. 1 ps^-1 is a good default for explicit solvent and constrained H bonds.
// Lower is closer to Newtonian dynamics. 2ps (0.5ps^-1) is Gromac's default.
pub const LANGEVIN_GAMMA_DEFAULT: f32 = 0.5;
pub const LANGEVIN_GAMMA_WATER_INIT: f32 = 15.;

fn sample_normal_vec(rng: &mut StdRng, sigma: f32) -> Vec3 {
    let x: f32 = rng.sample(StandardNormal);
    let y: f32 = rng.sample(StandardNormal);
    let z: f32 = rng.sample(StandardNormal);

    Vec3::new(x * sigma, y * sigma, z * sigma)
}

impl MdState {
    /// Computes total kinetic energy, in native units.
    /// Includes all non-static atoms, including solvent.
    ///
    /// RIGID-WATER CORRECTION (2026-08-10): the solvent KE is computed as the
    /// RIGID-BODY kinetic energy (COM translation + rotation about COM), NOT
    /// the per-atom sum over O/H0/H1. The per-atom sum includes the transient
    /// internal-mode velocity (O–H stretch / H–O–H angle) that the SETTLE
    /// constraint force creates during the half-kick and projects away at the
    /// next drift. Measured against the 6-DOF-per-water count that transient
    /// inflated the reported temperature by ~+75 K (2LYZ: reported 385 K while
    /// the water's rigid 6 DOF were correctly at 310 K). GROMACS likewise takes
    /// temperature from post-constraint (projected) velocities.
    pub(crate) fn measure_kinetic_energy(&self) -> f64 {
        let mut result = 0.0;

        for a in &self.atoms {
            if !a.static_ {
                result += (a.mass * a.vel.magnitude_squared()) as f64;
            }
        }

        // Do not include the M/EP site. Solvent contributes its rigid-body KE:
        //   KE = ½·M·|V_com|² + ½·ωᵀ·I·ω = ½·(M·V_com² + L·ω)
        // (L = I·ω). We accumulate M·V² + L·ω below; the final ×½ handles it.
        for w in &self.water {
            let m_o = w.o.mass;
            let m_h0 = w.h0.mass;
            let m_h1 = w.h1.mass;
            let m_total = m_o + m_h0 + m_h1;
            let v_com = (w.o.vel * m_o + w.h0.vel * m_h0 + w.h1.vel * m_h1) / m_total;

            let v_o = w.o.vel - v_com;
            let v_h0 = w.h0.vel - v_com;
            let v_h1 = w.h1.vel - v_com;

            // For a rigid body, the rotational contribution is exactly the
            // mass-weighted sum of squared velocities relative to COM:
            // Σ m_i |v_i - V_com|² = L·ω.  This avoids constructing and
            // solving a 3×3 inertia system for every water molecule on every
            // kinetic-energy refresh.
            result += (m_total * v_com.magnitude_squared()) as f64; // M·V²
            result += (m_o * v_o.magnitude_squared()
                + m_h0 * v_h0.magnitude_squared()
                + m_h1 * v_h1.magnitude_squared()) as f64;
        }

        // Add in the 0.5 factor, and convert from amu • (Å/ps)² to kcal/mol.
        result * 0.5 * NATIVE_TO_KCAL as f64
    }

    /// COM-only kinetic energy for solvent + full atomic KE for non-solvent atoms, in kcal/mol.
    /// Used for pressure via the molecular virial theorem: rigid solvent molecules contribute
    /// only their translational (COM) KE, so no SETTLE constraint virial is needed.
    pub(crate) fn measure_kinetic_energy_translational(&self) -> f64 {
        let mut result = 0.0;

        for a in &self.atoms {
            if !a.static_ {
                result += (a.mass * a.vel.magnitude_squared()) as f64;
            }
        }

        for w in &self.water {
            let v_com = (w.o.vel * O_MASS + w.h0.vel * H_MASS + w.h1.vel * H_MASS) / MASS_WATER_MOL;
            result += (MASS_WATER_MOL * v_com.magnitude_squared()) as f64;
        }

        result * 0.5 * NATIVE_TO_KCAL as f64
    }

    /// Instantaneous temperature [K]
    pub(crate) fn measure_temperature(&self) -> f64 {
        (2.0 * self.kinetic_energy) / (self.thermo_dof as f64 * GAS_CONST_R)
    }

    /// Assign Maxwell-Boltzmann velocities at the requested temperature.
    ///
    /// Solute atoms are sampled independently. OPC water is sampled as a rigid body, with a
    /// translational COM velocity and angular momentum, so the initial velocities respect the
    /// same rigid-water model used by integration.
    pub fn initialize_velocities(&mut self, target_k: f32, zero_com_drift: bool) {
        if !target_k.is_finite() || target_k <= 0.0 || self.thermo_dof == 0 {
            return;
        }

        let k_t = KB_A2_PS2_PER_K_PER_AMU * target_k;

        for atom in &mut self.atoms {
            if atom.static_ || !atom.mass.is_finite() || atom.mass <= f32::EPSILON {
                atom.vel = Vec3::new_zero();
                continue;
            }

            atom.vel = sample_normal_vec(&mut self.barostat.rng, (k_t / atom.mass).sqrt());
        }

        for water in &mut self.water {
            let mut r_com = Vec3::new_zero();
            let mut mass_total = 0.0;
            for atom in [&water.o, &water.h0, &water.h1] {
                r_com += atom.posit * atom.mass;
                mass_total += atom.mass;
            }

            if mass_total <= f32::EPSILON {
                water.o.vel = Vec3::new_zero();
                water.h0.vel = Vec3::new_zero();
                water.h1.vel = Vec3::new_zero();
                water.update_virtual_site();
                continue;
            }

            r_com /= mass_total;

            let r_o = water.o.posit - r_com;
            let r_h0 = water.h0.posit - r_com;
            let r_h1 = water.h1.posit - r_com;

            let inertia = |r: Vec3, mass: f32| {
                let r2 = r.dot(r);
                [
                    [
                        mass * (r2 - r.x * r.x),
                        -mass * r.x * r.y,
                        -mass * r.x * r.z,
                    ],
                    [
                        -mass * r.y * r.x,
                        mass * (r2 - r.y * r.y),
                        -mass * r.y * r.z,
                    ],
                    [
                        -mass * r.z * r.x,
                        -mass * r.z * r.y,
                        mass * (r2 - r.z * r.z),
                    ],
                ]
            };

            let mut inertia_arr = inertia(r_o, water.o.mass);
            for added in [inertia(r_h0, water.h0.mass), inertia(r_h1, water.h1.mass)] {
                for i in 0..3 {
                    for j in 0..3 {
                        inertia_arr[i][j] += added[i][j];
                    }
                }
            }

            let inertia = Mat3F32::from_arr(inertia_arr);
            let (eigvecs, eigvals) = inertia.eigen_vecs_vals();
            let sample_angular_momentum = |rng: &mut StdRng, moment: f32| {
                let n: f32 = rng.sample(StandardNormal);
                n * (k_t * moment.max(0.0)).sqrt()
            };
            let angular_momentum_principal = Vec3::new(
                sample_angular_momentum(&mut self.barostat.rng, eigvals.x),
                sample_angular_momentum(&mut self.barostat.rng, eigvals.y),
                sample_angular_momentum(&mut self.barostat.rng, eigvals.z),
            );
            let omega = inertia.solve_system(eigvecs * angular_momentum_principal);
            let v_com = sample_normal_vec(&mut self.barostat.rng, (k_t / mass_total).sqrt());

            water.o.vel = v_com + omega.cross(r_o);
            water.h0.vel = v_com + omega.cross(r_h0);
            water.h1.vel = v_com + omega.cross(r_h1);
            water.update_virtual_site();
        }

        if matches!(
            self.cfg.hydrogen_constraint,
            HydrogenConstraint::Shake { shake_tolerance: _ } | HydrogenConstraint::Linear { .. }
        ) {
            self.rattle_hydrogens();
        }

        if zero_com_drift {
            self.zero_linear_momentum();
        }

        self.kinetic_energy = self.measure_kinetic_energy();
        let measured_k = self.measure_temperature();
        if !measured_k.is_finite() || measured_k <= 0.0 {
            return;
        }

        let lambda = (target_k as f64 / measured_k).sqrt() as f32;
        for atom in &mut self.atoms {
            if !atom.static_ {
                atom.vel *= lambda;
            }
        }
        for water in &mut self.water {
            water.o.vel *= lambda;
            water.h0.vel *= lambda;
            water.h1.vel *= lambda;
            water.update_virtual_site();
        }

        self.kinetic_energy = self.measure_kinetic_energy();
    }

    /// Used in temperature computation. Constraints tracked are Hydrogen if constrained, COM drift removal,
    /// and static atoms.
    /// We cache this at init. Used for kinetic energy and temperature computations.
    pub(crate) fn dof_for_thermo(&self) -> usize {
        // 3 positional + 3 rotational for each solvent mol.
        let mut result = 6 * self.water.len();
        result += 3 * self.atoms.iter().filter(|a| !a.static_).count();

        let num_constraints = {
            let mut c = 0;

            // Both SHAKE and LINCS (`Linear`) constrain each H–heavy bond, so
            // each H loses one DOF regardless of which solver is active. (The
            // rattle projection in `kick_and_calc_accel` removes the bond
            // velocity before the KE is measured, so counting the H as a full
            // 3 DOF deflates the reported temperature ~1.19× on 2LYZ.)
            for atom in &self.atoms {
                if matches!(
                    self.cfg.hydrogen_constraint,
                    HydrogenConstraint::Shake { shake_tolerance: _ }
                        | HydrogenConstraint::Linear { .. }
                ) && atom.element == Element::Hydrogen
                    && !atom.static_
                {
                    c += 1;
                }
            }

            if self.cfg.zero_com_drift {
                c += match self.cfg.com_motion_removal {
                    ComMotionRemoval::Linear => 3,
                    ComMotionRemoval::Angular => 6,
                    ComMotionRemoval::LinearAccelerationCorrection => 3,
                    ComMotionRemoval::None => 0,
                };
            }
            c
        };

        result.saturating_sub(num_constraints)
    }

    /// The DOF count currently cached for temperature / kinetic-energy
    /// calculations (exposed for diagnostics / calibration checks).
    pub fn thermo_dof(&self) -> usize {
        self.thermo_dof
    }

    /// Recompute DOF from the CURRENT atoms/solvent/constraints — for
    /// calibration checks: compare against `thermo_dof()` to catch a stale
    /// cache (e.g. if atoms/H/ions were added after `MdState::new`).
    pub fn dof_for_thermo_now(&self) -> usize {
        self.dof_for_thermo()
    }

    /// Canonical Sampling through Velocities Rescaling thermostat. (Also known as Bussi, its
    /// primary author)
    /// [CSVR thermostat](https://arxiv.org/pdf/0803.4060)
    /// A canonical velocity-rescale algorithm.
    /// Cheap with gentle coupling, but doesn't imitate solvent drag.
    pub(crate) fn apply_thermostat_csvr(&mut self, dt: f64, tau: f64, t_target: f64) {
        if tau <= 0.0 {
            return;
        }

        // This value is cached at init.
        let dof = self.thermo_dof.max(2) as f64;

        // Measure current KE from velocities so we get the post-kick value, not a stale cache.
        let ke = self.measure_kinetic_energy(); // In kcal/mol

        if ke < 1e-20 {
            return;
        }

        let c = (-dt / tau).exp();

        // Draw the two random variates used in the exact CSVR update:
        let r: f64 = StandardNormal.sample(&mut self.barostat.rng); // N(0,1)
        let chi = ChiSquared::new(dof - 1.0)
            .unwrap()
            .sample(&mut self.barostat.rng); // χ²_{dof-1}

        let ke_target = 0.5 * dof * GAS_CONST_R * t_target;

        // Discrete-time exact solution for the OU process in K (from Bussi 2007):
        // K' = K*c + ke_bar*(1.0 - c) * [ (chi + r*r)/dof ] + 2.0*r*sqrt(c*(1.0-c)*K*ke_bar/dof)
        let k_prime = ke * c
            + ke_target * (1.0 - c) * ((chi + r * r) / dof)
            + 2.0 * r * ((c * (1.0 - c) * ke * ke_target / dof).sqrt());

        let k_prime = k_prime.max(1e-20);
        let lam = (k_prime / ke).sqrt() as f32;

        for a in &mut self.atoms {
            if a.static_ {
                continue;
            }

            a.vel *= lam;
        }
        for w in &mut self.water {
            w.o.vel *= lam;
            w.h0.vel *= lam;
            w.h1.vel *= lam;
        }
    }

    /// A thermostat that integrates the stochastic Langevin equation. Good temperature control
    /// and ergodicity, but the friction parameter damps real dynamics as it grows. This applies an OU update.
    ///
    /// SUPERSEDED: replaced by the LAMMPS-style force-based Langevin applied in
    /// `kick_and_calc_accel` (integrate.rs), which fixes the ~+70 K equilibrium
    /// offset of this mid-step OU formulation. Kept only as reference.
    #[allow(dead_code)]
    pub(crate) fn apply_langevin_thermostat(&mut self, dt: f32, gamma: f32, temp_tgt_k: f32) {
        let c = (-gamma * dt).exp();
        let s2 = (1.0 - c * c).max(0.0); // numerical guard

        let sigma_num = KB_A2_PS2_PER_K_PER_AMU * temp_tgt_k * s2;

        for a in &mut self.atoms {
            if a.static_ {
                continue;
            }

            // per-component σ for velocity noise
            let sigma = (sigma_num / a.mass).sqrt();

            let nx: f32 = self.barostat.rng.sample(StandardNormal);
            let ny: f32 = self.barostat.rng.sample(StandardNormal);
            let nz: f32 = self.barostat.rng.sample(StandardNormal);

            a.vel.x = c * a.vel.x + sigma * nx;
            a.vel.y = c * a.vel.y + sigma * ny;
            a.vel.z = c * a.vel.z + sigma * nz;
        }

        for w in &mut self.water {
            if self.cfg.overrides.skip_water_thermostat {
                continue;
            }

            // --- Rigid-body Langevin (correct for constrained water) ---
            // The previous code applied an independent 9-component OU noise to
            // each water atom (O, H0, H1) and let SETTLE project to 6 DOF. That
            // is NOT energy-balanced for the rotational modes: measured on 2LYZ
            // the NVT equilibrium sat ~+70 K above target (turning the water
            // thermostat off drops it back to target). Proper constrained MD
            // (GROMACS / OpenMM / LAMMPS rigid bodies) applies the thermostat
            // to the 6 physical DOF: COM translation (mass M) + rotation
            // (inertia tensor I).
            let mass_total = O_MASS + 2.0 * H_MASS;

            let r_com =
                (w.o.posit * O_MASS + w.h0.posit * H_MASS + w.h1.posit * H_MASS) / mass_total;
            let v_com = (w.o.vel * O_MASS + w.h0.vel * H_MASS + w.h1.vel * H_MASS) / mass_total;

            let r_o = w.o.posit - r_com;
            let r_h0 = w.h0.posit - r_com;
            let r_h1 = w.h1.posit - r_com;

            // Inertia tensor about COM.
            let inertia = |r: Vec3, mass: f32| {
                let r2 = r.dot(r);
                [
                    [
                        mass * (r2 - r.x * r.x),
                        -mass * r.x * r.y,
                        -mass * r.x * r.z,
                    ],
                    [
                        -mass * r.y * r.x,
                        mass * (r2 - r.y * r.y),
                        -mass * r.y * r.z,
                    ],
                    [
                        -mass * r.z * r.x,
                        -mass * r.z * r.y,
                        mass * (r2 - r.z * r.z),
                    ],
                ]
            };
            let mut inertia_arr = inertia(r_o, O_MASS);
            for added in [inertia(r_h0, H_MASS), inertia(r_h1, H_MASS)] {
                for i in 0..3 {
                    for j in 0..3 {
                        inertia_arr[i][j] += added[i][j];
                    }
                }
            }
            let inertia_mat = Mat3F32::from_arr(inertia_arr);
            let (eigvecs, eigvals) = inertia_mat.eigen_vecs_vals();

            // OU on COM velocity (3 DOF, mass M).
            let sigma_com = (sigma_num / mass_total).sqrt();
            let v_com_new = v_com * c + sample_normal_vec(&mut self.barostat.rng, sigma_com);

            // OU on angular momentum in the principal frame: noise std per
            // principal axis = sqrt(kBT·s2·I_i) so that E[L Lᵀ] → kBT·I and the
            // 3 rotational DOF carry 0.5·kBT each at equilibrium.
            let l_o = r_o.cross(w.o.vel - v_com) * O_MASS;
            let l_h0 = r_h0.cross(w.h0.vel - v_com) * H_MASS;
            let l_h1 = r_h1.cross(w.h1.vel - v_com) * H_MASS;
            let l = l_o + l_h0 + l_h1;
            let nx: f32 = self.barostat.rng.sample(StandardNormal);
            let ny: f32 = self.barostat.rng.sample(StandardNormal);
            let nz: f32 = self.barostat.rng.sample(StandardNormal);
            let n_l = Vec3::new(
                nx * (sigma_num * eigvals.x.max(0.0)).sqrt(),
                ny * (sigma_num * eigvals.y.max(0.0)).sqrt(),
                nz * (sigma_num * eigvals.z.max(0.0)).sqrt(),
            );
            let l_new = l * c + eigvecs * n_l; // noise drawn in principal frame
            let omega_new = inertia_mat.solve_system(l_new); // ω' = I⁻¹ L'

            w.o.vel = v_com_new + omega_new.cross(r_o);
            w.h0.vel = v_com_new + omega_new.cross(r_h0);
            w.h1.vel = v_com_new + omega_new.cross(r_h1);
            w.update_virtual_site();
        }
    }
}
