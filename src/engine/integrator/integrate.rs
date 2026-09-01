//! Contains integration code, including the primary time step.

use std::{
    fmt,
    fmt::{Display, Formatter},
    time::Instant,
};

#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use lin_alg::f32::{Mat3 as Mat3F32, Vec3};
use rand::RngExt;
use rand_distr::StandardNormal;

use crate::engine::md_core::{
    CENTER_SIMBOX_RATIO, COMPUTATION_TIME_RATIO, ComMotionRemoval, ComputationDevice,
    HydrogenConstraint, KCAL_TO_NATIVE, MdState, Solvent,
    barostat::measure_pressure,
    solvent::{
        ACCEL_CONV_WATER_H, ACCEL_CONV_WATER_O, H_MASS, O_MASS,
        opc_settle::{RESET_ANGLE_RATIO, integrate_rigid_water, reset_angle},
    },
    thermostat::{
        KB_A2_PS2_PER_K_PER_AMU, LANGEVIN_GAMMA_DEFAULT, LANGEVIN_GAMMA_WATER_INIT,
        TAU_TEMP_WATER_INIT,
    },
};

// The maximum allowed acceleration, in Å/ps^2.
// For example, pathological starting conditions including hydrogen placement.
const MAX_ACCEL: f32 = 1e5;
const MAX_ACCEL_SQ: f32 = MAX_ACCEL * MAX_ACCEL;

// todo: Make this Thermostat instead of Integrator? And have a WIP Integrator with just VV.
#[cfg_attr(feature = "encode", derive(Encode, Decode))]
#[derive(Debug, Clone, PartialEq)]
pub enum Integrator {
    // todo: Thermostat A/R for md integrator.
    /// Similar to GROMACS' `md` integrator.
    Leapfrog { thermostat: Option<f64> },
    /// The inner value is the temperature-coupling time constant if the thermostat is enabled.
    /// This value is in ps.
    /// Lower means more sensitive. 0.1ps is a good default.
    VerletVelocity { thermostat: Option<f64> },
    /// Velocity-verlet with a Langevin thermometer. Good temperature control
    /// and ergodicity, but the friction parameter damps real dynamics as it grows.
    /// γ is friction in 1/ps. Good initial gamma: 1 - 2.0. Default to 2.
    LangevinMiddle { gamma: f32 },
}

impl Default for Integrator {
    fn default() -> Self {
        Self::LangevinMiddle {
            gamma: LANGEVIN_GAMMA_DEFAULT,
        }
    }
}

impl Display for Integrator {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Integrator::Leapfrog { thermostat: _ } => write!(f, "Leap-frog"),
            Integrator::VerletVelocity { thermostat: _ } => write!(f, "Verlet Vel"),
            Integrator::LangevinMiddle { gamma: _ } => write!(f, "Langevin Mid"),
        }
    }
}

impl MdState {
    /// Perform one integration step. This is the entry point for running the simulation.
    /// One step of length `dt` is in picoseconds (10^-12),
    /// with typical values of 0.001, or 0.002ps (1 or 2fs).
    /// This method orchestrates the dynamics at each time step. Uses a Verlet Velocity base,
    /// with different thermostat approaches depending on configuration.
    ///
    /// `External force` allows injection of a specific force into the system. It's indexed by atom.
    pub fn step(&mut self, dev: &ComputationDevice, dt: f32, external_force: Option<Vec<Vec3>>) {
        self.step_with_external_force(dev, dt, external_force.as_deref());
    }

    /// Borrowing variant for RL callers. The force slice is consumed during the
    /// step and is not cloned or retained by the MD state.
    pub fn step_with_external_force(
        &mut self,
        dev: &ComputationDevice,
        dt: f32,
        external_force: Option<&[Vec3]>,
    ) {
        if let Some(f_ext) = external_force
            && f_ext.len() != self.atoms.len()
        {
            eprintln!(
                "Error: External force vector length does not match number of atoms; aborting step."
            );
            return;
        }

        if self.atoms.is_empty() && self.water.is_empty() {
            return;
        }

        let start_entire_step = Instant::now();
        self.last_step_neighbor_rebuild = false;
        self.last_step_pme = false;
        let mut start = Instant::now(); // Re-used for different items

        let log_time = self.step_count.is_multiple_of(COMPUTATION_TIME_RATIO);

        let dt_half = 0.5 * dt;

        // todo: YOu can remove this once we crush the root cause.
        if self.nb_pairs.len() == 0 {
            eprintln!("UHoh. Pairs count is 0. THis likely means the system blew up. :(");
            return;
        }

        let pressure = match self.cfg.integrator {
            Integrator::LangevinMiddle { gamma } => {
                if log_time {
                    start = Instant::now();
                }

                self.barostat.virial.constraints = 0.; // todo: Re-evaluate how you handle this.

                self.kick_and_drift(dt_half, dt_half);

                // We carry this over the reset.
                let virial_constr = self.barostat.virial.constraints;

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                if log_time {
                    start = Instant::now();
                }

                // LAMMPS-style force-based Langevin is applied inside
                // kick_and_calc_accel (friction + noise added to the accel,
                // integrated through the velocity Verlet's two half-kicks), so
                // there is no mid-step velocity OU here. The old mid-step OU
                // (c = exp(-gamma dt)) was miscalibrated: NVT equilibrium sat
                // ~+70 K above target. `gamma` is read again in
                // kick_and_calc_accel from self.cfg.integrator.
                let _ = gamma;
                // `kick_and_drift` already refreshed KE for this exact state;
                // avoid a second full solute + rigid-water traversal here.

                // Rattle after the thermostat run, as it updates velocities in a non-uniform manner.
                if matches!(
                    self.cfg.hydrogen_constraint,
                    HydrogenConstraint::Shake { shake_tolerance: _ }
                        | HydrogenConstraint::Linear { .. }
                ) {
                    self.rattle_hydrogens();
                }

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ambient_sum += elapsed;
                }

                if log_time {
                    start = Instant::now();
                }

                self.drift(dt_half);

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                // ------- Below: Compute new forces and accelerations.
                if log_time {
                    start = Instant::now();
                }

                self.reset_f_acc_pe_virial();
                self.apply_all_forces(dev, external_force);

                // Applying from our pre-reset calcs.
                self.barostat.virial.constraints = virial_constr;

                // Molecular virial theorem: use COM-only KE for solvent (rigid molecules
                // contribute no rotational KE to pressure; no SETTLE constraint virial needed).
                let pressure = measure_pressure(
                    self.measure_kinetic_energy_translational(),
                    &self.cell,
                    &self.barostat.virial.to_kcal_mol(),
                );

                let box_changed = if let Some(bc) = &self.cfg.barostat_cfg {
                    self.barostat.apply_isotropic(
                        dt as f64,
                        pressure,
                        self.cfg.temp_target as f64,
                        bc,
                        &mut self.cell,
                        &mut self.atoms,
                        &mut self.water,
                    )
                } else {
                    false
                };

                // Rebuild PME only when the barostat actually changed the box.
                if box_changed {
                    self.regen_pme(dev);
                }

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ambient_sum += elapsed;
                }

                // Final half-kick (atoms with mass/units conversion)
                if log_time {
                    start = Instant::now();
                }

                self.kick_and_calc_accel(dt_half);

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                pressure
            }
            Integrator::VerletVelocity { thermostat } => {
                if log_time {
                    start = Instant::now();
                }

                self.barostat.virial.constraints = 0.;

                self.kick_and_drift(dt_half, dt);

                // We carry this over the reset.
                let virial_constr = self.barostat.virial.constraints;

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                self.reset_f_acc_pe_virial();
                self.apply_all_forces(dev, external_force);

                if log_time {
                    start = Instant::now();
                }

                // Applying from our pre-reset calcs.
                self.barostat.virial.constraints = virial_constr;

                // Molecular virial theorem: COM-only KE for solvent (no SETTLE constraint virial).
                let pressure = measure_pressure(
                    self.measure_kinetic_energy_translational(),
                    &self.cell,
                    &self.barostat.virial.to_kcal_mol(),
                );

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ambient_sum += elapsed;
                }

                // Forces (bonded and nonbonded, to non-solvent and solvent atoms) have been applied; perform other
                // steps required for integration; second half-kick, RATTLE for hydrogens; SETTLE for solvent. -----

                // Second half-kick using the forces calculated this step, and update accelerations using the atom's mass;
                // Between the accel reset and this step, the accelerations have been missing those factors; this is an optimization to
                // do it once at the end.
                if log_time {
                    start = Instant::now();
                }

                self.kick_and_calc_accel(dt_half);

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                if log_time {
                    start = Instant::now();
                }

                // Note: We don't need to RATTLE hydrogens after applying the CSVR thermostat, because
                // it updates all velocites uniformly.
                if let Some(tau_temp) = thermostat
                    && !self.solvent_only_sim_at_init
                {
                    // Update KE from the current velocities (after both half-kicks) before
                    // passing it to CSVR, which uses self.kinetic_energy internally.  Without
                    // this the cached value from the previous step's CSVR would be used,
                    // making CSVR a near-no-op since it would think KE is already at target.
                    self.kinetic_energy = self.measure_kinetic_energy();
                    self.apply_thermostat_csvr(dt as f64, tau_temp, self.cfg.temp_target as f64);
                    self.kinetic_energy = self.measure_kinetic_energy();
                } else if self.solvent_only_sim_at_init {
                    self.apply_thermostat_csvr(
                        dt as f64,
                        TAU_TEMP_WATER_INIT,
                        self.cfg.temp_target as f64,
                    );
                    self.kinetic_energy = self.measure_kinetic_energy();
                }

                // Barostat runs last in VV: velocities are fully updated and the thermostat has
                // already set the correct KE, so the box/coordinate scaling happens cleanly.
                // Scaled positions feed into the next step's force computation.
                let box_changed = if let Some(bc) = &self.cfg.barostat_cfg {
                    self.barostat.apply_isotropic(
                        dt as f64,
                        pressure,
                        self.cfg.temp_target as f64,
                        bc,
                        &mut self.cell,
                        &mut self.atoms,
                        &mut self.water,
                    )
                } else {
                    false
                };
                // Rebuild PME only when the barostat actually changed the box.
                if box_changed {
                    self.regen_pme(dev);
                }

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ambient_sum += elapsed;
                }
                pressure
            }
            // Leapfrog integration (GROMACS `md` integrator).
            // Velocities live at half-integer steps; positions at integer steps.
            //   v(n+½) = v(n−½) + a(n)·dt   (full kick)
            //   x(n+1) = x(n)   + v(n+½)·dt  (full drift)
            // Constraints are applied after the drift, then forces are computed at x(n+1)
            // so that accelerations are ready for the next step's kick.
            Integrator::Leapfrog { thermostat } => {
                if log_time {
                    start = Instant::now();
                }

                self.barostat.virial.constraints = 0.;

                // Full kick then full drift.
                self.kick_and_drift(dt, dt);

                let virial_constr = self.barostat.virial.constraints;

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                // Optional CSVR thermostat applied to the half-step velocities.
                if let Some(tau_temp) = thermostat
                    && !self.solvent_only_sim_at_init
                {
                    self.kinetic_energy = self.measure_kinetic_energy();
                    self.apply_thermostat_csvr(dt as f64, tau_temp, self.cfg.temp_target as f64);
                    self.kinetic_energy = self.measure_kinetic_energy();
                } else if self.solvent_only_sim_at_init {
                    self.apply_thermostat_csvr(
                        dt as f64,
                        TAU_TEMP_WATER_INIT,
                        self.cfg.temp_target as f64,
                    );
                    self.kinetic_energy = self.measure_kinetic_energy();
                }

                if log_time {
                    start = Instant::now();
                }

                // Compute forces at x(n+1).
                self.reset_f_acc_pe_virial();
                self.apply_all_forces(dev, external_force);

                self.barostat.virial.constraints = virial_constr;

                let pressure = measure_pressure(
                    self.measure_kinetic_energy_translational(),
                    &self.cell,
                    &self.barostat.virial.to_kcal_mol(),
                );

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ambient_sum += elapsed;
                }

                if log_time {
                    start = Instant::now();
                }

                // Update accelerations a(n+1) = F(n+1)/m for the next step's kick.
                // Passing dt = 0 recalculates accels without an additional velocity kick.
                self.kick_and_calc_accel(0.);

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                if let Some(bc) = &self.cfg.barostat_cfg {
                    // todo temp
                    let box_changed = self.barostat.apply_isotropic(
                        dt as f64,
                        pressure,
                        self.cfg.temp_target as f64,
                        bc,
                        &mut self.cell,
                        &mut self.atoms,
                        &mut self.water,
                    );
                    if box_changed {
                        self.regen_pme(dev);
                    }
                }

                pressure
            }
        };

        let next_step_count = self.step_count + 1;

        if self.cfg.zero_com_drift
            && self.cfg.com_removal_interval > 0
            && next_step_count.is_multiple_of(self.cfg.com_removal_interval)
        {
            match self.cfg.com_motion_removal {
                ComMotionRemoval::Linear => self.zero_linear_momentum(),
                ComMotionRemoval::Angular => self.zero_angular_momentum(),
                ComMotionRemoval::LinearAccelerationCorrection => {
                    let interval_dt = dt * self.cfg.com_removal_interval as f32;
                    self.zero_linear_momentum_acceleration_corrected(interval_dt);
                }
                ComMotionRemoval::None => {}
            }
        }

        self.time += dt as f64;
        self.step_count = next_step_count;

        start = Instant::now(); // No ratio for neighbor times.

        self.update_max_displacement_since_rebuild();
        let rebuild_before = self.computation_time.neighbor_rebuild_count;
        self.build_neighbors_if_needed(dev);
        self.last_step_neighbor_rebuild =
            self.computation_time.neighbor_rebuild_count > rebuild_before;

        let elapsed = start.elapsed().as_micros() as u64;
        self.computation_time.neighbor_all_sum += elapsed;

        // We keeping the cell centered on the dynamics atoms. Note that we don't change the dimensions,
        // as these are under management by the barostat.
        if self.cfg.recenter_sim_box && self.step_count.is_multiple_of(CENTER_SIMBOX_RATIO) {
            // Recentering changes only the box origin; PME depends on the box
            // lengths, so its FFT/reciprocal workspace remains valid.
            self.cell.recenter(&self.atoms);
        }

        if self.step_count.is_multiple_of(RESET_ANGLE_RATIO) && self.step_count != 0 {
            for mol in &mut self.water {
                reset_angle(mol, &self.cell);
            }
        }

        if !self.solvent_only_sim_at_init {
            if self.step_count.is_multiple_of(1_000) {
                // self.print_ambient_data(pressure);
            }

            let start = Instant::now(); // Not sure how else to handle. (Option would work)
            self.handle_snapshots(pressure as f32);

            if log_time {
                let elapsed = start.elapsed().as_micros() as u64;
                self.computation_time.snapshot_sum += elapsed;
            }

            if log_time {
                let elapsed = start_entire_step.elapsed().as_micros() as u64;
                self.computation_time.total += elapsed;
            }
        }

        if self.cfg.overrides.snapshots_during_equilibration && self.solvent_only_sim_at_init {
            self.handle_snapshots(pressure as f32);
        }

        // Record the instantaneous kinetic temperature at the end of the step,
        // so applications can verify the thermostat actually reaches the target
        // temperature (a too-weak Langevin coupling would keep T_kin ≈ 298 K
        // even when `temp_target` is 380 K).
        self.last_temperature_k = self.measure_temperature() as f32;
    }

    /// Half kick and drift for non-solvent and solvent. We call this one or more time
    /// in the various integration approaches. Includes the SETTLE application for solvent,
    /// and SHAKE + RATTLE for hydrogens, if applicable. Updates kinetic energy.
    fn kick_and_drift(&mut self, dt_kick: f32, dt_drift: f32) {
        // Half-kick
        for a in &mut self.atoms {
            if a.static_ {
                continue;
            }

            a.vel += a.accel * dt_kick; // kick
            a.posit += a.vel * dt_drift; // drift
        }

        for w in &mut self.water {
            // Kick
            w.o.vel += w.o.accel * dt_kick;
            w.h0.vel += w.h0.accel * dt_kick;
            w.h1.vel += w.h1.accel * dt_kick;

            let _ = integrate_rigid_water(w, dt_drift, &self.cell);
        }

        match self.cfg.hydrogen_constraint {
            HydrogenConstraint::Shake { shake_tolerance } => {
                self.shake_hydrogens(dt_kick, shake_tolerance);
                self.rattle_hydrogens();
            }
            HydrogenConstraint::Linear { order, iter } => {
                self.lincs_hydrogens(dt_kick, order as usize, iter as usize);
                self.rattle_hydrogens();
            }
            HydrogenConstraint::Flexible => {}
        }

        self.kinetic_energy = self.measure_kinetic_energy();
    }

    /// Half kick for non-solvent and solvent. We call this one or more time
    /// in the various integration approaches. Updates kinetic energy.
    fn kick_and_calc_accel(&mut self, dt: f32) {
        // LAMMPS-style force-based Langevin (friction + noise as an acceleration
        // term, integrated through the velocity Verlet). The old mid-step OU
        // velocity update was miscalibrated (~+70 K NVT equilibrium offset); the
        // force-based form is the canonical MD Langevin (LAMMPS fix_langevin).
        // Noise variance per component: 2·gamma·kBT/(m·dt_step).
        let langevin: Option<(f32, f32)> = match self.cfg.integrator {
            Integrator::LangevinMiddle { gamma } => {
                let g = if self.solvent_only_sim_at_init {
                    LANGEVIN_GAMMA_WATER_INIT
                } else {
                    gamma
                };
                Some((g, 2.0 * dt)) // full step = 2·half-kick
            }
            _ => None,
        };
        let kbt = KB_A2_PS2_PER_K_PER_AMU * self.cfg.temp_target;

        // Rate-limit the clamp diagnostics: print once per step with a count,
        // instead of one line per atom — otherwise the log floods when many
        // atoms hit the bound (e.g. during stability scans at non-physiological
        // conditions).
        let mut clamped_count = 0usize;
        let mut clamped_first = 0usize;
        let mut clamped_mag = 0.0f32;

        for (i, a) in self.atoms.iter_mut().enumerate() {
            if a.static_ {
                continue;
            }

            a.accel = a.force * self.mass_accel_factor[i];
            if !(a.accel.x.is_finite() && a.accel.y.is_finite() && a.accel.z.is_finite()) {
                // Non-finite accel (NaN / ±INF): a diverged trajectory has produced a
                // non-finite force (e.g. LJ 1/r^12 overflowing f32 at near-zero
                // separation, or a NaN propagated upstream). The clamp below would turn
                // ±INF into NaN via `to_normalized()` and silently poison the velocities;
                // instead we zero this atom and flag the whole system as blown-up
                // (`potential_energy = NaN`), so the caller reports `crashed` on THIS step
                // rather than one step later (which would burn another integration round on
                // garbage forces).
                a.accel = Vec3::new_zero();
                a.vel = Vec3::new_zero();
                if clamped_count == 0 {
                    clamped_first = i;
                    clamped_mag = f32::INFINITY;
                }
                clamped_count += 1;
                self.potential_energy = f64::NAN;
                continue;
            }
            if a.accel.magnitude_squared() > MAX_ACCEL_SQ {
                if clamped_count == 0 {
                    clamped_first = i;
                    clamped_mag = a.accel.magnitude();
                }
                clamped_count += 1;
                a.accel = a.accel.to_normalized() * MAX_ACCEL;
            }

            // LAMMPS-style Langevin on this atom: accel += -gamma·v + noise.
            if let Some((gamma, dt_step)) = langevin {
                let m_inv = self.mass_accel_factor[i] / KCAL_TO_NATIVE;
                let s = (2.0 * gamma * kbt * m_inv / dt_step).max(0.0).sqrt();
                let nx: f32 = self.barostat.rng.sample(StandardNormal);
                let ny: f32 = self.barostat.rng.sample(StandardNormal);
                let nz: f32 = self.barostat.rng.sample(StandardNormal);
                a.accel += a.vel * (-gamma) + Vec3::new(nx * s, ny * s, nz * s);
            }

            a.vel += a.accel * dt;
        }

        if clamped_count > 0
            && !(self.solvent_only_sim_at_init && self.cfg.solvent == Solvent::OctanolWithWater)
        {
            let why = if clamped_mag.is_infinite() {
                "non-finite accel (NaN/INF)"
            } else {
                "accel clamp"
            };
            println!(
                "Warn: {clamped_count} atom(s) hit {why} on step {}, first atom {clamped_first} ({clamped_mag:.0} -> {MAX_ACCEL:.0})",
                self.step_count
            );
        }

        // Expose the clamp metrics as observables on the state, so callers can
        // detect sustained force spikes (e.g. from thermal kicks at high T) even
        // though the clamp itself keeps the trajectory alive.
        self.last_clamped_count = clamped_count;
        self.last_clamped_mag = clamped_mag;

        for w in &mut self.water {
            // Take the force on M/EP, and instead apply it to the other atoms. This leaves it at 0.
            // w.project_ep_force_to_real_sites(&self.cell);
            w.project_ep_force();

            w.o.accel = w.o.force * ACCEL_CONV_WATER_O;
            w.h0.accel = w.h0.force * ACCEL_CONV_WATER_H;
            w.h1.accel = w.h1.force * ACCEL_CONV_WATER_H;

            if !(w.o.accel.x.is_finite()
                && w.o.accel.y.is_finite()
                && w.o.accel.z.is_finite()
                && w.h0.accel.x.is_finite()
                && w.h0.accel.y.is_finite()
                && w.h0.accel.z.is_finite()
                && w.h1.accel.x.is_finite()
                && w.h1.accel.y.is_finite()
                && w.h1.accel.z.is_finite())
            {
                // Same non-finite guard as the solute loop. Rigid water has no
                // MAX_ACCEL clamp, so this is the only defense against a water
                // site receiving a non-finite kick (which would otherwise fly
                // across the box and destabilize everything around it).
                w.o.accel = Vec3::new_zero();
                w.h0.accel = Vec3::new_zero();
                w.h1.accel = Vec3::new_zero();
                w.o.vel = Vec3::new_zero();
                w.h0.vel = Vec3::new_zero();
                w.h1.vel = Vec3::new_zero();
                self.potential_energy = f64::NAN;
                continue;
            }

            // Rigid-body Langevin on the water's 6 physical DOF (COM translation
            // with total mass M + rigid rotation about COM with inertia I), the
            // LAMMPS/GROMACS standard for rigid solvent. Per-atom 9-component
            // noise over-injects into the 3 SETTLE-constrained DOF: measured on
            // 2LYZ the per-atom path kept WATER ~404 K (target 310) while the
            // solute thermostat sank to ~253 K as the heat sink — the exact
            // over-injection signature. Thermostatting COM + rotation gives each
            // of the 6 physical DOF exactly ½·kBT.
            if let Some((gamma, dt_step)) = langevin {
                if !self.cfg.overrides.skip_water_thermostat {
                    let m_total = O_MASS + 2.0 * H_MASS;
                    let r_com =
                        (w.o.posit * O_MASS + w.h0.posit * H_MASS + w.h1.posit * H_MASS) / m_total;
                    // Current COM velocity and angular momentum about COM.
                    let v_com =
                        (w.o.vel * O_MASS + w.h0.vel * H_MASS + w.h1.vel * H_MASS) / m_total;
                    let r_o = w.o.posit - r_com;
                    let r_h0 = w.h0.posit - r_com;
                    let r_h1 = w.h1.posit - r_com;
                    let v_o = w.o.vel - v_com;
                    let v_h0 = w.h0.vel - v_com;
                    let v_h1 = w.h1.vel - v_com;
                    let l = r_o.cross(v_o) * O_MASS
                        + r_h0.cross(v_h0) * H_MASS
                        + r_h1.cross(v_h1) * H_MASS;

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
                    let mut i_arr = inertia(r_o, O_MASS);
                    for add in [inertia(r_h0, H_MASS), inertia(r_h1, H_MASS)] {
                        for i in 0..3 {
                            for j in 0..3 {
                                i_arr[i][j] += add[i][j];
                            }
                        }
                    }
                    let i_mat = Mat3F32::from_arr(i_arr);
                    let (eigvecs, eigvals) = i_mat.eigen_vecs_vals();

                    // COM Langevin (3 translational DOF, mass M):
                    //   a_com = -γ·V_com + N(0, √(2γ·kBT/(M·dt)))
                    let s_com = (2.0 * gamma * kbt / (m_total * dt_step)).max(0.0).sqrt();
                    let (cx, cy, cz): (f32, f32, f32) = (
                        self.barostat.rng.sample(StandardNormal),
                        self.barostat.rng.sample(StandardNormal),
                        self.barostat.rng.sample(StandardNormal),
                    );
                    let a_com = Vec3::new(cx * s_com, cy * s_com, cz * s_com) - v_com * gamma;

                    // Rotational Langevin on angular momentum, principal frame
                    // (3 rotational DOF): dL/dt = -γ·L + noise, noise std per
                    // principal axis √(2γ·kBT·I_i/dt); rotate back to lab with
                    // the eigenvector matrix; ω-accel = I⁻¹·(dL/dt).
                    let (nx, ny, nz): (f32, f32, f32) = (
                        self.barostat.rng.sample(StandardNormal),
                        self.barostat.rng.sample(StandardNormal),
                        self.barostat.rng.sample(StandardNormal),
                    );
                    let noise_p = Vec3::new(
                        nx * (2.0 * gamma * kbt * eigvals.x.max(1e-6) / dt_step)
                            .max(0.0)
                            .sqrt(),
                        ny * (2.0 * gamma * kbt * eigvals.y.max(1e-6) / dt_step)
                            .max(0.0)
                            .sqrt(),
                        nz * (2.0 * gamma * kbt * eigvals.z.max(1e-6) / dt_step)
                            .max(0.0)
                            .sqrt(),
                    );
                    let d_l = l * (-gamma) + eigvecs * noise_p;
                    let a_rot = i_mat.solve_system(d_l);

                    // Per-atom accel = a_com + a_rot × r_i (adds to force accel).
                    w.o.accel += a_com + a_rot.cross(r_o);
                    w.h0.accel += a_com + a_rot.cross(r_h0);
                    w.h1.accel += a_com + a_rot.cross(r_h1);
                }
            }

            w.o.vel += w.o.accel * dt;
            w.h0.vel += w.h0.accel * dt;
            w.h1.vel += w.h1.accel * dt;
        }

        if matches!(
            self.cfg.hydrogen_constraint,
            HydrogenConstraint::Shake { shake_tolerance: _ } | HydrogenConstraint::Linear { .. }
        ) {
            self.rattle_hydrogens();
        }

        self.kinetic_energy = self.measure_kinetic_energy();
    }

    /// Drifts all non-static atoms in the system.  Includes the SETTLE application for solvent,
    /// and SHAKE + RATTLE for hydrogens, if applicable.
    fn drift(&mut self, dt: f32) {
        for a in &mut self.atoms {
            if a.static_ {
                continue;
            }
            a.posit += a.vel * dt;
        }

        for w in &mut self.water {
            let _ = integrate_rigid_water(w, dt, &self.cell);
        }

        match self.cfg.hydrogen_constraint {
            HydrogenConstraint::Shake { shake_tolerance } => {
                self.shake_hydrogens(dt, shake_tolerance);
            }
            HydrogenConstraint::Linear { order, iter } => {
                self.lincs_hydrogens(dt, order as usize, iter as usize);
            }
            HydrogenConstraint::Flexible => {}
        }
    }
}
