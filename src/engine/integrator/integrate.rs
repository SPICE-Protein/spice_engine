//! Contains integration code, including the primary time step.

use std::{
    fmt,
    fmt::{Display, Formatter},
};

use crate::engine::md_core::clock::Mono;
#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use lin_alg::f32::Vec3;
use rand::RngExt;
use rand_distr::StandardNormal;

use crate::engine::md_core::{
    CENTER_SIMBOX_RATIO, COMPUTATION_TIME_RATIO, ComMotionRemoval, ComputationDevice,
    HydrogenConstraint, KCAL_TO_NATIVE, MdState, Solvent,
    barostat::measure_pressure,
    solvent::{
        ACCEL_CONV_WATER_H, ACCEL_CONV_WATER_O, H_MASS, H_O_H_θ, O_H_R, O_MASS,
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

        let start_entire_step = Mono::now();
        self.last_step_neighbor_rebuild = false;
        self.last_step_pme = false;
        let mut start = Mono::now(); // Re-used for different items

        let log_time = self.step_count.is_multiple_of(COMPUTATION_TIME_RATIO);

        let dt_half = 0.5 * dt;

        // Diverged-configuration tripwire: an exploded structure empties the
        // neighbor lists, and continuing would integrate on garbage (or
        // panic in downstream index math). Abort the step instead.
        if self.cpu_pairs.is_empty() {
            eprintln!("UHoh. Pairs count is 0. THis likely means the system blew up. :(");
            return;
        }

        let pressure = match self.cfg.integrator {
            Integrator::LangevinMiddle { gamma } => {
                if log_time {
                    start = Mono::now();
                }

                // The constraint bucket is zeroed per step and carried across
                // the force reset below (SHAKE/RATTLE run in kick_and_drift,
                // i.e. before the reset): correct, just fiddly — the pattern
                // is load-bearing for the LangevinMiddle ordering.
                self.barostat.virial.constraints = 0.;

                let vc_settle1 = self.kick_and_drift(dt_half, dt_half);

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                if log_time {
                    start = Mono::now();
                }

                // Langevin friction + OU noise ride inside
                // kick_and_calc_accel as ONE exact velocity increment per
                // step (v1.3.8: formerly folded into `accel`, which is
                // consumed twice per step — that double-counted γ and the
                // OU variance; the old mid-step OU c=exp(-γdt) formulation
                // this note once guarded is long gone). `gamma` is read
                // again in kick_and_calc_accel from self.cfg.integrator.
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
                    // Constraint solving is stepper work, not glue.
                    self.computation_time.integration_sum += elapsed;
                }

                // We carry SHAKE/RATTLE bucket writes over the reset below.
                let virial_constr = self.barostat.virial.constraints;

                if log_time {
                    start = Mono::now();
                }

                let vc_settle2 = self.drift(dt_half);

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                    start = Mono::now();
                }

                // Barostat teleport goes HERE — after this step's drift, before
                // its force evaluation (GROMACS c-rescale placement): forces,
                // the PME grid and the half-step kicks all then see the CURRENT
                // box instead of a geometry that a teleport is about to move.
                // (Honest attribution from the 2026-09 forced-μ bisecting: the
                // old end-of-step placement was structurally wrong, but moving
                // the teleport here did NOT by itself change the hot-start
                // heating slope — the pump is the external P·dV work of the
                // position teleport while P_inst is huge, which is why
                // `equilibrate()` now runs NVT and sheds that strain BEFORE
                // production NPT starts. See tests/npt_virial_smoke.rs.)
                // The drive term uses the PREVIOUS step's pressure — a
                // one-step lag that is inert at dlnV ≲ 10⁻³.
                if let Some(bc) = &self.cfg.barostat_cfg
                    && !self.solvent_only_sim_at_init
                {
                    let p_prev = self.barostat.last_p_inst_bar;
                    let box_changed = self.barostat.apply_isotropic(
                        dt as f64,
                        p_prev,
                        self.cfg.temp_target as f64,
                        bc,
                        &mut self.cell,
                        &mut self.atoms,
                        &mut self.water,
                    );
                    // Rebuild PME only when the barostat actually changed the
                    // box — and before forces, so SPME sees the new cell.
                    if box_changed {
                        self.regen_pme(dev);
                    }
                }
                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.barostat_sum += elapsed;
                    start = Mono::now();
                }

                // ------- Below: Compute new forces and accelerations.
                self.reset_f_acc_pe_virial();
                self.apply_all_forces(dev, external_force);

                // `apply_all_forces` records its own bonded/nonbonded/ewald
                // buckets; restart the clock so wrapping it here does not also
                // bill that work to the glue buckets below.
                if log_time {
                    start = Mono::now();
                }

                // Applying from our pre-reset calcs. SETTLE ran TWICE this step
                // (the two half-drifts), each impulse already divided by the
                // half-drift dt — so halve to bill exactly one full-step's
                // worth of constraint virial (GROMACS: once per step, /dt).
                self.barostat.virial.constraints = virial_constr + 0.5 * (vc_settle1 + vc_settle2);

                // Molecular virial theorem: use COM-only translational KE for solvent
                // (rotation excluded); pair site-virial needs the SETTLE constraint term.
                let pressure = measure_pressure(
                    self.measure_kinetic_energy_translational(),
                    &self.cell,
                    &self.barostat.virial.to_kcal_mol(),
                );
                self.barostat.last_p_inst_bar = pressure;

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ambient_sum += elapsed;
                    start = Mono::now();
                }

                // Final half-kick (atoms with mass/units conversion). NOTE:
                // Langevin friction/noise are NO LONGER folded into `accel`
                // (they used to be, and the accel is consumed by TWO
                // half-kicks — this step's and next step's kd — which
                // double-counted γ and OU variance per step: γ_eff=2γ and
                // the historic watT +5-8% overshoot). The stochastic step is
                // applied as ONE exact OU velocity update per step inside
                // kick_and_calc_accel, preserving the deterministic
                // half-kick symplectic bookkeeping untouched.
                self.kick_and_calc_accel(dt_half);

                // SOLUTE-GROUP velocity-rescale thermostat (v1.3.8, GROMACS
                // `tc-grps` semantics). Langevin per-site noise can equalize
                // temperature but CANNOT cool one species below another's
                // bath — γ scales friction and noise together, so a cold
                // solute setpoint loses to hot-bath collisions (v8c measured
                // solT 648 K at solute_k 360, γ=2: worse than no coupling).
                // Rescaling is the deterministic feedback that locks the
                // solute's own kinetic temperature to `temp_target_solute`
                // at a 1 ps time constant regardless of the bath.
                if let Some(t_sol) = self.cfg.temp_target_solute {
                    const R_KK: f64 = 0.001_987_204; // kcal/mol/K
                    const TAU_PS: f64 = 1.0;
                    let (mut ke2_s, mut dof_s) = (0.0f64, 0.0f64);
                    for a in self.atoms.iter() {
                        if !a.static_ {
                            ke2_s += (a.mass as f64) * a.vel.magnitude_squared() as f64;
                            dof_s += 3.0;
                        }
                    }
                    if dof_s > 0.0 && ke2_s > 1e-12 {
                        // T = Σ m v² / (dof·R) with Σmv² in NATIVE units
                        // (amu·Å²/ps²) converted to kcal/mol by /418.4 —
                        // omitting that factor (as the first cut did) reads T
                        // 418× high and turns the "thermostat" into a
                        // refrigerator (v8c4: solT pinned ~0.7×setpoint).
                        let t_now = ke2_s / (418.4 * dof_s * R_KK);
                        let lam = (1.0 + (dt as f64 / TAU_PS) * ((t_sol as f64) / t_now - 1.0))
                            .clamp(0.5, 2.0);
                        if (lam - 1.0).abs() > 1e-9 {
                            for a in self.atoms.iter_mut() {
                                if !a.static_ {
                                    a.vel *= lam as f32;
                                }
                            }
                        }
                    }
                }

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                pressure
            }
            Integrator::VerletVelocity { thermostat } => {
                if log_time {
                    start = Mono::now();
                }

                self.barostat.virial.constraints = 0.;

                // One SETTLE per step, over the FULL drift dt: bill it directly.
                let vc_settle = self.kick_and_drift(dt_half, dt);

                // We carry this over the reset.
                let virial_constr = self.barostat.virial.constraints + vc_settle;

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                    start = Mono::now();
                }

                // Barostat teleport between drift and force evaluation — see
                // the LangevinMiddle branch for the ordering rationale.
                if let Some(bc) = &self.cfg.barostat_cfg
                    && !self.solvent_only_sim_at_init
                {
                    let p_prev = self.barostat.last_p_inst_bar;
                    let box_changed = self.barostat.apply_isotropic(
                        dt as f64,
                        p_prev,
                        self.cfg.temp_target as f64,
                        bc,
                        &mut self.cell,
                        &mut self.atoms,
                        &mut self.water,
                    );
                    if box_changed {
                        self.regen_pme(dev); // before forces: SPME needs the new cell
                    }
                }
                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.barostat_sum += elapsed;
                }

                self.reset_f_acc_pe_virial();
                self.apply_all_forces(dev, external_force);

                if log_time {
                    start = Mono::now();
                }

                // Applying from our pre-reset calcs.
                self.barostat.virial.constraints = virial_constr;

                // Molecular virial theorem: COM-only KE for solvent; the single
                // full-step SETTLE impulse is carried in via virial_constr.
                let pressure = measure_pressure(
                    self.measure_kinetic_energy_translational(),
                    &self.cell,
                    &self.barostat.virial.to_kcal_mol(),
                );
                self.barostat.last_p_inst_bar = pressure;

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
                    start = Mono::now();
                }

                self.kick_and_calc_accel(dt_half);

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                }

                if log_time {
                    start = Mono::now();
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

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.thermostat_sum += elapsed;
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
                    start = Mono::now();
                }

                self.barostat.virial.constraints = 0.;

                // Full kick then full drift. One SETTLE over the full dt: bill directly.
                let vc_settle = self.kick_and_drift(dt, dt);

                let virial_constr = self.barostat.virial.constraints + vc_settle;

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
                    start = Mono::now();
                }

                // Barostat teleport before this step's force evaluation — see
                // the LangevinMiddle branch for the ordering rationale. (In
                // Leapfrog an end-of-step teleport was doubly wrong: even the
                // dt=0 accel refresh at the tail would have been stale
                // against the scaled geometry.)
                if let Some(bc) = &self.cfg.barostat_cfg
                    && !self.solvent_only_sim_at_init
                {
                    let p_prev = self.barostat.last_p_inst_bar;
                    let box_changed = self.barostat.apply_isotropic(
                        dt as f64,
                        p_prev,
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
                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.barostat_sum += elapsed;
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
                    start = Mono::now();
                }

                // Compute forces at x(n+1).
                self.reset_f_acc_pe_virial();
                self.apply_all_forces(dev, external_force);

                // Restart after the self-metered force kernels (see above).
                if log_time {
                    start = Mono::now();
                }

                self.barostat.virial.constraints = virial_constr;

                let pressure = measure_pressure(
                    self.measure_kinetic_energy_translational(),
                    &self.cell,
                    &self.barostat.virial.to_kcal_mol(),
                );
                self.barostat.last_p_inst_bar = pressure;

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ambient_sum += elapsed;
                    start = Mono::now();
                }

                // Update accelerations a(n+1) = F(n+1)/m for the next step's kick.
                // Passing dt = 0 recalculates accels without an additional velocity kick.
                self.kick_and_calc_accel(0.);

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.integration_sum += elapsed;
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

        start = Mono::now(); // No ratio for neighbor times.

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
            let start = Mono::now(); // Not sure how else to handle. (Option would work)
            self.handle_snapshots(pressure as f32);

            if log_time {
                let elapsed = start.elapsed().as_micros() as u64;
                self.computation_time.snapshot_sum += elapsed;
            }

            // `total` accumulates on every step (one extra clock read per
            // step) so `other` compares whole-step wall against bucket
            // averages; sampling only every RATIO-th step aliases against
            // the ~9-11-step rebuild cadence and biases `total` upward.
            {
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
        self.last_pressure_bar = pressure;
    }

    /// Half kick and drift for non-solvent and solvent. We call this one or more time
    /// in the various integration approaches. Includes the SETTLE application for solvent,
    /// and SHAKE + RATTLE for hydrogens, if applicable. Updates kinetic energy.
    /// Returns the SETTLE constraint virial (each impulse divided by THIS drift's
    /// dt) so the caller can bill it once per full step — a stepper that settles
    /// twice per half-step dt must halve, or it double-counts (v1.3.8 lesson:
    /// discarding it entirely was the +25-40 kbar water-box pressure bug).
    fn kick_and_drift(&mut self, dt_kick: f32, dt_drift: f32) -> f64 {
        // Half-kick
        for a in &mut self.atoms {
            if a.static_ {
                continue;
            }

            a.vel += a.accel * dt_kick; // kick
            a.posit += a.vel * dt_drift; // drift
        }

        // GROMACS-style settle: SETTLE's constraint impulse contributes a
        // constraint virial (m·r·Δv/dt per bond) that MUST accompany the raw
        // site-force pair virial. Without it, rigid-molecule internal torques
        // (M-site force projections) leak into the pressure as a large
        // fictitious positive virial — the historical +25-40 kbar water-box
        // anomaly. Discarding this value was THE pressure bug. It is RETURNED
        // rather than bucketed here because a step may call this primitive
        // more than once with sub-step dt (LangevinMiddle's two half-drifts);
        // the caller bills exactly one step's worth (÷ full dt, no more).
        let mut settle_virial = 0f64;
        for w in &mut self.water {
            // Kick
            w.o.vel += w.o.accel * dt_kick;
            w.h0.vel += w.h0.accel * dt_kick;
            w.h1.vel += w.h1.accel * dt_kick;

            settle_virial += integrate_rigid_water(w, dt_drift, &self.cell);
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
        settle_virial
    }

    /// Half kick for non-solvent and solvent. We call this one or more time
    /// in the various integration approaches. Updates kinetic energy.
    fn kick_and_calc_accel(&mut self, dt: f32) {
        // LAMMPS-style force-based Langevin (friction + noise as an acceleration
        // term, integrated through the velocity Verlet). The old mid-step OU
        // velocity update was miscalibrated (~+70 K NVT equilibrium offset); the
        // force-based form is the canonical MD Langevin (LAMMPS fix_langevin).
        // Noise variance per component: 2·gamma·kBT/(m·dt_step).
        // Two dt_step_eff values: the water rigid-body kick consumed its
        // accel at FULL dt here (single `vel += accel * dt`), so its FDT
        // variance matches at dt_step_eff = 2·dt (empirically watT lands on
        // target: 500 K setpoint → 525 measured, +5% is physical). The
        // SOLUTE accel is drawn once per step but consumed COHERENTLY by two
        // half-kicks (this function's kick + the next step's kick_and_drift),
        // total σ_v = s·dt — matching σ_v² = 2γkBT·dt/m needs dt_step_eff =
        // dt. Sharing the water value pinned solute T at ~76% of setpoint
        // (v1.3.8 solT-lag; the +24% gap was collision-pumping from the hot
        // bath, so bulk-T readouts looked fine while the folding window
        // never got its nominal temperature).
        let langevin_solute: Option<(f32, f32)> = match self.cfg.integrator {
            Integrator::LangevinMiddle { gamma } => {
                let g = if self.solvent_only_sim_at_init {
                    LANGEVIN_GAMMA_WATER_INIT
                } else {
                    gamma
                };
                Some((g, dt))
            }
            _ => None,
        };
        let langevin: Option<(f32, f32)> = match self.cfg.integrator {
            Integrator::LangevinMiddle { gamma } => {
                let g = if self.solvent_only_sim_at_init {
                    LANGEVIN_GAMMA_WATER_INIT
                } else {
                    gamma
                };
                Some((g, 2.0 * dt))
            }
            _ => None,
        };
        let kbt = KB_A2_PS2_PER_K_PER_AMU * self.cfg.temp_target;
        // Dual-bath (v1.3.8): the SOLUTE per-site noise may ride its own
        // setpoint so the folding window can be reached without waiting on
        // the water heat bath to relax. None = single bath, bit-identical.
        let kbt_solute =
            KB_A2_PS2_PER_K_PER_AMU * self.cfg.temp_target_solute.unwrap_or(self.cfg.temp_target);

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

            a.vel += a.accel * dt;

            // Exact OU increment, ONCE per step (γ and noise are no longer
            // folded into `accel` — that accel is consumed by TWO half-kicks
            // which double-counted both terms: γ_eff=2γ plus ~2× OU variance,
            // the historical watT +5-8% and the v8c4 finding that OU's own
            // equilibrium (~425 K at bath 500) sat far above solute_k 300).
            // v' = v·d + N(0, (kT/m)(1-d²)), d = e^{-γΔ}, Δ = FULL step
            // (this kick dt + next step's kick_and_drift dt).
            if let Some((gamma, _)) = langevin_solute {
                let d = (-gamma * 2.0 * dt).exp();
                let s = (kbt_solute * self.mass_accel_factor[i] / KCAL_TO_NATIVE * (1.0 - d * d))
                    .max(0.0)
                    .sqrt();
                a.vel = a.vel * d
                    + Vec3::new(
                        s * self.barostat.rng.sample::<f32, _>(StandardNormal),
                        s * self.barostat.rng.sample::<f32, _>(StandardNormal),
                        s * self.barostat.rng.sample::<f32, _>(StandardNormal),
                    );
            }
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

            w.o.vel += w.o.accel * dt;
            w.h0.vel += w.h0.accel * dt;
            w.h1.vel += w.h1.accel * dt;

            // Rigid-body Langevin on the water's 6 PHYSICAL DOF as ONE exact
            // OU increment per step (COM translation mass M + rotation
            // inertia I), applied AFTER the deterministic velocity kick:
            // friction and noise used to ride inside `accel`, which the
            // integrator consumes twice per step — γ_eff = 2γ and ~2× OU
            // variance (the historic watT +5-8% overshoot). v' = v·d +
            // N(0, (kT/m)(1-d²)), d = exp(-γΔ), Δ = 2·dt (full step).
            // Per-atom 9-component noise over-injects the 3 SETTLE-dead DOF
            // (2LYZ calibration: per-atom path kept water ~404 K at target
            // 310) — hence COM+rotation, GROMACS/LAMMPS standard.
            if let Some((gamma, _)) = langevin {
                if !self.cfg.overrides.skip_water_thermostat {
                    let m_total = O_MASS + 2.0 * H_MASS;
                    let r_com =
                        (w.o.posit * O_MASS + w.h0.posit * H_MASS + w.h1.posit * H_MASS) / m_total;
                    let r_o = w.o.posit - r_com;
                    let r_h0 = w.h0.posit - r_com;
                    let r_h1 = w.h1.posit - r_com;
                    let v_com =
                        (w.o.vel * O_MASS + w.h0.vel * H_MASS + w.h1.vel * H_MASS) / m_total;
                    let l = r_o.cross(w.o.vel - v_com) * O_MASS
                        + r_h0.cross(w.h0.vel - v_com) * H_MASS
                        + r_h1.cross(w.h1.vel - v_com) * H_MASS;

                    // Rigid water is a planar asymmetric top whose inertia
                    // PRINCIPAL FRAME is known analytically from the canonical
                    // geometry (v1.3.8 SETTLE keeps it exact): u = H–H line,
                    // v = in-plane ⊥ u (bisector), n = plane normal — mirror
                    // symmetry kills all off-diagonals. The per-step 3×3
                    // inertia build + eigen-decomposition + linear solve this
                    // replaces (~200+ flops/molecule, sampled 8ms/step hot on
                    // profiling) reduces to 3 normalizations + 9 dots. The
                    // principal moments are CONSTANTS of the canonical
                    // geometry (perpendicular-axis theorem), not per-step
                    // quantities. OU in this frame is the same stationary
                    // Gaussian measured distribution (isotropic noise under a
                    // rotating orthonormal basis; per-axis scales carry per-axis
                    // moments); bit-level differs from the eigen path, which
                    // was never bit-stable across builds anyway (fresh StdRng).
                    let u_hh = {
                        let v = r_h0 - r_h1;
                        v * (1.0 / v.magnitude().max(1e-6))
                    };
                    let n_pl = {
                        let v = r_h0.cross(r_h1);
                        v * (1.0 / v.magnitude().max(1e-6))
                    };
                    let v_bp = n_pl.cross(u_hh);
                    // Principal moments: z_c = COM offset along the bisector.
                    let alpha = H_O_H_θ * 0.5;
                    let ra = O_H_R * alpha.cos(); // O->H along bisector
                    let rb = O_H_R * alpha.sin(); // O->H across (half H–H)
                    let z_c = ra * (2.0 * H_MASS) / m_total;
                    let i_u = O_MASS * z_c * z_c + 2.0 * H_MASS * (ra - z_c) * (ra - z_c);
                    let i_v = 2.0 * H_MASS * rb * rb;
                    let i_n = i_u + i_v;

                    let d = (-gamma * 2.0 * dt).exp();
                    let var = (1.0 - d * d).max(0.0);
                    let s_com = (kbt / m_total * var).max(0.0).sqrt();
                    let v_com = v_com * d
                        + Vec3::new(
                            s_com * self.barostat.rng.sample::<f32, _>(StandardNormal),
                            s_com * self.barostat.rng.sample::<f32, _>(StandardNormal),
                            s_com * self.barostat.rng.sample::<f32, _>(StandardNormal),
                        );
                    let s_rot = |i_axis: f32| (kbt * i_axis.max(1e-6) * var).max(0.0).sqrt();
                    let lu = l.dot(u_hh) * d
                        + s_rot(i_u) * self.barostat.rng.sample::<f32, _>(StandardNormal);
                    let lv = l.dot(v_bp) * d
                        + s_rot(i_v) * self.barostat.rng.sample::<f32, _>(StandardNormal);
                    let ln = l.dot(n_pl) * d
                        + s_rot(i_n) * self.barostat.rng.sample::<f32, _>(StandardNormal);
                    let w_rot = u_hh * (lu / i_u) + v_bp * (lv / i_v) + n_pl * (ln / i_n);
                    w.o.vel = v_com + w_rot.cross(r_o);
                    w.h0.vel = v_com + w_rot.cross(r_h0);
                    w.h1.vel = v_com + w_rot.cross(r_h1);
                }
            }
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
    /// and SHAKE + RATTLE for hydrogens, if applicable. Returns the SETTLE
    /// constraint virial (see `kick_and_drift` for the billing contract).
    fn drift(&mut self, dt: f32) -> f64 {
        for a in &mut self.atoms {
            if a.static_ {
                continue;
            }
            a.posit += a.vel * dt;
        }

        // Settle constraint virial — see kick_and_drift for why the caller
        // must bill it exactly once per step (the +25-40 kbar water-box fix).
        let mut settle_virial = 0f64;
        for w in &mut self.water {
            settle_virial += integrate_rigid_water(w, dt, &self.cell);
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

        settle_virial
    }

    /// Pressure audit (v1.3.8): apply a fully affine isotropic dilation by
    /// `lam` to every site (solute and solvent alike; intra-water pairs are
    /// excluded from E, so this is the pure inter-molecular dilation), rebuild
    /// PME for the new cell, re-evaluate forces, and return
    /// (potential_energy, total_virial_kcal, pressure_bar). Finite-differencing
    /// the energy across ±lam gives the thermodynamic virial −∂E/∂lnV, which
    /// discriminates "the pair kernel says so" from "the accumulator says so".
    pub(crate) fn debug_rigid_scale_probe(
        &mut self,
        lam: f64,
        dev: &ComputationDevice,
    ) -> (f64, f64, f64) {
        let l = lam as f32;
        let c = self.cell.center();
        for a in &mut self.atoms {
            if !a.static_ {
                a.posit = c + (a.posit - c) * l;
            }
        }
        for w in &mut self.water {
            // Fully affine site dilation. Intramolecular distances change too,
            // but every intra-molecule pair is excluded from the nonbonded
            // energy, so E(λ) is still exactly Σ_inter U(λ r_ij) — the only
            // scaling under which −∂E/∂lnV equals the site-pair virial Σ r·F.
            // (A COM-preserving "rigid" dilation makes inter-site distances
            // non-affine and corrupts the derivative by Σ F·Δ_intra.)
            for p in [
                &mut w.o.posit,
                &mut w.h0.posit,
                &mut w.h1.posit,
                &mut w.m.posit,
            ] {
                *p = c + (*p - c) * l;
            }
            w.update_virtual_site();
        }
        self.cell.scale_isotropic(l);
        // Force a full neighbor/pair rebuild: the cached pair lists carry
        // pre-scaled distances, and without this the probe would differentiate
        // energies evaluated at stale geometry.
        self.neighbors_nb.max_displacement_sq = f32::MAX;
        self.build_neighbors_if_needed(dev);
        self.regen_pme(dev);
        self.reset_f_acc_pe_virial();
        self.apply_all_forces(dev, None);
        let w = self.barostat.virial.to_kcal_mol();
        let p = measure_pressure(self.measure_kinetic_energy_translational(), &self.cell, &w);
        let w_total = w.bonded + w.nonbonded_short_range + w.nonbonded_long_range + w.constraints;
        (self.potential_energy, w_total, p)
    }
}
