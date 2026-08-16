//! The high-level SPICE MD engine: wraps `dynamics::MdState` with topology + env.

use std::collections::VecDeque;

use dynamics::{ComputationDevice, MdState};
use lin_alg::f32::Vec3;

use crate::env::EnvParams;
use crate::topology::ProteinTopology;

/// Potential energy (kcal/mol) above which the system is treated as blown up.
const CRASH_ENERGY_KCAL: f64 = 1.0e8;

/// Maximum number of potential-energy samples kept for the `m1` variance metric.
const U_HISTORY_CAP: usize = 512;

/// Result of one integration step.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Instantaneous potential energy, kcal/mol (dynamics native units).
    pub u_t_kcal: f64,
    /// Instantaneous potential energy, kJ/mol.
    pub u_t_kj: f64,
    /// Cα coordinates, `[L, 3]` Å, aligned with `topology.residues`.
    pub coords_ca: Vec<[f32; 3]>,
    pub step_count: usize,
    /// Simulation time, ps.
    pub time_ps: f64,
    pub crashed: bool,
    /// Reason of the crash, if crashed.
    pub crash_reason: Option<String>,
}

/// SPICE's MD engine: one protein + solvent system plus its topology and env.
///
/// `Clone` gives each environment point a fully independent copy of a *pristine*
/// (built + minimized, never-run) system — used by the stability-domain scans so
/// that no point carries state over from another point's simulation.
#[derive(Clone)]
pub struct SpiceEngine {
    pub state: MdState,
    pub topology: ProteinTopology,
    pub env: EnvParams,
    pub dev: ComputationDevice,
    pub dt_ps: f32,
    /// Recent potential-energy samples (kcal/mol), used by the `m1` metric.
    pub u_history: VecDeque<f64>,
    /// Running sum of Cα coordinates (Å) for time-averaged pseudo-labels.
    pub(crate) ca_acc: Vec<[f64; 3]>,
    /// Number of frames accumulated into `ca_acc`.
    pub(crate) ca_n: usize,
}

impl SpiceEngine {
    /// Advance one integration step. `external_force` is per-atom (indexed by
    /// `state.atoms` order) — the hook for SAC bias forces.
    pub fn step(&mut self, external_force: Option<Vec<Vec3>>) -> StepResult {
        self.state.step(&self.dev, self.dt_ps, external_force);

        let u_kcal = self.state.potential_energy;
        let crashed = !u_kcal.is_finite() || u_kcal > CRASH_ENERGY_KCAL;

        let mut crash_reason = None;
        if crashed {
            if !u_kcal.is_finite() {
                crash_reason = Some("potential_energy_nan".to_string());
                // Find first atom with non-finite position
                for (i, a) in self.state.atoms.iter().enumerate() {
                    if !a.posit.x.is_finite() || !a.posit.y.is_finite() || !a.posit.z.is_finite() {
                        let res_desc = self.topology.residues.iter()
                            .find(|r| r.atom_indices.contains(&i))
                            .map(|r| format!("{} (seq_id: {})", r.one_letter, r.seq_id))
                            .unwrap_or_else(|| "unknown_residue".to_string());
                        crash_reason = Some(format!("nan_coordinates_at_atom_index_{}_in_residue_{}", i, res_desc));
                        break;
                    }
                }
            } else if u_kcal > CRASH_ENERGY_KCAL {
                crash_reason = Some(format!("potential_energy_spike_exceeded_crash_threshold_{:.2e}_kcal", u_kcal));
            }
        }

        if u_kcal.is_finite() {
            self.u_history.push_back(u_kcal);
            if self.u_history.len() > U_HISTORY_CAP {
                self.u_history.pop_front();
            }
        }

        let coords_ca: Vec<[f32; 3]> = self
            .topology
            .ca_indices
            .iter()
            .map(|&i| {
                let p = self.state.atoms[i].posit;
                [p.x, p.y, p.z]
            })
            .collect();

        // Accumulate time-averaged Cα (pseudo-label source) on finite steps.
        if !crashed && self.ca_acc.len() == coords_ca.len() {
            for (acc, c) in self.ca_acc.iter_mut().zip(&coords_ca) {
                acc[0] += c[0] as f64;
                acc[1] += c[1] as f64;
                acc[2] += c[2] as f64;
            }
            self.ca_n += 1;
        }

        StepResult {
            u_t_kcal: u_kcal,
            u_t_kj: u_kcal * 4.184,
            coords_ca,
            step_count: self.state.step_count,
            time_ps: self.state.time,
            crashed,
            crash_reason,
        }
    }

    /// Live temperature change (environment perturbation, e.g. +ΔT).
    pub fn set_temperature(&mut self, k: f32) {
        self.state.cfg.temp_target = k;
    }

    /// Re-zero all velocities (e.g. before a fresh run) — SOLUTE and SOLVENT.
    /// Water is included so a cold start is actually cold: leaving the rigid
    /// water at the build/solvent-init temperature (~420 K on 2LYZ) means the
    /// equilibrate ramp and production both start far above the target and the
    /// weak production thermostat (gamma=0.5) can't pull them down in a short
    /// window (measured: t_kin sits ~+65-75 K above target for hundreds of steps).
    pub fn reset_velocities(&mut self) {
        for a in &mut self.state.atoms {
            a.vel = Vec3::new_zero();
        }
        for w in &mut self.state.water {
            w.o.vel = Vec3::new_zero();
            w.h0.vel = Vec3::new_zero();
            w.h1.vel = Vec3::new_zero();
        }
    }

    /// Time-averaged Cα coordinates (Å), the pseudo-label source. Falls back
    /// to current coordinates when no steps have been accumulated.
    pub fn time_averaged_ca(&self) -> Vec<[f32; 3]> {
        if self.ca_n == 0 {
            // Fallback: return instantaneous coordinates of current state
            return self
                .topology
                .ca_indices
                .iter()
                .map(|&i| {
                    let p = self.state.atoms[i].posit;
                    [p.x, p.y, p.z]
                })
                .collect();
        }
        let inv = 1.0 / self.ca_n as f64;
        self.ca_acc
            .iter()
            .map(|a| [a[0] as f32 * inv as f32, a[1] as f32 * inv as f32, a[2] as f32 * inv as f32])
            .collect()
    }

    /// Clear the potential-energy history and the pseudo-label accumulator, so
    /// metrics computed afterwards start from a clean (post-equilibration) state.
    pub fn reset_history(&mut self) {
        self.u_history.clear();
        for a in &mut self.ca_acc {
            *a = [0.0f64; 3];
        }
        self.ca_n = 0;
    }

    /// Reset the pseudo-label accumulator (start a fresh averaging window).
    pub fn reset_pseudo_labels(&mut self) {
        let n = self.topology.ca_indices.len();
        self.ca_acc = vec![[0.0f64; 3]; n];
        self.ca_n = 0;
    }

    /// Register a harmonic distance restraint between two atoms (e.g. for AlphaFold 3 ligand/ion coordination).
    pub fn add_distance_restraint(&mut self, atom_0_idx: usize, atom_1_idx: usize, r0: f32, k: f32) {
        self.state.distance_restraints.push(dynamics::DistanceRestraint {
            atom_0_idx,
            atom_1_idx,
            r0,
            k,
        });
    }
}
