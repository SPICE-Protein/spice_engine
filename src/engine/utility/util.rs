//! Misc utility functions.

use std::{
    fmt::{Display, Formatter},
    io,
    io::ErrorKind,
};

use bio_files::{AtomGeneric, BondGeneric};

use crate::engine::md_core::{COMPUTATION_TIME_RATIO, ParamError};

/// Build a list of indices that relate atoms that are connected by covalent bonds.
/// For each outer atom index, the inner values are indices of the atom it's bonded to.
///
/// Note: If you store bonds with atom indices directly, you may wish to build this in a faster
/// way and cache it, vice this serial-number lookup.
pub(crate) fn build_adjacency_list(
    atoms: &[AtomGeneric],
    bonds: &[BondGeneric],
) -> Result<Vec<Vec<usize>>, ParamError> {
    let mut result = vec![Vec::new(); atoms.len()];

    // For each bond, record its atoms as neighbors of each other
    for bond in bonds {
        let mut atom_0 = None;
        let mut atom_1 = None;

        let mut found = false;
        for (i, atom) in atoms.iter().enumerate() {
            if atom.serial_number == bond.atom_0_sn {
                atom_0 = Some(i);
            }
            if atom.serial_number == bond.atom_1_sn {
                atom_1 = Some(i);
            }

            if let (Some(a0), Some(a1)) = (atom_0, atom_1) {
                result[a0].push(a1);
                result[a1].push(a0);

                found = true;
                break;
            }
        }

        if !found {
            return Err(ParamError::new(
                "Invalid bond to atom mapping when building adjacency list.",
            ));
        }
    }

    Ok(result)
}

/// The output of `ComputationTime`, averaged over step count. In μs.
#[derive(Debug)]
pub struct ComputationTime {
    pub step_count: usize,
    pub bonded: u32,
    pub non_bonded_short_range: u32,
    pub ewald_long_range: u32,
    pub neighbor_all: u32,
    pub neighbor_rebuild: u32,
    pub neighbor_rebuild_ratio: f32,
    pub integration: u32,
    pub ambient: u32,
    pub kinetic: u32,
    pub water_settle: u32,
    pub thermostat: u32,
    pub barostat: u32,
    pub snapshots: u32,
    /// Others substracted from `total`. Assumes no overlap. Uses `neighbor_all`, since `neighbor_rebuild`
    /// is part of it.
    pub other: i32,
    pub total: u32,
}

impl Display for ComputationTime {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        writeln!(
            f,
            "Computation time from {} steps, averaged per step, sampled every {COMPUTATION_TIME_RATIO} steps:",
            self.step_count
        )?;

        writeln!(f, "--Bonded: {} μs", self.bonded)?;
        writeln!(
            f,
            "--Non-bonded (short-range): {} μs",
            self.non_bonded_short_range
        )?;
        writeln!(f, "--Non-bonded (long-range): {} μs", self.ewald_long_range)?;

        writeln!(f, "--Neighbor rebuild: {} μs", self.neighbor_rebuild)?;
        writeln!(
            f,
            "--Neighbor rebuild ratio: {:.2}",
            self.neighbor_rebuild_ratio
        )?;

        writeln!(f, "--Integration: {} μs", self.integration)?;
        writeln!(f, "--Baro/Thermo: {} μs", self.ambient)?;
        writeln!(f, "--Other: {} μs", self.other + self.snapshots as i32)?;
        writeln!(f, "--Total: {} μs", self.total)?;

        Ok(())
    }
}

/// We use this to monitor performance, by component. Times are in μs.
/// The buckets are disjoint by construction: `apply_all_forces` records the
/// bonded/nonbonded/ewald sums itself and the stepper restarts its region
/// clocks around it, so no work is billed twice; `total` is the whole-step
/// wall time and `other` is the untagged residual.
#[derive(Debug, Default, Clone)]
pub struct ComputationTimeSums {
    pub bonded_sum: u64,
    pub non_bonded_short_range_sum: u64,
    pub ewald_long_range_sum: u64,
    /// The ratio doesn't apply to neighbors; log each time we do this.
    /// `neighbor_all` includes the per-step check that decides whether to
    /// rebuild; `neighbor_rebuild` counts only actual rebuilds. Both are kept:
    /// the check runs every step, the rebuild spikes, and the pair separates
    /// steady-state cost from tail events.
    pub neighbor_all_sum: u64,
    /// Just the actual rebuild time; not each time.
    pub neighbor_rebuild_sum: u64,
    pub neighbor_rebuild_count: u16,
    pub integration_sum: u64,
    /// Pressure measurement + constraint-virial carry glue. (Thermostat and
    /// barostat work now land in their own named buckets.)
    pub ambient_sum: u64,
    /// Currently unfilled: the KE traversals run inside the integration and
    /// thermostat regions, so their cost is already counted there.
    pub kinetic_sum: u64,
    /// Currently unfilled: SETTLE runs inside `kick_and_drift`'s integration
    /// region for the same reason.
    pub water_settle_sum: u64,
    pub thermostat_sum: u64,
    pub barostat_sum: u64,
    pub snapshot_sum: u64,
    /// Whole-step wall time averaged over every step. If the other values do
    /// not add up to nearly this, untagged work is significant; inspect `other`.
    pub total: u64,
}

impl ComputationTimeSums {
    pub fn time_per_step(&self, num_steps: usize) -> io::Result<ComputationTime> {
        if num_steps == 0 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "No steps to perform the computation on",
            ));
        }

        let c = COMPUTATION_TIME_RATIO as f32 / num_steps as f32;

        let apply = |v| (v as f32 * c) as u32;

        let bonded = apply(self.bonded_sum);
        let non_bonded_short_range = apply(self.non_bonded_short_range_sum);
        let ewald_long_range = apply(self.ewald_long_range_sum);
        let neighbor_all = (self.neighbor_all_sum / num_steps as u64) as u32;
        let neighbor_rebuild = (self.neighbor_rebuild_sum / num_steps as u64) as u32;
        let integration = apply(self.integration_sum);
        let ambient = apply(self.ambient_sum);
        let kinetic = apply(self.kinetic_sum);
        let water_settle = apply(self.water_settle_sum);
        let thermostat = apply(self.thermostat_sum);
        let barostat = apply(self.barostat_sum);
        let snapshots = apply(self.snapshot_sum);
        // `total` is accumulated every step (unlike the ratio-sampled
        // buckets), so it averages by plain division.
        let total = (self.total / num_steps as u64) as u32;

        let other = total as i32
            - (bonded
                + non_bonded_short_range
                + ewald_long_range
                + neighbor_all
                + integration
                + ambient
                + snapshots) as i32;

        Ok(ComputationTime {
            step_count: num_steps,
            bonded,
            non_bonded_short_range,
            ewald_long_range,
            neighbor_all,
            neighbor_rebuild,
            neighbor_rebuild_ratio: self.neighbor_rebuild_count as f32 / num_steps as f32,
            integration,
            ambient,
            kinetic,
            water_settle,
            thermostat,
            barostat,
            snapshots,
            other,
            total,
        })
    }
}
