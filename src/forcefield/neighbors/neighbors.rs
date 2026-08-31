//! This module contains code for maintaining non-bonded neighbor (Verlet) neighbor lists.
//! This is an optimization to determine which atoms we count in Lennard Jones
//! and short-term Ewald Coulomb interactions.
//!
//! We generally use cell-wrapped distances for solvent, and direct distances for non-solvent.

use std::time::Instant;

use lin_alg::f32::Vec3;

use crate::{ComputationDevice, MdState, barostat::SimBox};

/// By index for fast lookups; separate fields, as these indices are applied differently for non-solvent atoms
/// and solvent.
///
/// These are historically called "Verlet lists", but we're not using that term, as we use "Verlet" to refer
/// to the integrator, which this has nothing to do with. They do have to do with their applicability to
/// non-bonded interactions, so we call them "Non-bonded neighbors".
#[derive(Default, Clone)]
pub struct NeighborsNb {
    /// Symmetric std-std indices.
    pub std_std: Vec<Vec<usize>>,
    /// Outer: standard. Inner: solvent.
    pub std_water: Vec<Vec<usize>>,
    /// Symmetric solvent-solvent indices.
    pub water_water: Vec<Vec<usize>>,
    //
    /// Reference positions used to determine when we rebuild.
    pub atom_posits_last_rebuild: Vec<Vec3>,
    /// We use O as proxy for the rigid solvent, omitting its hydrogens to save computation; they
    /// will always be near.
    pub water_o_posits_last_rebuild: Vec<Vec3>,
    /// Used to determine when to rebuild neighbor lists.
    pub max_displacement_sq: f32,
    /// These values are set up at init.
    pub half_skin_sq: f32,
    pub skin_sq_w_cutoff: f32,
}

impl NeighborsNb {
    pub fn new(skin: f32, cutoff: f32) -> Self {
        Self {
            half_skin_sq: (skin * 0.5).powi(2),
            skin_sq_w_cutoff: (cutoff + skin) * (cutoff + skin),
            ..Default::default()
        }
    }
}

impl MdState {
    pub(crate) fn update_max_displacement_since_rebuild(&mut self) {
        for (i, a) in self.atoms.iter().enumerate() {
            // Static atoms always have 0 displacement.
            if a.static_ {
                continue;
            }

            let dv = a.posit - self.neighbors_nb.atom_posits_last_rebuild[i];

            self.neighbors_nb.max_displacement_sq = self
                .neighbors_nb
                .max_displacement_sq
                .max(dv.magnitude_squared());
        }

        // We only track oxygen position here. The hydrogens affect the need to rebuild as well,
        // but since they're always near their oxygen, we omit them, with a slight accuracy impact.
        for (i, w) in self.water.iter().enumerate() {
            let diff_water_o = self
                .cell
                .min_image(w.o.posit - self.neighbors_nb.water_o_posits_last_rebuild[i]);

            let dist_sq = diff_water_o.magnitude_squared();

            self.neighbors_nb.max_displacement_sq =
                self.neighbors_nb.max_displacement_sq.max(dist_sq);
        }
    }

    fn save_rebuild_posits(&mut self) {
        self.neighbors_nb.atom_posits_last_rebuild.clear();
        self.neighbors_nb
            .atom_posits_last_rebuild
            .extend(self.atoms.iter().map(|a| a.posit));

        self.neighbors_nb.water_o_posits_last_rebuild.clear();

        self.neighbors_nb
            .water_o_posits_last_rebuild
            .extend(self.water.iter().map(|w| self.cell.wrap(w.o.posit)));

        self.neighbors_nb.max_displacement_sq = 0.0;
    }

    /// This rebuilds all neighbor lists.
    pub(crate) fn build_all_neighbors(&mut self, dev: &ComputationDevice) {
        let atom_posits: Vec<_> = self.atoms.iter().map(|a| a.posit).collect();
        let water_posits: Vec<_> = self
            .water
            .iter()
            .map(|m| self.cell.wrap(m.o.posit))
            .collect();

        // Compute a static mask. We use this to prevent building static-static neighbors; this is an
        // optimization, as their displacement will always be 0.
        let is_static: Vec<_> = self.atoms.iter().map(|a| a.static_).collect();

        self.neighbors_nb.std_std = build_neighbors(
            &atom_posits,
            &atom_posits,
            Some(&is_static),
            &self.cell,
            true,
            self.neighbors_nb.skin_sq_w_cutoff,
        );

        self.neighbors_nb.std_water = build_neighbors(
            &atom_posits,
            &water_posits,
            None,
            &self.cell,
            false,
            self.neighbors_nb.skin_sq_w_cutoff,
        );

        self.neighbors_nb.water_water = build_neighbors(
            &water_posits,
            &water_posits,
            None,
            &self.cell,
            true,
            self.neighbors_nb.skin_sq_w_cutoff,
        );

        self.setup_pairs();


        // Refresh refs and reset displacement
        self.save_rebuild_posits();
    }

    /// Call during each step; determines if we need to rebuild neighbors, and if so, do it.
    pub(crate) fn build_neighbors_if_needed(&mut self, dev: &ComputationDevice) {
        if self.neighbors_nb.max_displacement_sq >= self.neighbors_nb.half_skin_sq {
            let start = Instant::now();

            self.build_all_neighbors(dev);
            self.computation_time.neighbor_rebuild_count += 1;

            let elapsed = start.elapsed().as_micros() as u64;
            self.computation_time.neighbor_rebuild_sum += elapsed;
        }
    }
}

/// [Re]build a neighbor list, used for non-bonded interactions. Run this periodically.
/// The static mask both prevents computing distance for re-build here, and prevents
/// running unnecessary non-bonded calculations downstream.
///
/// Result Outer index: target atoms. Inner: Source atoms that are within our cutoff distance.
/// These get converted to pairs for CPU force evaluation.
///
/// Dynamic nodes will include static neighbors, and static nodes will have empty lists.
pub fn build_neighbors(
    posits_outer: &[Vec3],
    posits_inner: &[Vec3],
    // This helps us skip static-static rebuilds. Symmetric only. Indices must match source and tgt posits.
    is_static: Option<&[bool]>,
    cell: &SimBox,
    symmetric: bool,
    skin_sq_w_cutoff: f32,
) -> Vec<Vec<usize>> {
    if is_static.is_some() && !symmetric {
        panic!("Invalid neighbor build config; can't pass static indices if non-symmetric.")
    }
    if symmetric {
        assert_eq!(
            posits_inner.len(),
            posits_outer.len(),
            "symmetric=true requires identical sets"
        );
    }

    // LAMMPS `npair_bin`-style bin-cell construction (O(N) instead of O(N²)).
    //
    // Every atom is bucketed into a 3D grid of cells whose size is >= the
    // neighbor cutoff, so all within-cutoff partners of an atom live in the
    // atom's own cell or one of its 26 neighbors. We then walk that 3×3×3
    // stencil (wrapping at the box faces for PBC) and keep the pairs within
    // `skin_sq_w_cutoff` — no O(N²) all-pairs scan, and — for `symmetric`
    // sets — no half-list/expansion bookkeeping: each (i, j) pair is emitted
    // from both ends, exactly like the old full symmetric expansion.
    //
    // Built serially on purpose: the rebuild is now cheap enough that the
    // per-step rayon dispatch/join overhead would cost more than the work.
    let cutoff = skin_sq_w_cutoff.sqrt();
    let lo = cell.bounds_low;
    let ext = cell.extent;

    // Grid resolution: bin edge >= cutoff ⇒ neighbors stay within the stencil.
    let nx = ((ext.x / cutoff).ceil() as usize).max(1);
    let ny = ((ext.y / cutoff).ceil() as usize).max(1);
    let nz = ((ext.z / cutoff).ceil() as usize).max(1);
    let bx = ext.x / nx as f32;
    let by = ext.y / ny as f32;
    let bz = ext.z / nz as f32;

    let n_inner = posits_inner.len();

    // LAMMPS linked-list bins: binhead[bin] → first atom index, bins[j] → next.
    let n_bins = nx * ny * nz;
    let mut binhead: Vec<isize> = vec![-1; n_bins];
    let mut bins: Vec<isize> = vec![-1; n_inner];

    // Wrapped cell + flat bin index for a position.
    let bin_index = |p: Vec3| -> (usize, usize, usize, usize) {
        let wx = (p.x - lo.x).rem_euclid(ext.x);
        let wy = (p.y - lo.y).rem_euclid(ext.y);
        let wz = (p.z - lo.z).rem_euclid(ext.z);
        let ix = (wx / bx) as usize;
        let iy = (wy / by) as usize;
        let iz = (wz / bz) as usize;
        let ix = ix.min(nx - 1);
        let iy = iy.min(ny - 1);
        let iz = iz.min(nz - 1);
        (ix, iy, iz, (ix * ny + iy) * nz + iz)
    };

    for (j, &p) in posits_inner.iter().enumerate() {
        let (_, _, _, b) = bin_index(p);
        bins[j] = binhead[b];
        binhead[b] = j as isize;
    }

    let outer_len = posits_outer.len();
    (0..outer_len)
        .map(|i_outer| {
            let mut out = Vec::new();
            let pos_outer = posits_outer[i_outer];
            let (ix, iy, iz, _) = bin_index(pos_outer);

            for dz in -1i32..=1 {
                let niz = (iz as i32 + dz).rem_euclid(nz as i32) as usize;
                for dy in -1i32..=1 {
                    let niy = (iy as i32 + dy).rem_euclid(ny as i32) as usize;
                    for dx in -1i32..=1 {
                        let nix = (ix as i32 + dx).rem_euclid(nx as i32) as usize;
                        let nb = (nix * ny + niy) * nz + niz;
                        let mut j = binhead[nb];
                        while j >= 0 {
                            let jj = j as usize;
                            if jj != i_outer {
                                // Skip static-static pairs; downstream also
                                // skips the actual NB force between them.
                                let mut st_st = false;
                                if let Some(st) = is_static
                                    && st[i_outer]
                                    && st[jj]
                                {
                                    st_st = true;
                                }
                                if !st_st {
                                    let d = cell.min_image(pos_outer - posits_inner[jj]);
                                    if d.magnitude_squared() < skin_sq_w_cutoff {
                                        out.push(jj);
                                    }
                                }
                            }
                            j = bins[jj];
                        }
                    }
                }
            }
            out
        })
        .collect()
}
