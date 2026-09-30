use super::*;
use crate::forcefield::neighbors::ClusterPairStream;

impl MdState {
    /// Run the appropriate force-computation function to get force on non-solvent atoms, force
    /// on solvent atoms, and virial sum for the barostat. Uses GPU if available.
    ///
    /// Applies Coulomb and Van der Waals (Lennard-Jones) forces on non-solvent atoms, in place.
    /// We use the MD-standard [S]PME approach to handle approximated Coulomb forces. This function
    /// applies forces from non-solvent, and solvent sources.
    pub fn apply_nonbonded_forces(&mut self, dev: &ComputationDevice) {
        // See `refresh_soa_posits` (neighbors.rs): the x86 fused kernel reads
        // the SoA view of the std positions, so it must be live at every
        // evaluation, not only right after a neighbor rebuild.
        if matches!(dev, ComputationDevice::Cpu) {
            self.refresh_soa_posits();
        }
        let (f_on_non_water, f_on_water, virial, energy, energy_between_mols, alch_dh_dl) =
            match dev {
                ComputationDevice::Cpu => calc_force_cpu_dispatch(
                    &self.cpu_pairs,
                    &self.simd_pairs,
                    &self.scalar_pairs,
                    &self.water_simd_pairs,
                    &self.water_water_simd_pairs,
                    &self.atoms,
                    &self.water,
                    &self.cell,
                    &self.lj_tables,
                    &self.cfg.overrides,
                    &self.mol_start_indices,
                    self.alchemical.lambda,
                    self.cfg.spme_alpha,
                    self.cfg.coulomb_cutoff,
                    self.cfg.lj_cutoff,
                    Some(self.neighbors_nb.soa_x.as_slice()),
                    Some(self.neighbors_nb.soa_y.as_slice()),
                    Some(self.neighbors_nb.soa_z.as_slice()),
                ),
                #[cfg(feature = "cuda")]
                ComputationDevice::Gpu(stream) => {
                    let (f_std, f_wat, virial, energy, energy_between_mols, alch_dh_dl) =
                        force_nonbonded_gpu(
                            stream,
                            self.gpu_kernels.as_ref().unwrap(),
                            &self.nb_pairs,
                            &self.atoms,
                            &self.water,
                            self.cell.extent,
                            self.forces_posits_gpu.as_mut().unwrap(),
                            self.per_neighbor_gpu.as_ref().unwrap(),
                            &self.cfg.overrides,
                            self.alchemical.lambda,
                        );
                    (
                        f_std,
                        f_wat,
                        virial,
                        energy,
                        energy_between_mols,
                        alch_dh_dl,
                    )
                }
            };

        // println!("\nF short-range: {}", f_on_non_water[0]);

        // `.into()` below converts accumulated forces to f32.
        for (i, tgt) in self.atoms.iter_mut().enumerate() {
            let f: Vec3 = f_on_non_water[i].into();
            tgt.force += f;
        }

        for (i, tgt) in self.water.iter_mut().enumerate() {
            let f = f_on_water[i];
            let f_0: Vec3 = f.f_o.into();
            let f_m: Vec3 = f.f_m.into();
            let f_h0: Vec3 = f.f_h0.into();
            let f_h1: Vec3 = f.f_h1.into();

            tgt.o.force += f_0;
            tgt.m.force += f_m;
            tgt.h0.force += f_h0;
            tgt.h1.force += f_h1;
        }

        self.potential_energy += energy;
        self.potential_energy_nonbonded += energy;

        self.barostat.virial.nonbonded_short_range += virial;

        // The two vectors only agree when molecule-energy tracking had the
        // same state at build time and at this accumulation (a one-molecule
        // system illustrates why: tracking on yields n_mol² = 1, off yields
        // an empty vector). The length check keeps a flipped
        // SPICE_DENSE_MOLECULE_ENERGY a safe no-op merge instead of a panic.
        if energy_between_mols.len() == self.potential_energy_between_mols.len() {
            for (i, e) in self.potential_energy_between_mols.iter_mut().enumerate() {
                *e += energy_between_mols[i];
            }
        }

        self.alchemical.dh_dl += alch_dh_dl;
    }

    /// [Re] initialize non-bonded interaction pairs between atoms. Do this whenever we rebuild neighbors.
    /// Build the neighbors set prior to running this.
    pub(crate) fn setup_pairs(&mut self) {
        let atoms = &self.atoms;
        let n_std = self.atoms.len();
        let n_water_mols = self.water.len();
        let atom_to_mol = atom_to_mol_indices(n_std, &self.mol_start_indices);
        let atom_to_mol = atom_to_mol.as_slice();

        let alch_mol_idx = self.alchemical.mol_idx;

        let sites = [WaterSite::O, WaterSite::M, WaterSite::H0, WaterSite::H1];

        // ------ Forces from other dynamic atoms on dynamic ones ------

        // Exclusions and scaling apply to std-std interactions only.
        let exclusions = &self.pairs_excluded_12_13;
        let scaled_set = &self.pairs_14_scaled;

        // Set up pairs ahead of time; conducive to parallel iteration. We skip excluded pairs,
        // and mark scaled ones. These pairs, in symmetric cases (e.g. std-std), only
        let pairs_std_std: Vec<_> = (0..n_std)
            .flat_map(|i_tgt| {
                self.neighbors_nb
                    .std_std_csr
                    .row(i_tgt)
                    .iter()
                    .copied()
                    .map(|j| j as usize)
                    .filter(move |&j| j > i_tgt) // Ensure stable order
                    .filter_map(move |i_src| {
                        if atoms[i_src].bonded_only || atoms[i_tgt].bonded_only {
                            return None;
                        }

                        let key = (i_tgt, i_src);
                        if exclusions.contains(&key) {
                            return None;
                        }
                        let scale_14 = scaled_set.contains(&key);
                        let alch_interaction = alch_mol_idx.is_some_and(|m_alch| {
                            let tgt_is_alch = atom_to_mol[i_tgt] == m_alch;
                            let src_is_alch = atom_to_mol[i_src] == m_alch;
                            tgt_is_alch ^ src_is_alch
                        });

                        Some(NonBondedPair {
                            tgt: BodyRef::NonWater(i_tgt),
                            src: BodyRef::NonWater(i_src),
                            scale_14,
                            lj_indices: LjTableIndices::StdStd(key),
                            calc_lj: true,
                            calc_coulomb: true,
                            symmetric: true,
                            alch_interaction,
                            water_full: false,
                        })
                    })
            })
            .collect();

        // Forces from solvent on non-solvent atoms, and vice-versa
        // Non-alchemical: ONE compact pair per std–water molecule; the dedicated
        // `f_water_std_cpu` kernel computes O-solute LJ + (M,H0,H1)-solute Coulomb
        // in a single call (4× fewer pairs than the old per-site expansion).
        // Alchemical: fall back to the per-site expansion (soft-core needs it).
        let mut pairs_std_water: Vec<_> = if alch_mol_idx.is_none() {
            (0..n_std)
                .flat_map(|i_std| {
                    self.neighbors_nb
                        .std_water_csr
                        .row(i_std)
                        .iter()
                        .copied()
                        .map(|i_water| i_water as usize)
                        .map(move |i_water| NonBondedPair {
                            tgt: BodyRef::NonWater(i_std),
                            src: BodyRef::Water {
                                mol: i_water,
                                site: WaterSite::O,
                            },
                            scale_14: false,
                            lj_indices: LjTableIndices::StdWater(i_std),
                            calc_lj: true,
                            calc_coulomb: true,
                            symmetric: true,
                            alch_interaction: false,
                            water_full: true,
                        })
                })
                .collect()
        } else {
            (0..n_std)
                .flat_map(|i_std| {
                    self.neighbors_nb
                        .std_water_csr
                        .row(i_std)
                        .iter()
                        .copied()
                        .map(|i_water| i_water as usize)
                        .flat_map(move |i_water| {
                            let alch_interaction =
                                alch_mol_idx.is_some_and(|m_alch| atom_to_mol[i_std] == m_alch);
                            sites.into_iter().map(move |site| NonBondedPair {
                                tgt: BodyRef::NonWater(i_std),
                                src: BodyRef::Water { mol: i_water, site },
                                scale_14: false,
                                lj_indices: LjTableIndices::StdWater(i_std),
                                calc_lj: site == WaterSite::O,
                                calc_coulomb: site != WaterSite::O,
                                symmetric: true,
                                alch_interaction,
                                water_full: false,
                            })
                        })
                })
                .collect()
        };

        // ------ Water on solvent ------
        // Non-alchemical: ONE pair per water–water molecule pair;
        // `f_water_water_cpu` computes O-O LJ + all 9 charged-site Coulomb in a
        // single call (~10× fewer pairs than the old per-site expansion).
        let mut pairs_water_water = if alch_mol_idx.is_none() {
            let mut v = Vec::new();
            for i_0 in 0..n_water_mols {
                for &i_1_u32 in self.neighbors_nb.water_water_csr.row(i_0) {
                    let i_1 = i_1_u32 as usize;
                    if i_1 <= i_0 {
                        continue;
                    }
                    v.push(NonBondedPair {
                        tgt: BodyRef::Water {
                            mol: i_0,
                            site: WaterSite::O,
                        },
                        src: BodyRef::Water {
                            mol: i_1,
                            site: WaterSite::O,
                        },
                        scale_14: false,
                        lj_indices: LjTableIndices::WaterWater,
                        calc_lj: true,
                        calc_coulomb: true,
                        symmetric: true,
                        alch_interaction: false,
                        water_full: true,
                    });
                }
            }
            v
        } else {
            let mut v = Vec::new();
            for i_0 in 0..n_water_mols {
                for &i_1_u32 in self.neighbors_nb.water_water_csr.row(i_0) {
                    let i_1 = i_1_u32 as usize;
                    if i_1 <= i_0 {
                        continue;
                    }
                    for &site_0 in &sites {
                        for &site_1 in &sites {
                            let calc_lj = site_0 == WaterSite::O && site_1 == WaterSite::O;
                            let calc_coulomb = site_0 != WaterSite::O && site_1 != WaterSite::O;

                            if !(calc_lj || calc_coulomb) {
                                continue;
                            }

                            v.push(NonBondedPair {
                                tgt: BodyRef::Water {
                                    mol: i_0,
                                    site: site_0,
                                },
                                src: BodyRef::Water {
                                    mol: i_1,
                                    site: site_1,
                                },
                                scale_14: false,
                                lj_indices: LjTableIndices::WaterWater,
                                calc_lj,
                                calc_coulomb,
                                symmetric: true,
                                alch_interaction: false,
                                water_full: false,
                            });
                        }
                    }
                }
            }
            v
        };

        // Note: the per-class builders are kept (not one direct push into
        // `pairs`) because each class has distinct exclusions/alchemical
        // expansion and feeds the exact `len_added` reserve; direct-append
        // would re-implement the retain-gated prune below for each class.
        // Combine pairs into a single set; we compute in one parallel pass.
        let len_added = pairs_std_water.len() + pairs_water_water.len();

        let mut pairs = pairs_std_std;
        // Conservative cluster-level pruning. The neighbour rebuild threshold
        // guarantees each atom moves by less than skin/2 before the next
        // rebuild, so expanded cluster boxes remain valid throughout the
        // current neighbour-list lifetime.
        let cluster_size = 16usize;
        let cluster_count = n_std.div_ceil(cluster_size);
        let cutoff = self.cfg.coulomb_cutoff.max(self.cfg.lj_cutoff);
        let expansion = self.cfg.neighbor_skin * 0.5;
        let mut cluster_min =
            vec![Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY); cluster_count];
        let mut cluster_max =
            vec![Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY); cluster_count];
        let mut cluster_periodic_unsafe = vec![false; cluster_count];
        for (i, atom) in atoms.iter().enumerate() {
            let c = i / cluster_size;
            let p = atom.posit;
            cluster_min[c].x = cluster_min[c].x.min(p.x - expansion);
            cluster_min[c].y = cluster_min[c].y.min(p.y - expansion);
            cluster_min[c].z = cluster_min[c].z.min(p.z - expansion);
            cluster_max[c].x = cluster_max[c].x.max(p.x + expansion);
            cluster_max[c].y = cluster_max[c].y.max(p.y + expansion);
            cluster_max[c].z = cluster_max[c].z.max(p.z + expansion);
            let lo = self.cell.bounds_low;
            let hi = self.cell.bounds_high;
            cluster_periodic_unsafe[c] |= p.x - expansion <= lo.x + cutoff
                || p.y - expansion <= lo.y + cutoff
                || p.z - expansion <= lo.z + cutoff
                || p.x + expansion >= hi.x - cutoff
                || p.y + expansion >= hi.y - cutoff
                || p.z + expansion >= hi.z - cutoff;
        }
        pairs.retain(|pair| {
            let (BodyRef::NonWater(i), BodyRef::NonWater(j)) = (pair.tgt, pair.src) else {
                return true;
            };
            let ci = i / cluster_size;
            let cj = j / cluster_size;
            cluster_periodic_unsafe[ci]
                || cluster_periodic_unsafe[cj]
                || ClusterPairStream::bbox_may_interact(
                    cluster_min[ci],
                    cluster_max[ci],
                    cluster_min[cj],
                    cluster_max[cj],
                    cutoff,
                )
        });

        let n_std_water_pairs = pairs_std_water.len();
        pairs.reserve(len_added);
        // Compile-time architecture gate, with an A/B opt-in so the SoA water
        // data layer can be measured on an architecture whose default is off
        // (see `WATER_SIMD_ACTIVE`), and an env kill-switch for validation.
        // The opt-in must never exceed the architectures whose dispatch
        // actually consumes batches, or the generic fallback would drop pairs.
        let water_simd_enabled = (WATER_SIMD_ACTIVE
            || (cfg!(any(target_arch = "x86_64", target_arch = "aarch64"))
                && std::env::var_os("SPICE_FORCE_WATER_SIMD").is_some_and(|v| v == "1")))
            && std::env::var_os("SPICE_DISABLE_WATER_SIMD").is_none();
        // Measurement-only stream ablation (`SPICE_WATER_SIMD_ONLY=ws|ww`) so
        // the cost split between the water-solute and water-water batches can
        // be read on one build. Gates batch *building* and the matching scalar
        // skips identically, so the partition assert stays exact.
        let water_only = std::env::var("SPICE_WATER_SIMD_ONLY").ok();
        let ww_simd_on = water_simd_enabled && water_only.as_deref().map_or(true, |v| v == "ww");
        let ws_simd_on = water_simd_enabled && water_only.as_deref().map_or(true, |v| v == "ws");
        self.water_simd_pairs.clear();
        self.water_water_simd_pairs.clear();
        // Stream the water-water candidate walk directly into 8-lane
        // batches (the <=7-entry ring tail is simply not batched and stays
        // scalar via the coverage counter) instead of materializing a
        // 640k-entry intermediate vector on every rebuild.
        let mut water_water_candidate_count = 0usize;
        if ww_simd_on {
            if alch_mol_idx.is_none() {
                self.water_water_simd_pairs
                    .reserve(pairs_water_water.len() / 8);
            }
            let mut ring: [(u32, u32); 8] = [(0, 0); 8];
            for a in 0..n_water_mols {
                for &b in self.neighbors_nb.water_water_csr.row(a) {
                    let b = b as usize;
                    if b <= a {
                        continue;
                    }
                    let k = water_water_candidate_count & 7;
                    ring[k] = (a as u32, b as u32);
                    water_water_candidate_count += 1;
                    if k == 7 && alch_mol_idx.is_none() {
                        self.water_water_simd_pairs.push(WaterWaterBatch8 {
                            a: std::array::from_fn(|i| ring[i].0),
                            b: std::array::from_fn(|i| ring[i].1),
                        });
                    }
                }
            }
        }

        pairs.append(&mut pairs_std_water);
        pairs.append(&mut pairs_water_water);

        self.cpu_pairs = pairs.iter().map(CompactNonBondedPair::from).collect();
        self.water_water_candidate_count = water_water_candidate_count;
        self.water_water_scalar_tail_count = 0;
        self.simd_pairs.clear();
        self.scalar_pairs.clear();
        let mut candidates = Vec::new();
        let mut eligible_scalar = Vec::new();
        // Keep only the scalar tail of SIMD-eligible pairs.  Cloning every
        // eligible pair here wastes rebuild-time allocations; full SIMD
        // batches are represented by NumericSimdPair alone.
        let mut simd_tail = Vec::new();
        // Build complete water-water SIMD batches. Incomplete tail pairs stay
        // in the scalar list and remain the reference implementation.
        // `pairs_water_water` was appended in exactly the same iteration order
        // as `water_water_batch_candidates`, so a plain counter — not a hash
        // set — decides coverage per pair. Hashing 600k+ candidates here made
        // every neighbor-list rebuild noticeably slower.
        let water_water_simd_active = ww_simd_on && alch_mol_idx.is_none();
        let water_water_simd_full = if water_water_simd_active {
            water_water_candidate_count / 8 * 8
        } else {
            0
        };
        self.water_water_scalar_tail_count = if water_water_simd_active {
            water_water_candidate_count - water_water_simd_full
        } else {
            0
        };
        let mut water_water_seen = 0usize;
        for pair in &pairs {
            let simd_eligible = matches!(
                (pair.tgt, pair.src),
                (BodyRef::NonWater(_), BodyRef::NonWater(_))
            ) && !pair.scale_14
                && !pair.alch_interaction
                && pair.calc_lj
                && pair.calc_coulomb
                && !pair.water_full;
            if simd_eligible {
                let (BodyRef::NonWater(tgt), BodyRef::NonWater(src)) = (pair.tgt, pair.src) else {
                    unreachable!("eligible SIMD pair must be standard atoms")
                };
                let (sigma, epsilon, c4) = self.lj_tables.lookup(&pair.lj_indices);
                candidates.push(NumericSimdPair {
                    tgt: tgt as u32,
                    src: src as u32,
                    sigma,
                    epsilon,
                    c4,
                    charge_product: self.atoms[tgt].partial_charge * self.atoms[src].partial_charge,
                });
            } else {
                // Pairs covered by complete water SIMD batches must leave the
                // scalar stream exactly when a dispatch consumes those
                // batches (`water_simd_enabled` is gated on
                // `WATER_SIMD_ACTIVE`); the batch tail stays scalar.
                if pair.water_full
                    && matches!(
                        (pair.tgt, pair.src),
                        (BodyRef::NonWater(_), BodyRef::Water { .. })
                    )
                    && alch_mol_idx.is_none()
                    && ws_simd_on
                {
                    continue;
                }
                if pair.water_full
                    && matches!(
                        (pair.tgt, pair.src),
                        (BodyRef::Water { .. }, BodyRef::Water { .. })
                    )
                {
                    // Same creation order as the candidate list: the first
                    // `water_water_simd_full` compact pairs are batch-covered.
                    if water_water_seen < water_water_simd_full {
                        water_water_seen += 1;
                        continue;
                    }
                    water_water_seen += 1;
                }
                self.scalar_pairs.push(CompactNonBondedPair::from(pair));
            }
        }
        {
            // Same streaming for water-solute batches; the ring tail (<=7)
            // re-enters the scalar stream below, exactly as before.
            if ws_simd_on && alch_mol_idx.is_none() {
                self.water_simd_pairs.reserve(n_std_water_pairs / 8);
            }
            let mut pending: Vec<(u32, u32, f32, f32, f32)> = Vec::with_capacity(8);
            for pair in pairs.iter() {
                if !ws_simd_on || !pair.water_full || alch_mol_idx.is_some() {
                    continue;
                }
                let (BodyRef::NonWater(std), BodyRef::Water { mol: water, .. }) =
                    (pair.tgt, pair.src)
                else {
                    continue;
                };
                // water_std table entries carry the solute atom's own one-sided
                // ion–water C4 (0 for every non-ion → bitwise no-op in lj8).
                let (sigma, epsilon, c4) = self.lj_tables.lookup(&pair.lj_indices);
                pending.push((std as u32, water as u32, sigma, epsilon, c4));
                if pending.len() == 8 {
                    self.water_simd_pairs.push(WaterSoluteBatch8 {
                        std: std::array::from_fn(|i| pending[i].0),
                        water: std::array::from_fn(|i| pending[i].1),
                        sigma: std::array::from_fn(|i| pending[i].2),
                        epsilon: std::array::from_fn(|i| pending[i].3),
                        c4: std::array::from_fn(|i| pending[i].4),
                    });
                    pending.clear();
                }
            }
            for &(std, water, sigma, epsilon, _c4) in pending.iter() {
                self.scalar_pairs.push(CompactNonBondedPair {
                    tgt: BodyRef::NonWater(std as usize),
                    src: BodyRef::Water {
                        mol: water as usize,
                        site: WaterSite::O,
                    },
                    scale_14: false,
                    lj_indices: LjTableIndices::StdWater(std as usize),
                    calc_lj: true,
                    calc_coulomb: true,
                    symmetric: true,
                    alch_interaction: false,
                    water_full: true,
                });
                let _ = (sigma, epsilon);
            }
        }
        // Cluster-sort the numeric SIMD stream before batching. This is the
        // production traversal order: lanes from nearby target/source
        // clusters stay together, improving cache locality and making the
        // stream compatible with cluster-level pruning.
        candidates
            .sort_unstable_by_key(|p| (p.tgt as usize / 16, p.src as usize / 16, p.tgt, p.src));
        let cluster_pairs: Vec<(u32, u32)> = candidates.iter().map(|p| (p.tgt, p.src)).collect();
        self.neighbors_nb.std_cluster_stream =
            crate::forcefield::neighbors::ClusterPairStream::from_pairs(&cluster_pairs, 16)
                .expect("fixed non-zero cluster size");
        let full_len = candidates.len() / 8 * 8;
        self.simd_pairs.extend_from_slice(&candidates[..full_len]);
        // Revisit only the short tail, preserving the exact scalar semantics.
        // The tail follows the same sorted order as the SIMD stream.
        for pair in &pairs {
            let eligible = matches!(
                (pair.tgt, pair.src),
                (BodyRef::NonWater(_), BodyRef::NonWater(_))
            ) && !pair.scale_14
                && !pair.alch_interaction
                && pair.calc_lj
                && pair.calc_coulomb
                && !pair.water_full;
            if eligible {
                eligible_scalar.push(CompactNonBondedPair::from(pair));
            }
        }
        eligible_scalar.sort_unstable_by_key(|p| match (p.tgt, p.src) {
            (BodyRef::NonWater(t), BodyRef::NonWater(s)) => (t / 16, s / 16, t, s),
            _ => (usize::MAX, usize::MAX, usize::MAX, usize::MAX),
        });
        simd_tail.extend(
            eligible_scalar
                .into_iter()
                .skip(full_len)
                .take(candidates.len() - full_len),
        );
        self.scalar_pairs.extend(simd_tail);
        self.simd_pair_count = self.simd_pairs.len();
        self.scalar_pair_count = self.scalar_pairs.len();
        // Every pair must land in exactly one stream: SIMD, scalar (incl. both
        // water batch tails), or a complete water SIMD lane. If setup and
        // dispatch ever disagree on `WATER_SIMD_ACTIVE`, or the water-water
        // coverage counter desyncs from candidate order, pairs get dropped or
        // double-counted. Four integer ops per rebuild: a normal assert.
        assert_eq!(
            self.cpu_pairs.len(),
            self.simd_pairs.len()
                + self.scalar_pairs.len()
                + self.water_simd_pairs.len() * 8
                + self.water_water_simd_pairs.len() * 8,
            "SIMD/scalar/water-SIMD streams must partition cpu_pairs exactly once"
        );
    }

    /// We return the values for the case of not running SPME every step; store them for application
    /// in future steps.
    pub(crate) fn handle_spme_recip(&mut self, dev: &ComputationDevice) -> (Vec<Vec3>, f64, f64) {
        let (pos_all, q_all) = self.pack_pme_pos_q();
        let schedule = staged_decoupling_schedule(self.alchemical.lambda);
        let scale = schedule.coulomb_scale as f64;
        let alch_atom_range = self.alchemical_atom_range();

        let (mut f_recip, mut e_recip, mut virial_from_kspace, alch_cross_dh_dl) =
            match &mut self.pme_recip {
                Some(pme_recip) => {
                    let mut eval = |charges: &[f32]| -> (Vec<Vec3>, f64, f64) {
                        match dev {
                            ComputationDevice::Cpu => {
                                let (forces, energy, virial) =
                                    pme_recip.forces_and_virial(&pos_all, charges);
                                (forces, energy as f64, virial)
                            }
                            #[cfg(feature = "cuda")]
                            #[allow(unused)]
                            ComputationDevice::Gpu(stream) => {
                                #[cfg(not(any(feature = "cufft", feature = "vkfft")))]
                                let (f, e) = pme_recip.forces(&pos_all, charges);
                                #[cfg(any(feature = "cufft", feature = "vkfft"))]
                                let (f, e) = pme_recip.forces_gpu(stream, &pos_all, charges);

                                (f, e as f64, 0.0_f64)
                            }
                        }
                    };

                    if let Some((start, end)) = alch_atom_range {
                        let (f_full, e_full, virial_full) = eval(&q_all);

                        let mut q_env = q_all.clone();
                        for q in &mut q_env[start..end] {
                            *q = 0.0;
                        }
                        let (f_env, e_env, virial_env) = eval(&q_env);

                        let mut q_alch = vec![0.0; q_all.len()];
                        q_alch[start..end].copy_from_slice(&q_all[start..end]);
                        let (f_alch, e_alch, virial_alch) = eval(&q_alch);

                        let f_scaled = f_full
                            .iter()
                            .zip(&f_env)
                            .zip(&f_alch)
                            .map(|((f_full, f_env), f_alch)| {
                                let cross = *f_full - *f_env - *f_alch;
                                *f_env + *f_alch + cross * scale as f32
                            })
                            .collect();

                        let cross_energy = e_full - e_env - e_alch;
                        let cross_virial = virial_full - virial_env - virial_alch;
                        let e_scaled = e_env + e_alch + scale * cross_energy;
                        let virial_scaled = virial_env + virial_alch + scale * cross_virial;

                        (
                            f_scaled,
                            e_scaled,
                            virial_scaled,
                            schedule.coulomb_dscale_dlambda as f64 * cross_energy,
                        )
                    } else {
                        let (f, e, virial) = eval(&q_all);
                        (f, e, virial, 0.0)
                    }
                }
                None => {
                    panic!("No PME recip available; not computing SPME recip.");
                }
            };

        // println!("F Recip: {:.6?}", f_recip[0]);

        // ---- PME reciprocal-space molecular (exclusion) correction ----
        // Real-space accumulation skips covalently excluded pairs (1-2/1-3) and
        // every intra-rigid-water site pair, but `pack_pme_pos_q` feeds ALL
        // charges to S(k): the excluded pairs leaked their reciprocal kernel
        // term q_i q_j erf(alpha r)/r (and equal/opposite central pair forces)
        // into the energy/virial. With rigid water distances the leak is a
        // configuration CONSTANT: analytic -133.2 kcal/mol per water (rigid
        // H-H + 2x H-M erf-kernel pairs at alpha=0.26); measured -134.2 on a
        // 106-water probe box whose entire Delta-U(300->800 K) was +1336 kcal
        // - energy that barely responds to temperature. Subtract them
        // analytically (Essmann 1995 correction; the crate's `force_correction`
        // is exactly this pair term - "may be useful for exclusions").
        // The k-space virial leak was ONE contributor to the historic +38 kbar
        // cold-box reading (v1.3.3). v1.3.8 closed the rest (discarded SETTLE
        // virial, the crate's flipped k^2 window sign, TIP3P-geometry template):
        // post-fix P ~ +12 kbar, stable. Water site force deltas are NOT applied
        // forces project to zero net force/torque through the SETTLE fold.
        // Cosolvent species enter `atoms` with bonds -> covered by the same
        // exclusion set. Alchemical runs keep the uncorrected path (TODO).
        {
            let alpha = self.pme_recip.as_ref().map(|p| p.alpha).unwrap_or(0.0);
            if alpha > 0.0 {
                let mut e_corr = 0.0_f64;
                let mut e_sol = 0.0_f64;
                let mut v_sol = 0.0_f64;
                for &(i, j) in &self.pairs_excluded_12_13 {
                    let qi = self.atoms[i].partial_charge;
                    let qj = self.atoms[j].partial_charge;
                    if qi == 0.0 || qj == 0.0 {
                        continue;
                    }
                    let diff = self
                        .cell
                        .min_image(self.atoms[i].posit - self.atoms[j].posit);
                    let r = diff.magnitude();
                    if r < 1e-6 {
                        continue;
                    }
                    e_sol += (qi as f64) * (qj as f64) * libm::erf((alpha * r) as f64) / (r as f64);
                    let f_pair = ewald::force_correction(diff / r, r, qi, qj, alpha);
                    f_recip[i] -= f_pair;
                    f_recip[j] += f_pair;
                    v_sol += diff.dot(f_pair) as f64;
                }
                e_corr += e_sol;
                virial_from_kspace -= v_sol;
                if !self.water.is_empty() {
                    use crate::engine::md_core::solvent::{H_O_H_θ, O_EP_R, O_H_R};
                    let qh = self.water[0].h0.partial_charge as f64;
                    let qm = self.water[0].m.partial_charge as f64;
                    let r_hh = 2.0 * O_H_R as f64 * (H_O_H_θ as f64 / 2.0).sin();
                    let r_hm = ((O_H_R as f64 * O_H_R as f64 + O_EP_R as f64 * O_EP_R as f64)
                        - 2.0 * O_H_R as f64 * O_EP_R as f64 * (H_O_H_θ as f64 / 2.0).cos())
                    .max(0.0)
                    .sqrt();
                    let two_a_sqrt_pi = 2.0 * alpha as f64 / std::f64::consts::PI.sqrt();
                    let a2 = (alpha as f64) * (alpha as f64);
                    let ek = |r: f64| libm::erf(alpha as f64 * r) / r;
                    let vk = |r: f64| {
                        libm::erf(alpha as f64 * r) / r - two_a_sqrt_pi * (-a2 * r * r).exp()
                    };
                    let n = self.water.len() as f64;
                    let e_wat = (qh * qh * ek(r_hh) + 2.0 * qh * qm * ek(r_hm)) * n;
                    let v_wat = (qh * qh * vk(r_hh) + 2.0 * qh * qm * vk(r_hm)) * n;
                    e_corr += e_wat;
                    virial_from_kspace -= v_wat;
                }
                e_recip -= e_corr;
            }
        }

        self.potential_energy += e_recip as f64;
        self.potential_energy_nonbonded += e_recip as f64;
        self.alchemical.dh_dl += alch_cross_dh_dl;

        // Apply forces; virial comes from the analytical k-space formula, not r·F.
        self.unpack_apply_pme_forces(&f_recip);
        let mut virial_lr_recip = virial_from_kspace;

        // 1–4 Coulomb scaling correction (vacuum correction)
        for &(i, j) in &self.pairs_14_scaled {
            let diff = self
                .cell
                .min_image(self.atoms[i].posit - self.atoms[j].posit);

            let r = diff.magnitude();
            if r.abs() < 1e-6 {
                continue;
            }

            let dir = diff / r;

            let qi = self.atoms[i].partial_charge;
            let qj = self.atoms[j].partial_charge;

            // Vacuum Coulomb force (K=1 if charges are Amber-scaled)
            let inv_r = 1.0 / r;
            let inv_r2 = inv_r * inv_r;
            let f_vac = dir * (qi * qj * inv_r2);

            let df = f_vac * (SCALE_COUL_14 - 1.0);

            self.atoms[i].force += df;
            self.atoms[j].force -= df;

            virial_lr_recip += (dir * r).dot(df) as f64; // r·F
        }

        self.barostat.virial.nonbonded_long_range += virial_lr_recip;

        (f_recip, e_recip as f64, virial_lr_recip)
    }

    /// Gather all particles that contribute to PME (non-solvent atoms, solvent sites).
    /// Returns positions wrapped to the primary box, and their charges. We pack (and unpack)
    /// in a predictable way: non-solvent atoms, then solvent, with order as defined below.
    /// `pub(crate)`: the electrostatic-potential probe (`MdState::electrostatic_potential`)
    /// reuses this exact packing, so the analysis and the hot path can never drift apart.
    pub(crate) fn pack_pme_pos_q(&self) -> (Vec<Vec3>, Vec<f32>) {
        let n_std = self.atoms.len();
        let n_wat = self.water.len();

        let mut pos = Vec::with_capacity(n_std + 3 * n_wat);
        let mut q = Vec::with_capacity(pos.capacity());

        // Non-solvent atoms.
        for a in &self.atoms {
            pos.push(self.cell.wrap(a.posit)); // [0,L) per axis
            q.push(a.partial_charge); // already scaled to Amber units
        }

        // Water sites. We omit O, as it has no charge.
        for w in &self.water {
            pos.push(self.cell.wrap(w.m.posit));
            q.push(w.m.partial_charge);

            pos.push(self.cell.wrap(w.h0.posit));
            q.push(w.h0.partial_charge);

            pos.push(self.cell.wrap(w.h1.posit));
            q.push(w.h1.partial_charge);
        }

        (pos, q)
    }

    /// Apply PME reciprocal forces to atoms and water sites. In the same order as pack_pme_pos_q.
    /// Virial is computed analytically in the ewald library (forces_and_virial), not here.
    pub(crate) fn unpack_apply_pme_forces(&mut self, forces: &[Vec3]) {
        let water_start = self.atoms.len();

        for (i, f) in forces.iter().enumerate() {
            if i < water_start {
                self.atoms[i].force += *f;
            } else {
                let i_wat = i - water_start;
                let i_wat_mol = i_wat / 3;
                match i_wat % 3 {
                    0 => self.water[i_wat_mol].m.force += *f,
                    1 => self.water[i_wat_mol].h0.force += *f,
                    _ => self.water[i_wat_mol].h1.force += *f,
                }
            }
        }
    }

    /// Re-initializes the SPME based on sim box dimensions. Run this at init, and whenever you
    /// update the sim box. Sets FFT planner dimensions.
    pub(crate) fn regen_pme(&mut self, dev: &ComputationDevice) {
        let [lx, ly, lz] = self.cell.extent.to_arr();
        let l = (lx, ly, lz);
        let n = get_grid_n(l, self.cfg.spme_mesh_spacing);

        self.pme_recip = Some(match dev {
            ComputationDevice::Cpu => {
                #[cfg(any(feature = "vkfft", feature = "cufft"))]
                let v = PmeRecip::new(None, n, l, self.cfg.spme_alpha);
                #[cfg(not(any(feature = "vkfft", feature = "cufft")))]
                let v = PmeRecip::new(n, l, self.cfg.spme_alpha);

                v
            }
            #[cfg(feature = "cuda")]
            ComputationDevice::Gpu(stream) => {
                #[cfg(any(feature = "vkfft", feature = "cufft"))]
                let v = PmeRecip::new(Some(stream), n, l, self.cfg.spme_alpha);

                #[cfg(not(any(feature = "vkfft", feature = "cufft")))]
                let v = PmeRecip::new(n, l, self.cfg.spme_alpha);

                v
            }
        });
    }
}

// Pressure-audit accessor (v1.3.8 dev): expose the exclusion pair lists to the
// FFI `debug_state_dump` without shifting any of the lines above (anchors).
impl MdState {
    pub(crate) fn pairs_lists(
        &self,
    ) -> (
        &std::collections::HashSet<(usize, usize)>,
        &std::collections::HashSet<(usize, usize)>,
    ) {
        (&self.pairs_excluded_12_13, &self.pairs_14_scaled)
    }
}
