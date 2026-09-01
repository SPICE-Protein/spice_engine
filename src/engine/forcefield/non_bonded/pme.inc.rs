impl MdState {
    /// Run the appropriate force-computation function to get force on non-solvent atoms, force
    /// on solvent atoms, and virial sum for the barostat. Uses GPU if available.
    ///
    /// Applies Coulomb and Van der Waals (Lennard-Jones) forces on non-solvent atoms, in place.
    /// We use the MD-standard [S]PME approach to handle approximated Coulomb forces. This function
    /// applies forces from non-solvent, and solvent sources.
    pub fn apply_nonbonded_forces(&mut self, dev: &ComputationDevice) {
        let (f_on_non_water, f_on_water, virial, energy, energy_between_mols, alch_dh_dl) =
            match dev {
                ComputationDevice::Cpu => calc_force_cpu_dispatch(
                    &self.nb_pairs,
                    &self.simd_pairs,
                    &self.scalar_pairs,
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

        // todo; not sure. For one mol, we get 1 and 0.
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

        // todo: You can probably consolidate even further. Instead of calling apply_force
        // todo per each category, you can assemble one big set of pairs, and call it once.
        // todo: This has performance and probably code organization benefits. Maybe try
        // todo after you get the intial version working. Will have to add symmetric to pairs.

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

        // todo: Look at water_water
        // todo: In general, your static exclusions will get messed up with this logic.

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

        // todo: Consider just removing the functional parts above, and add to `pairs` directly.
        // Combine pairs into a single set; we compute in one parallel pass.
        let len_added = pairs_std_water.len() + pairs_water_water.len();

        let mut pairs = pairs_std_std;
        pairs.reserve(len_added);

        pairs.append(&mut pairs_std_water);
        pairs.append(&mut pairs_water_water);

        self.nb_pairs = pairs;
        self.simd_pairs.clear();
        self.scalar_pairs.clear();
        let mut candidates = Vec::new();
        // Keep only the scalar tail of SIMD-eligible pairs.  Cloning every
        // eligible pair here wastes rebuild-time allocations; full SIMD
        // batches are represented by NumericSimdPair alone.
        let mut simd_tail = Vec::new();
        for pair in &self.nb_pairs {
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
                let (sigma, epsilon) = self.lj_tables.lookup(&pair.lj_indices);
                candidates.push(NumericSimdPair {
                    tgt: tgt as u32,
                    src: src as u32,
                    sigma,
                    epsilon,
                    charge_product: self.atoms[tgt].partial_charge * self.atoms[src].partial_charge,
                });
            } else {
                self.scalar_pairs.push(pair.clone());
            }
        }
        let full_len = candidates.len() / 8 * 8;
        self.simd_pairs.extend_from_slice(&candidates[..full_len]);
        // Revisit only the short tail, preserving the exact scalar semantics.
        for pair in self.nb_pairs.iter().filter(|pair| {
            matches!(
                (pair.tgt, pair.src),
                (BodyRef::NonWater(_), BodyRef::NonWater(_))
            ) && !pair.scale_14
                && !pair.alch_interaction
                && pair.calc_lj
                && pair.calc_coulomb
                && !pair.water_full
        }) {
            if simd_tail.len() < candidates.len() - full_len {
                simd_tail.push(pair.clone());
            }
        }
        self.scalar_pairs.extend(simd_tail);
        self.simd_pair_count = self.simd_pairs.len();
        self.scalar_pair_count = self.scalar_pairs.len();
    }

    /// We return the values for the case of not running SPME every step; store them for application
    /// in future steps.
    pub(crate) fn handle_spme_recip(&mut self, dev: &ComputationDevice) -> (Vec<Vec3>, f64, f64) {
        let (pos_all, q_all) = self.pack_pme_pos_q();
        let schedule = staged_decoupling_schedule(self.alchemical.lambda);
        let scale = schedule.coulomb_scale as f64;
        let alch_atom_range = self.alchemical_atom_range();

        let (f_recip, e_recip, virial_from_kspace, alch_cross_dh_dl) = match &mut self.pme_recip {
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
    fn pack_pme_pos_q(&self) -> (Vec<Vec3>, Vec<f32>) {
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

