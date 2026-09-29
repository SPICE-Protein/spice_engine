//! System construction: prepare a peptide from mmCIF and build the MD state.

use crate::engine::md_core::params::{FfParamSet, prepare_peptide_mmcif};
use crate::engine::md_core::{
    ComputationDevice, FfMolType, HydrogenConstraint, MdConfig, MdState, MolDynamics, SimBoxInit,
};
use bio_files::MmCif;

use crate::engine::SpiceEngine;
use crate::env::EnvParams;
use crate::equilibrate::{EquilConfig, equilibrate};
use crate::forcefield::{ComputationContent, ForceFieldSelection};
use crate::topology::ProteinTopology;

/// Options controlling system construction.
#[derive(Debug, Clone)]
pub struct BuildOptions {
    pub env: EnvParams,
    /// Physical content being prepared; selects the compatible parameter domain.
    pub computation_content: ComputationContent,
    /// Existing force-field family used for this content.
    pub force_field: ForceFieldSelection,
    /// Box padding (Å) around the solute.
    pub box_padding_angstrom: f32,
    /// SPME reciprocal mesh spacing in Å. Smaller values improve reciprocal
    /// resolution at increased cost.
    pub spme_mesh_spacing: f32,
    /// Ewald/ SPME splitting parameter in Å⁻¹.
    pub spme_alpha: f32,
    /// Neighbor-list skin in Å.
    pub neighbor_skin: f32,
    pub hydrogen_constraint: HydrogenConstraint,
    /// Max energy-minimization iterations at init. `None` disables.
    pub relax_iters: Option<usize>,
    /// Energy-minimization convergence tolerance, kcal mol⁻¹ Å⁻¹. Tighten
    /// (e.g. 1.0–5.0) to push away added-H clashes before MD; the dynamics
    /// default (~23.9) is loose and can leave systems that explode early.
    pub energy_minimization_tolerance: f32,
    /// Post-minimization equilibration (positional restraints + NVT ramp).
    /// `None` disables it (fast build, but residual strain can crash MD early).
    pub equil: Option<EquilConfig>,
    /// Reject structures with residues missing any charge-lib sidechain heavy
    /// atom (disordered/truncated crystal sidechains; per-residue check, also
    /// catches "only CB survived"). Default `true`: fail with a clear error
    /// listing each residue's missing atoms. Set `false` to build them with
    /// just the atoms present (warning) — physics is wrong for those residues,
    /// so only use this to explore.
    pub strict_incomplete_residues: bool,
    /// Optional dynamic protonation map resolved from PROPKA/H++
    pub custom_protonation:
        Option<std::collections::HashMap<usize, na_seq::AminoAcidProtenationVariant>>,
    /// User-parameterized cosolutes (urea-style denaturants, osmolytes):
    /// displacement mechanism in the engine, parameter data from the caller.
    pub cosolvents: Vec<crate::engine::md_core::CosolventSpec>,
    /// General electrolytes (v1.3.2 species channel): any charge-balanced
    /// cation/anion pair (KCl today, Na₂SO₄-style when polyatomic species
    /// register), counted per formula unit like the named salt knobs.
    pub salts: Vec<crate::engine::md_core::SaltSpec>,
    /// REPACK IONS on the solvent-reuse path (v1.3.8, mutate-only). Default
    /// `false`: reuse transfers the parent's ion population and a differing
    /// salt request is a warned footgun (v1.3.5 guard). With `true` the
    /// reuse entry instead strips the parent's monatomic ions and re-inserts
    /// the REQUESTED composition (counterions for the child net charge + the
    /// env salt knobs) against the unchanged solvent box, genion-style, then
    /// folds the new sites into the local relaxation shell. This makes salt
    /// scans through mutate legal at reuse cost. Explicit contract: the
    /// stripped sites are not healed back to water and slots are drawn from
    /// the parent's remaining waters, so the box carries a small ACCEPTED
    /// density drift and is NOT bitwise-aligned with a fresh build at the
    /// same state (only the ion COUNTS are; the fresh-build golden is
    /// untouched because this path never runs on it).
    pub repack_ions: bool,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            env: EnvParams::default(),
            computation_content: ComputationContent::Protein,
            force_field: ForceFieldSelection::Amber19,
            box_padding_angstrom: 10.0,
            spme_mesh_spacing: 1.0,
            spme_alpha: 0.26,
            neighbor_skin: 2.0,
            hydrogen_constraint: HydrogenConstraint::default(),
            relax_iters: Some(2_000),
            energy_minimization_tolerance: 2.0,
            // L-BFGS minimization (real forces, no false convergence) already
            // relaxes the structure enough — the restraint+NVT-ramp equilibrate
            // was only needed when the minimizer false-converged and left clash.
            // Disabled while we validate L-BFGS alone; re-enable later if the
            // hot start still causes stochastic blow-ups.
            equil: None,
            // Reject truncated residues by default (honest physics).
            strict_incomplete_residues: true,
            custom_protonation: None,
            cosolvents: Vec::new(),
            salts: Vec::new(),
            repack_ions: false,
        }
    }
}

/// Prepare a peptide `MmCif` and build a `SpiceEngine` (MD state + topology + env).
///
/// The environment (pH, temperature, pressure, ionic strength) is baked into the
/// system at build time: pH sets protonation states, T/P configure thermostat and
/// barostat, and ionic strength adds Na⁺/Cl⁻ salt pairs (via the dynamics fork).
pub fn build_system(
    dev: &ComputationDevice,
    param_set: &FfParamSet,
    protein: MmCif,
    opts: &BuildOptions,
) -> Result<SpiceEngine, String> {
    let mut protein = protein;

    if !opts.force_field.supports_content(opts.computation_content) {
        return Err(format!(
            "force field '{}' does not support computation content {:?}",
            opts.force_field.name(),
            opts.computation_content
        ));
    }
    // Prepare the selected existing force-field adapter at the build boundary.
    // Numerical evaluation remains on the legacy dynamics path in v1.1.
    let _prepared_force_field = opts
        .force_field
        .prepare_for_content(opts.computation_content)
        .map_err(|e| e.to_string())?;

    let ff_map = param_set
        .peptide_ff_q_map
        .as_ref()
        .ok_or("FfParamSet missing peptide ff/q map — was FfParamSet::new_amber used?")?;

    // Assign hydrogens, ff types, partial charges and bonds at the target pH.
    let (bonds, _dihedrals) = prepare_peptide_mmcif(
        &mut protein,
        ff_map,
        opts.env.ph,
        opts.custom_protonation.as_ref(),
        opts.strict_incomplete_residues,
        opts.env.redox_reducing,
    )
    .map_err(|e| e.to_string())?;

    let topology = ProteinTopology::from_prepared(&protein)?;

    let mol = MolDynamics {
        ff_mol_type: FfMolType::Peptide,
        atoms: protein.atoms.clone(),
        bonds,
        ..Default::default()
    };

    let cfg = MdConfig {
        temp_target: opts.env.temp_k,
        barostat_cfg: if opts.env.pressure_bar > 0.0 {
            Some(Default::default())
        } else {
            None
        },
        hydrogen_constraint: opts.hydrogen_constraint,
        sim_box: SimBoxInit::Pad(opts.box_padding_angstrom),
        spme_mesh_spacing: opts.spme_mesh_spacing,
        spme_alpha: opts.spme_alpha,
        neighbor_skin: opts.neighbor_skin,
        max_init_relaxation_iters: opts.relax_iters,
        energy_minimization_tolerance: opts.energy_minimization_tolerance,
        salt_concentration_m: if opts.env.ionic_strength_m > 0.0 {
            Some(opts.env.ionic_strength_m)
        } else {
            None
        },
        efield: opts.env.efield,
        efield_omega: opts.env.efield_omega,
        cosolvents: opts.cosolvents.clone(),
        salts: opts.salts.clone(),
        divalent_salts: {
            let mut v = Vec::new();
            if opts.env.mg_cl2_m > 0.0 {
                v.push((crate::engine::md_core::DivalentSalt::Mg, opts.env.mg_cl2_m));
            }
            if opts.env.ca_cl2_m > 0.0 {
                v.push((crate::engine::md_core::DivalentSalt::Ca, opts.env.ca_cl2_m));
            }
            if opts.env.sr_cl2_m > 0.0 {
                v.push((crate::engine::md_core::DivalentSalt::Sr, opts.env.sr_cl2_m));
            }
            if opts.env.ba_cl2_m > 0.0 {
                v.push((crate::engine::md_core::DivalentSalt::Ba, opts.env.ba_cl2_m));
            }
            v
        },
        // Disable recenter: it translates only the protein (not the water),
        // changing protein-water distances after minimization and spiking the
        // forces (minimizer "converges" then production step 1 sees ~100-1000×
        // larger forces). Our runs are short and the box is padded, so drift is
        // not a concern.
        recenter_sim_box: false,
        ..Default::default()
    };

    let (state, _added_solvent) =
        MdState::new(dev, &cfg, &[mol], param_set).map_err(|e| e.to_string())?;

    let n_ca = topology.ca_indices.len();
    let mut engine = SpiceEngine {
        state,
        topology,
        env: opts.env,
        dev: dev.clone(),
        dt_ps: 0.002,
        computation_content: opts.computation_content,
        force_field: opts.force_field,
        u_history: Default::default(),
        ca_acc: vec![[0.0f64; 3]; n_ca],
        ca_n: 0,
        // Fail-fast trend monitor: opt-in, off at build (golden path unchanged).
        trend: None,
    };

    // Post-minimization equilibration: positional restraints + NVT ramp, so the
    // residual build strain is released before production MD.
    if let Some(eq) = &opts.equil {
        equilibrate(&mut engine, eq).map_err(|e| format!("equilibration failed: {e}"))?;
    }

    Ok(engine)
}

/// Create a clone of a parent engine with a mutated structure, reusing the solvent box,
/// water molecules, and ions of the parent. Unchanged residues ride the parent's
/// RELAXED coordinates (index-aligned copy — the reused box is at the parent state,
/// a raw solute start would detonate the equilibration ramp). Cost routing (v1.3.6):
/// an environment / temperature change on an unchanged structure SKIPS minimization
/// (~0.2 s on 2LYZ); a real single-point mutation runs a 6 Å LOCAL minimization
/// (~1 s); both avoid the ~30 s cold build's solvent packing. This path carries the
/// PARENT's ion population and does NOT re-pack salt — see the guard below.
pub fn build_mutant_by_solvent_reuse(
    parent: &SpiceEngine,
    param_set: &FfParamSet,
    mut_input: &crate::structure::StructureInput,
    opts: &BuildOptions,
) -> Result<SpiceEngine, String> {
    let dev = &parent.dev;

    // Correctness guard: the reuse path transfers the parent's ions and cannot
    // re-pack salt for a different concentration — ions are inserted by
    // DISPLACING (destroying) a water, so lowering the target salt would leave
    // irrecoverable cavities and the box density could not match a fresh build
    // (decided v1.3.5 after reading the pipeline: salt is constant across RL).
    // The RL pipeline always calls this with the parent's own
    // `ionic_default` (only the structure changes), so this never fires there; it
    // exists so a future caller who varies ionic / divalent / salts through
    // mutate is warned loudly instead of silently served the parent's salt.
    {
        let want_ionic = opts.env.ionic_strength_m;
        let have_ionic = parent.state.cfg.salt_concentration_m.unwrap_or(0.0);
        let want_div: f32 =
            opts.env.mg_cl2_m + opts.env.ca_cl2_m + opts.env.sr_cl2_m + opts.env.ba_cl2_m;
        let have_div: f32 = parent
            .state
            .cfg
            .divalent_salts
            .iter()
            .map(|(_, c)| *c)
            .sum();
        // v1.3.8: `repack_ions` turns this footgun into a feature - the guard
        // stays silent because `repack_mobile_ions` below re-packs the
        // requested population against the unchanged solvent box (accepted
        // density drift, NOT bitwise-equal to a fresh build).
        if !opts.repack_ions
            && ((want_ionic - have_ionic).abs() > 1e-6
                || (want_div - have_div).abs() > 1e-6
                || opts.salts.len() != parent.state.cfg.salts.len())
        {
            eprintln!(
                "[solvent_reuse] WARNING: requested ionic/divalent/salt differs from the parent's; \
                 the reuse path KEEPS the parent ion population (salt is NOT re-packed). Vary salt \
                 through a rebuild (Engine.build / domain::scan_stability) or pass repack_ions=True."
            );
        }
    }

    let protein = crate::structure::atoms_to_mmcif(mut_input)?;
    let mut protein = protein;

    let ff_map = param_set
        .peptide_ff_q_map
        .as_ref()
        .ok_or("FfParamSet missing peptide ff/q map")?;

    // Assign hydrogens, ff types, partial charges and bonds at the target pH.
    let (bonds, _dihedrals) = prepare_peptide_mmcif(
        &mut protein,
        ff_map,
        opts.env.ph,
        opts.custom_protonation.as_ref(),
        opts.strict_incomplete_residues,
        opts.env.redox_reducing,
    )
    .map_err(|e| e.to_string())?;

    let topology = ProteinTopology::from_prepared(&protein)?;

    let mol = MolDynamics {
        ff_mol_type: FfMolType::Peptide,
        atoms: protein.atoms.clone(),
        bonds,
        ..Default::default()
    };

    // Prepare the mutant solute atoms using MdState's build logic,
    // but do not pack water/ions!
    let mut cfg = MdConfig {
        temp_target: opts.env.temp_k,
        efield: opts.env.efield,
        efield_omega: opts.env.efield_omega,
        hydrogen_constraint: opts.hydrogen_constraint,
        spme_mesh_spacing: opts.spme_mesh_spacing,
        spme_alpha: opts.spme_alpha,
        neighbor_skin: opts.neighbor_skin,
        sim_box: SimBoxInit::Fixed((parent.state.cell.bounds_low, parent.state.cell.bounds_high)),
        solvent: crate::engine::md_core::Solvent::None,
        max_init_relaxation_iters: None, // No minimization here
        recenter_sim_box: false,
        ..Default::default()
    };
    cfg.overrides.skip_counterion_insertion = true;
    cfg.overrides.skip_water_relaxation = true;

    let (mut_solute_state, _) =
        MdState::new(dev, &cfg, &[mol], param_set).map_err(|e| e.to_string())?;

    // Call the new mutant solvent reuse builder in dynamics!
    let mutated_state = parent.state.build_mutant_by_solvent_reuse(
        dev,
        mut_solute_state.atoms,
        mut_solute_state.adjacency_list,
        mut_solute_state.force_field_params,
    );

    let n_ca = topology.ca_indices.len();
    let mut engine = SpiceEngine {
        state: mutated_state,
        topology,
        env: opts.env,
        dev: dev.clone(),
        dt_ps: parent.dt_ps,
        computation_content: opts.computation_content,
        force_field: opts.force_field,
        u_history: Default::default(),
        ca_acc: vec![[0.0f64; 3]; n_ca],
        ca_n: 0,
        // Fail-fast trend monitor: opt-in, off at build (golden path unchanged).
        trend: None,
    };

    // Identify structurally-changed solute atoms so we relax ONLY around them.
    //
    // The old code keyed relaxation on a ONE-LETTER sequence mismatch, so:
    //   * it MISSED pH changes — (de)protonation adds/removes hydrogens on the
    //     SAME residue letter, leaving the sequence identical → the detector
    //     found nothing and fell through to a full-box GLOBAL minimize_energy;
    //   * and a pure TEMPERATURE change (structure & protonation identical) also
    //     found nothing → paid that same global 2000-iter L-BFGS (measured
    //     17–25 s on 2LYZ, 150–320 s on HSFA2) even though a T change introduces
    //     ZERO new steric clash and needs NO relaxation at all.
    //
    // A residue is "changed" if any signal fires:
    //   (a) its one-letter identity differs (a mutation);
    //   (b) its atom count differs (protonation added/removed H at a new pH, or
    //       a disulfide CYS↔CYX switch), or the residue is present on only one
    //       side (an indel);
    //   (c) the index-aligned copy (below) trips: element mismatch at the same
    //       k (order-divergence guard; see the MOVE NOTE) — never fired in
    //       practice for a parent-derived wt_input.
    //
    // For UNchanged residues we transfer the parent's relaxed coordinates by
    // index (k-th atom of the residue ↔ k-th atom; order is deterministic
    // through prepare, and the caller's wt_input is serialized from the
    // parent's own prepared order). This closes a latent detonation: the child
    // solute otherwise arrives at RAW, un-relaxed positions (`max_init_
    // relaxation_iters: None`) while the reused water/ions sit at the PARENT's
    // relaxed state. Before the env-only skip, the old global minimize masked
    // the mismatch; with the skip, WT→WT reuse detonated in the equilibration
    // ramp (U ~1e8, ~step 224/400) — and real mutants carried the same raw-
    // start hazard on every UNmutated residue (the RL *_failures.csv class).
    //
    // A genuine same-structure environment/temperature-only reuse therefore
    // ends up with `changed_positions` EMPTY *and* the solute at the parent's
    // relaxed coordinates → safe to skip minimization outright (sub-second
    // env/T scans). Changed residues keep their raw coordinates and are
    // relaxed by the localized 6 Å L-BFGS shell; we NEVER fall back to a
    // global minimize on the reuse path.
    let p_res = &parent.topology.residues;
    let m_res = &engine.topology.residues;
    let p_seq: Vec<char> = parent.topology.sequence.chars().collect();
    let m_seq: Vec<char> = engine.topology.sequence.chars().collect();
    let nres = p_res.len().max(m_res.len());
    // Index alignment (the copy's only contract) is guaranteed by the reuse
    // pipeline itself — per-residue atom order is deterministic through
    // prepare_peptide_mmcif and the caller's wt_input is serialized from the
    // parent's own prepared order (the same order-contract the R2 restraint
    // index-shift already relies on). No distance caps: genuine raw→relaxed
    // drift reaches ~2.5–3 Å on surface loops after build relaxation, so any
    // cap low enough to reject a same-element swap would reject reality.
    // Instead: an ELEMENT equality check per aligned pair (catches the
    // realistic order-divergence pathologies; a residue tripping it is routed
    // to the local shell), and a logged max-move for observability.

    let mut changed_positions: Vec<lin_alg::f32::Vec3> = Vec::new();
    let mut copy_atoms = 0usize;
    let mut copy_max_move = 0.0f32;
    let mut changed_residues = 0usize;
    for r in 0..nres {
        let (Some(p), Some(m)) = (p_res.get(r), m_res.get(r)) else {
            // Residue present on only one side → changed (or vanished) tail.
            if let Some(m) = m_res.get(r) {
                for &ai in &m.atom_indices {
                    if ai < engine.state.atoms.len() {
                        changed_positions.push(engine.state.atoms[ai].posit);
                    }
                }
            }
            continue;
        };
        let identity = r >= p_seq.len() || r >= m_seq.len() || p_seq[r] != m_seq[r];
        let mut residue_changed = identity || p.atom_indices.len() != m.atom_indices.len();
        if !residue_changed {
            // Index-aligned coordinate copy, validated pair-by-pair for
            // element agreement (and index bounds) before any transfer.
            let mut ok = true;
            for (&pi, &mi) in p.atom_indices.iter().zip(m.atom_indices.iter()) {
                if pi >= parent.state.atoms.len() || mi >= engine.state.atoms.len() {
                    ok = false;
                    break;
                }
                if parent.state.atoms[pi].element != engine.state.atoms[mi].element {
                    ok = false;
                    break;
                }
            }
            if ok {
                for (&pi, &mi) in p.atom_indices.iter().zip(m.atom_indices.iter()) {
                    let d =
                        (parent.state.atoms[pi].posit - engine.state.atoms[mi].posit).magnitude();
                    copy_max_move = copy_max_move.max(d);
                    copy_atoms += 1;
                    engine.state.atoms[mi].posit = parent.state.atoms[pi].posit;
                }
            } else {
                // Order/divergence pathology (should not happen for a same-
                // structure wt_input): treat the residue as changed so the
                // local shell relaxes whatever the copy left behind.
                residue_changed = true;
            }
        }
        if residue_changed {
            changed_residues += 1;
            // Shell seed = the residue's current (raw) child positions.
            for &ai in &m.atom_indices {
                if ai < engine.state.atoms.len() {
                    changed_positions.push(engine.state.atoms[ai].posit);
                }
            }
        }
    }

    eprintln!(
        "[solvent_reuse] solute coords: copied parent-relaxed onto {copy_atoms} atom(s) \
         (max move {copy_max_move:.2} A); {changed_residues} residue(s) routed to local \
         relaxation (0 = pure env/T reuse, minimization skipped)."
    );

    // Relax the reused box ONLY where the solute actually changed. A non-empty
    // set → localized 6 Å L-BFGS (seconds). An empty set → pure environment /
    // temperature change on an unchanged structure: the reused solvent AND the
    // copied solute are all already at the parent's relaxed state, there is no
    // new clash, so we skip minimization outright (this is what makes an env/T
    // scan through the reuse path ~head+neighbors+PME ≈ sub-second instead of a
    // global-min explosion).
    // OPTIONAL ion repack (v1.3.8): strip the parent's monatomic ions and
    // re-insert the REQUESTED composition against the unchanged solvent box,
    // then fold the new sites into the relaxation shell below.
    if opts.repack_ions {
        repack_mobile_ions(&mut engine.state, opts, &mut changed_positions);
    }

    if let Some(relax_iters) = opts.relax_iters
        && !changed_positions.is_empty()
    {
        engine.state.minimize_local_region(
            dev,
            &changed_positions,
            6.0, // 6.0 Å search radius
            relax_iters,
        );
    }

    Ok(engine)
}

// ---------------------------------------------------------------------------
// v1.3.8 optional ion repack (see `BuildOptions::repack_ions`).
// ---------------------------------------------------------------------------

fn remap_idx1<V: Clone>(
    m: &std::collections::HashMap<usize, V>,
    map: &[usize],
) -> std::collections::HashMap<usize, V> {
    m.iter()
        .filter(|(i, _)| map[**i] != usize::MAX)
        .map(|(i, v)| (map[*i], v.clone()))
        .collect()
}

fn remap_idx2<V: Clone>(
    m: &std::collections::HashMap<(usize, usize), V>,
    map: &[usize],
) -> std::collections::HashMap<(usize, usize), V> {
    m.iter()
        .filter(|((i, j), _)| map[*i] != usize::MAX && map[*j] != usize::MAX)
        .map(|((i, j), v)| ((map[*i], map[*j]), v.clone()))
        .collect()
}

fn remap_idx3<V: Clone>(
    m: &std::collections::HashMap<(usize, usize, usize), V>,
    map: &[usize],
) -> std::collections::HashMap<(usize, usize, usize), V> {
    m.iter()
        .filter(|((i, j, k), _)| {
            map[*i] != usize::MAX && map[*j] != usize::MAX && map[*k] != usize::MAX
        })
        .map(|((i, j, k), v)| ((map[*i], map[*j], map[*k]), v.clone()))
        .collect()
}

fn remap_idx4<V: Clone>(
    m: &std::collections::HashMap<(usize, usize, usize, usize), V>,
    map: &[usize],
) -> std::collections::HashMap<(usize, usize, usize, usize), V> {
    m.iter()
        .filter(|((i, j, k, l), _)| {
            map[*i] != usize::MAX
                && map[*j] != usize::MAX
                && map[*k] != usize::MAX
                && map[*l] != usize::MAX
        })
        .map(|((i, j, k, l), v)| ((map[*i], map[*j], map[*k], map[*l]), v.clone()))
        .collect()
}

fn remap_pairs(
    s: &std::collections::HashSet<(usize, usize)>,
    map: &[usize],
) -> std::collections::HashSet<(usize, usize)> {
    s.iter()
        .filter(|(i, j)| map[*i] != usize::MAX && map[*j] != usize::MAX)
        .map(|(i, j)| (map[*i], map[*j]))
        .collect()
}

/// Strip every MONATOMIC ion from `state`'s tail (cosolvent molecules stay)
/// and re-insert the population requested by `opts`: counterions for the
/// recomputed child net charge, then the NaCl background, divalent salts and
/// generic salts, exactly like `build_system` (same helpers, same counts).
/// New ion sites are appended to `relax_seeds` so the reuse path's 6 Å local
/// shell relaxes them. Ionless targets (pure de-salting) work too.
///
/// Index safety: monatomic ions carry no bonds/angles/restraints, so removal
/// only renumbers tail atoms; every index-keyed structure is rebuilt through
/// the old->new map. The fresh-build path never calls this, so the
/// ion_layout golden is structurally untouched.
///
/// Contract (deliberate, documented to the user): ion COUNTS and effective I
/// match a fresh build to integer rounding, positions do NOT - slots are
/// drawn from the parent's surviving waters and stripped sites are not
/// healed back to water, leaving a small ACCEPTED density drift. Never
/// compare a repacked box bitwise against a fresh build.
fn repack_mobile_ions(
    state: &mut MdState,
    opts: &BuildOptions,
    relax_seeds: &mut Vec<lin_alg::f32::Vec3>,
) {
    use crate::engine::md_core::DivalentSalt;

    const N_A: f64 = 6.022_140_76e23;

    // 1. Targets: mirror build_system's env -> cfg salt mapping.
    state.cfg.salt_concentration_m = if opts.env.ionic_strength_m > 0.0 {
        Some(opts.env.ionic_strength_m)
    } else {
        None
    };
    {
        let e = &opts.env;
        let mut v = Vec::new();
        if e.mg_cl2_m > 0.0 {
            v.push((DivalentSalt::Mg, e.mg_cl2_m));
        }
        if e.ca_cl2_m > 0.0 {
            v.push((DivalentSalt::Ca, e.ca_cl2_m));
        }
        if e.sr_cl2_m > 0.0 {
            v.push((DivalentSalt::Sr, e.sr_cl2_m));
        }
        if e.ba_cl2_m > 0.0 {
            v.push((DivalentSalt::Ba, e.ba_cl2_m));
        }
        state.cfg.divalent_salts = v;
    }
    state.cfg.salts = opts.salts.clone();

    // 2. Strip the parent's monatomic ions from the tail. Same closed
    // registry the ionic-strength accounting uses (generic `salts` species
    // draw their names from it too); a future POLYATOMIC ion (backlog #32)
    // will need molecule-aware removal - the name filter would otherwise
    // leave its sibling sites behind.
    const ION_FF: [&str; 7] = ["Na+", "K+", "Cl-", "Mg2+", "Ca2+", "Sr2+", "Ba2+"];
    let solute_n = state.solute_atom_count;
    let old_len = state.atoms.len();
    let keep: Vec<bool> = state
        .atoms
        .iter()
        .enumerate()
        .map(|(i, a)| i < solute_n || !ION_FF.contains(&a.force_field_type.as_str()))
        .collect();
    let removed: usize = keep.iter().filter(|k| !**k).count();

    if removed > 0 {
        let mut map: Vec<usize> = vec![usize::MAX; old_len];
        let mut prefix: Vec<usize> = vec![0; old_len + 1];
        let mut nxt = 0usize;
        for i in 0..old_len {
            prefix[i] = nxt;
            if keep[i] {
                map[i] = nxt;
                nxt += 1;
            }
        }
        prefix[old_len] = nxt;

        let fp = &mut state.force_field_params;
        fp.mass = remap_idx1(&fp.mass, &map);
        fp.lennard_jones = remap_idx1(&fp.lennard_jones, &map);
        fp.bond_stretching = remap_idx2(&fp.bond_stretching, &map);
        fp.bond_rigid_constraints = remap_idx2(&fp.bond_rigid_constraints, &map);
        fp.bonds_topology = remap_pairs(&fp.bonds_topology, &map);
        fp.angle = remap_idx3(&fp.angle, &map);
        fp.dihedral = remap_idx4(&fp.dihedral, &map);
        fp.improper = remap_idx4(&fp.improper, &map);

        state.distance_restraints = state
            .distance_restraints
            .iter()
            .filter(|r| map[r.atom_0_idx] != usize::MAX && map[r.atom_1_idx] != usize::MAX)
            .map(|r| crate::engine::md_core::DistanceRestraint {
                atom_0_idx: map[r.atom_0_idx],
                atom_1_idx: map[r.atom_1_idx],
                r0: r.r0,
                k: r.k,
            })
            .collect();

        let old_atoms = std::mem::take(&mut state.atoms);
        let old_adj = std::mem::take(&mut state.adjacency_list);
        state.atoms = old_atoms
            .iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, a)| a.clone())
            .collect();
        // mass_accel_factor is rebuilt by refresh_species_tables below (private field).
        state.adjacency_list = (0..old_atoms.len())
            .filter(|&i| keep[i])
            .map(|i| {
                old_adj
                    .get(i)
                    .map(|row| {
                        row.iter()
                            .filter_map(|&j| {
                                if map[j] == usize::MAX {
                                    None
                                } else {
                                    Some(map[j])
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect();

        let old_starts = std::mem::take(&mut state.mol_start_indices);
        let mut new_starts = Vec::with_capacity(old_starts.len());
        for (mi, &s) in old_starts.iter().enumerate() {
            let e = if mi + 1 < old_starts.len() {
                old_starts[mi + 1]
            } else {
                old_len
            };
            if (s..e).any(|i| keep[i]) {
                new_starts.push(prefix[s]);
            }
        }
        state.mol_start_indices = new_starts;
        // Inter-molecule energy matrix geometry is stale after renumbering.
        state.potential_energy_between_mols.clear();
        state.refresh_species_tables();
        state.setup_nonbonded_exclusion_scale_flags();
    }

    // 3. Re-insert the requested population (same order as build_system:
    // counterions, NaCl background, divalent, generic salts).
    let pre_insert = state.atoms.len();
    let vol_l = f64::from(state.cell.volume()) * 1.0e-27;
    let net_q_e = state.net_charge_e();
    let n_neut = net_q_e.round().abs() as usize;
    if n_neut > 0 {
        crate::engine::md_core::add_ions(state, net_q_e, n_neut);
    }
    if let Some(c) = state.cfg.salt_concentration_m.filter(|c| *c > 0.0) {
        let n_pairs = ((f64::from(c)) * vol_l * N_A).round() as usize;
        state.add_salt_ions(n_pairs);
    }
    for (salt, c) in state.cfg.divalent_salts.clone() {
        if c > 0.0 {
            let n_units = ((f64::from(c)) * vol_l * N_A).round() as usize;
            state.add_divalent_salt(salt, n_units);
        }
    }
    for salt in state.cfg.salts.clone() {
        if salt.molarity > 0.0 {
            let n_units = ((f64::from(salt.molarity)) * vol_l * N_A).round() as usize;
            state.add_salt(&salt, n_units);
        }
    }

    // 4. Relaxation seeds + accounting line.
    let inserted = state.atoms.len() - pre_insert;
    relax_seeds.extend(state.atoms[pre_insert..].iter().map(|a| a.posit));
    eprintln!(
        "[repack] stripped {removed} parent ion(s), inserted {inserted} species atom(s) \
         (target: NaCl {} M, I_eff {:.4} M). Density drift ACCEPTED; repacked box is \
         NOT bitwise-comparable to a fresh build.",
        opts.env.ionic_strength_m,
        state.effective_ionic_strength_m(),
    );
}
