//! System construction: prepare a peptide from mmCIF and build the MD state.

use bio_files::MmCif;
use dynamics::params::{FfParamSet, prepare_peptide_mmcif};
use dynamics::{
    ComputationDevice, FfMolType, HydrogenConstraint, MdConfig, MdState, MolDynamics, SimBoxInit,
};

use crate::engine::SpiceEngine;
use crate::env::EnvParams;
use crate::equilibrate::{EquilConfig, equilibrate};
use crate::topology::ProteinTopology;

/// Options controlling system construction.
#[derive(Debug, Clone)]
pub struct BuildOptions {
    pub env: EnvParams,
    /// Box padding (Å) around the solute.
    pub box_padding_angstrom: f32,
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
    /// Reject structures with residues missing their sidechain heavy atoms
    /// (disordered crystal sidechains). Default `true`: fail with a? clear error
    /// listing the incomplete residues. Set `false` to build them truncated
    /// (backbone + Cα H only, with a warning) — physics is wrong for those
    /// residues, so only use this to explore.
    pub strict_incomplete_residues: bool,
    /// Optional dynamic protonation map resolved from PROPKA/H++
    pub custom_protonation: Option<std::collections::HashMap<usize, na_seq::AminoAcidProtenationVariant>>,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            env: EnvParams::default(),
            box_padding_angstrom: 10.0,
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
        max_init_relaxation_iters: opts.relax_iters,
        energy_minimization_tolerance: opts.energy_minimization_tolerance,
        salt_concentration_m: if opts.env.ionic_strength_m > 0.0 {
            Some(opts.env.ionic_strength_m)
        } else {
            None
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
        u_history: Default::default(),
        ca_acc: vec![[0.0f64; 3]; n_ca],
        ca_n: 0,
    };

    // Post-minimization equilibration: positional restraints + NVT ramp, so the
    // residual build strain is released before production MD.
    if let Some(eq) = &opts.equil {
        equilibrate(&mut engine, eq).map_err(|e| format!("equilibration failed: {e}"))?;
    }

    Ok(engine)
}

/// Create a clone of a parent engine with a mutated structure, reusing the solvent box,
/// water molecules, and ions of the parent. This avoids the expensive solvent packing
/// and full solvent-only equilibration (saving 20-40 seconds of CPU time).
pub fn build_mutant_by_solvent_reuse(
    parent: &SpiceEngine,
    param_set: &FfParamSet,
    mut_input: &crate::structure::StructureInput,
    opts: &BuildOptions,
) -> Result<SpiceEngine, String> {
    let dev = &parent.dev;
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
        hydrogen_constraint: opts.hydrogen_constraint,
        sim_box: SimBoxInit::Fixed((parent.state.cell.bounds_low, parent.state.cell.bounds_high)),
        solvent: dynamics::Solvent::None,
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
        u_history: Default::default(),
        ca_acc: vec![[0.0f64; 3]; n_ca],
        ca_n: 0,
    };

    // Run a fast energy minimization on the mutated system to resolve sidechain clashes!
    // 100-200 iterations are plenty and take <0.2 seconds because water is already minimized.
    if let Some(relax_iters) = opts.relax_iters {
        engine.state.minimize_energy(dev, relax_iters, None);
    }

    Ok(engine)
}
