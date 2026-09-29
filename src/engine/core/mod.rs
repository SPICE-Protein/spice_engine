#![allow(non_snake_case)]
#![allow(confusable_idents)]

//! See the [Readme](https://github.com/David-OConnor/dynamics/blob/main/README.md) for a general overview,
//! or [Molchanica docs, MD section](https://www.athanorlab.com/docs/md.html) for more information about
//! assumptions. Or see the [examples folder on Github](https://github.com/David-OConnor/dynamics/tree/main/examples)
//! for how to use this in your application.
//!
//! The textual information here is informal, and aimed at code maintenance; not library use.
//!
//! This module contains high-level tools for running Newtonian molecular dynamics simulations.
//!
//! [Good article](https://www.owlposting.com/p/a-primer-on-molecular-dynamics)
//! [A summary  on molecular dynamics](https://arxiv.org/pdf/1401.1181)
//!
//! [Amber Force Fields reference](https://ambermd.org/AmberModels.php)
//! [Small molucules using GAFF2](https://ambermd.org/downloads/amber_geostd.tar.bz2)
//! [Amber RM 2025](https://ambermd.org/doc12/Amber25.pdf)
//!
//! To download .dat files (GAFF2), download Amber source (Option 2) [here](https://ambermd.org/GetAmber.php#ambertools).
//! Files are in dat -> leap -> parm
//!
//! Base units: Å, ps (10^-12), Dalton (AMU), native charge units (derive from other base units;
//! not a traditional named unit).
//!
//! Amber: ff19SB for proteins, gaff2 for ligands. (Based on recommendations from https://ambermd.org/AmberModels.php).
//!
//! We use the term "Non-bonded" interactions to refer to Coulomb, and Lennard Interactions, the latter
//! of which is an approximation for both Van der Waals force and exclusion.
//!
//! ## A broad list of components of this simulation:
//! - Water: Rigid OPC solvent molecules that have mutual non-bonded interactions with dynamic atoms and solvent
//! - Thermostat/barostat, with a way to specify temp, pressure, solvent density
//! - OPC solvent model
//! - Cell wrapping
//! - Velocity Verlet integration (Water and non-solvent)
//! - Amber parameters for mass, partial charge, VdW (via LJ), dihedral/improper, angle, bond len
//! - Optimizations for Coulomb: Ewald/SPME.
//! - Optimizations for LJ: Dist cutoff for now.
//! - Amber 1-2, 1-3 exclusions, and 1-4 scaling of covalently-bonded atoms.
//! - Rayon parallelization of non-bonded forces
//! - CPU SIMD (x86_64 AVX2/AVX-512, arm64 NEON) for std-std and water
//!   non-bonded streams is in the production hot path; CUDA parallelization
//!   of non-bonded forces remains WIP.
//! - A thermostat and barostat
//! - An energy-measuring system.
//! - An integrated tool for inferring atom types, bonded-parameter overrides, and partial charges for arbitrary
//!   small organic molecules. (Similar to Amber's Antechamber)
//!
//! --------
//! A timing test, using bond-stretching forces between two atoms only. Measure the period
//! of oscillation for these atom combinations, e.g. using custom Mol2 files.
//! c6-c6: 35fs (correct).   os-os: 47fs        nc-nc: 34fs        hw-hw: 9fs
//! Our measurements, 2025-08-04
//! c6-c6: 35fs    os-os: 31fs        nc-nc: 34fs (Correct)       hw-hw: 6fs
//!
//! --------
//!
//! We use traditional MD non-bonded terms to maintain geometry: Bond length, valence angle between
//! 3 bonded atoms, dihedral angle between 4 bonded atoms (linear), and improper dihedral angle between
//! each hub and 3 spokes. (E.g. at ring intersections). We also apply Coulomb force between atom-centered
//! partial charges, and Lennard Jones potentials to simulate Van der Waals forces. These use spring-like
//! forces to retain most geometry, while allowing for flexibility.
//!
//! We use the OPC solvent model. (See `water_opc.rs`). For both maintaining the geometry of each solvent
//! molecule, and for maintaining Hydrogen atom positions, we do not apply typical non-bonded interactions:
//! We use SHAKE + RATTLE algorithms for these. In the case of solvent, it's required for OPC compliance.
//! For H, it allows us to maintain integrator stability with a greater timestep, e.g. 2fs instead of 1fs.
//!
//! On f32 vs f64 floating point precision: f32 may be good enough for most things, and typical MD packages
//! use mixed precision. Long-range electrostatics are a good candidate for using f64. Or, very long
//! runs.
//!
//! Note on performance: It appears that non-bonded forces dominate computation time. This is my observation,
//! and it's confirmed by an LLM. Both LJ and Coulomb take up most of the time; bonded forces
//! are comparatively insignificant. Building neighbor lists are also significant. These are the areas
//! we focus on for parallel computation (Thread pools, SIMD, CUDA)

// todo: You should keep more data on the GPU betwween time steps, instead of passing back and
// todo forth each time. If practical.

#[path = "../preparation/add_hydrogens/mod.rs"]
mod add_hydrogens;
#[path = "../analysis.rs"]
pub mod analysis;
#[path = "../geometry/barostat.rs"]
mod barostat;
#[path = "../forcefield/bonded.rs"]
mod bonded;
#[path = "../forcefield/bonded_forces.rs"]
mod bonded_forces;
#[path = "../utility/clock.rs"]
pub mod clock;
#[path = "../geometry/config.rs"]
mod config;
#[path = "../cosolvent_presets.rs"]
pub mod cosolvent_presets;
#[path = "../utility/entropy.rs"]
pub mod entropy;
#[path = "../forcefield/forces.rs"]
mod forces;
#[path = "../integrator/integrate.rs"]
pub mod integrate;
#[path = "../forcefield/neighbors.rs"]
mod neighbors;
#[path = "../forcefield/non_bonded/mod.rs"]
mod non_bonded;
#[path = "../panteva.rs"]
pub mod panteva;
#[path = "../forcefield/params.rs"]
pub mod params;
#[path = "../preparation/prep.rs"]
mod prep;
#[cfg(target_arch = "x86_64")]
#[path = "../forcefield/simd.rs"]
mod simd;
#[path = "../utility/snapshot.rs"]
pub mod snapshot;
#[path = "../solvent/mod.rs"]
mod solvent;
#[path = "../species.rs"]
pub mod species;
#[path = "../integrator/thermostat.rs"]
mod thermostat;
#[path = "../utility/util.rs"]
mod util;

#[path = "../constraints/com_zero.rs"]
mod com_zero;
#[cfg(feature = "cuda")]
#[path = "../forcefield/gpu_interface.rs"]
mod gpu_interface;
#[path = "../constraints/minimize_energy.rs"]
pub mod minimize_energy;

#[path = "../forcefield/alchemical.rs"]
pub mod alchemical;
#[path = "../preparation/param_inference/mod.rs"]
pub mod param_inference;
// v1.3.9 web split. `pci_files` is the candle-free bincode half (water
// template load + preference-file save) and compiles in every build.
// `partial_charge_inference` is the candle GNN half; slim builds (wasm)
// opt out via `--no-default-features` so candle/getrandom never enter the
// graph. Callers reach the always-on helpers through `md_core::pci_files`.
#[cfg(feature = "inference")]
#[path = "../preparation/partial_charge_inference/mod.rs"]
pub mod partial_charge_inference;
#[path = "../preparation/partial_charge_inference/files.rs"]
pub mod pci_files;
#[path = "../utility/sa_surface.rs"]
mod sa_surface;

#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::{
    collections::{BTreeMap, HashSet},
    error::Error,
    fmt,
    fmt::{Display, Formatter},
    io,
};

pub use add_hydrogens::{
    add_hydrogens_2::Dihedral,
    bond_vecs::{find_planar_posit, find_tetra_posit_final, find_tetra_posits},
    populate_hydrogens_dihedrals,
};
pub use barostat::SimBox;
#[cfg(feature = "encode")]
use bincode::{Decode, Encode};
use bio_files::{
    AtomGeneric, BondGeneric, Sdf,
    gromacs::gro::Gro,
    md_params::{ForceFieldParams, ForceFieldParamsIndexed, LjParams, MassParams},
    mol2::Mol2,
};
pub use bonded::{LINCS_ITER_DEFAULT, LINCS_ORDER_DEFAULT, SHAKE_TOL_DEFAULT};
pub use config::{ComMotionRemoval, MdConfig};
#[cfg(feature = "cuda")]
use cudarc::{
    driver::{CudaContext, CudaStream},
    nvrtc::Ptx,
};
use ewald::PmeRecip;
pub use integrate::Integrator;
#[allow(unused)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use lin_alg::f32::{Vec3x8, Vec3x16, f32x8, f32x16};
use lin_alg::{f32::Vec3, f64::Vec3 as Vec3F64};
use na_seq::Element;
use neighbors::NeighborsNb;
pub use prep::{HydrogenConstraint, merge_params};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
pub use solvent::{
    ForcesOnWaterMol, Solvent, WaterMolOpc,
    init::{
        OCTANOL_WATER_TEMPLATE, SolventTemplateType, WATER_TEMPLATE_60A, WaterInitTemplate,
        water_mols_from_template, water_mols_from_template_in_region,
    },
    shrinking_box::{
        ShrinkingBoxCfg, ShrinkingBoxPackingCfg, pack_solvent_with_shrinking_box,
        pack_solvent_with_shrinking_box_cfg,
    },
    template_creation::{CustomSolventCount, make_water_mols_grid},
};

#[cfg(feature = "cuda")]
use crate::engine::md_core::gpu_interface::{ForcesPositsGpu, GpuKernels, PerNeighborGpu};
use crate::engine::md_core::non_bonded::{WaterSoluteBatch8, WaterWaterBatch8};
#[cfg(target_arch = "x86_64")]
use crate::engine::md_core::solvent::{WaterMolx8, WaterMolx16};
use crate::engine::md_core::{
    alchemical::StateAlchemical,
    barostat::Barostat,
    non_bonded::{CHARGE_UNIT_SCALER, CompactNonBondedPair, LjTables, NumericSimdPair},
    param_inference::update_small_mol_params,
    params::FfParamSet,
    snapshot::Snapshot,
    solvent::{init::water_mols_from_template_in_region_avoiding, octanol::octanol_mols_from_gro},
    util::{ComputationTime, ComputationTimeSums, build_adjacency_list},
};
pub use crate::engine::md_core::{
    barostat::{BarostatCfg, PRESSURE_DEFAULT, TAU_PRESSURE_DEFAULT},
    solvent::octanol::make_octanol,
    thermostat::{LANGEVIN_GAMMA_DEFAULT, TAU_TEMP_DEFAULT},
};

// Note: If you haven't generated this file yet when compiling (e.g. from a freshly-cloned repo),
// make an edit to one of the CUDA files (e.g. add a newline), then run, to create this file.
#[cfg(feature = "cuda")]
const PTX: &str = include_str!("../dynamics.ptx");

// Multiply by this to convert from kcal/mol to amu • (Å/ps)²  Multiply all accelerations by this.
// Converts *into* our internal units.
const KCAL_TO_NATIVE: f32 = 418.4;

// Multiply by this to convert from amu • (Å/ps)² to kcal/mol.  We use this when accumulating kinetic
// energy, for example. This, in practice, is for temperature and pressure computations.
// Converts *out of * our internal units.
const NATIVE_TO_KCAL: f32 = 1. / KCAL_TO_NATIVE;
// Avogadro constant, for converting molarity -> ion counts.
const AVOGADRO: f64 = 6.02214076e23;

// Every this many steps, re-center the sim (solvent) box.
const CENTER_SIMBOX_RATIO: usize = 30;

// Run SPME once every these steps. It's the slowest computation, and is comparatively
// smooth over time compared to Coulomb and LJ.
//
// MUST stay 1: with SPME_RATIO=2 the reciprocal-space forces/energy are cached on
// the SPME step and reused on the off-step, but the atoms have already moved —
// applying a force computed at STALE positions breaks the symplectic structure and
// injects energy. Measured on 2LYZ in NVE (no thermostat): total energy climbs
// +7.4 kcal/mol/step with the cache, vs +0.85 (≈ noise) with long-range recip
// disabled. Computing SPME every step is the robust fix; a correct cache would need
// reference-position compensation. LAMMPS also computes kspace every step by default.
const SPME_RATIO: usize = 1;

// todo: This may not be necessary, other than having it be a multiple of SPME_RATIO.
// todo: This is because the recording is very fast. (ns order)
// Log computation time every this many steps. (Except for neighbor rebuild)
const COMPUTATION_TIME_RATIO: usize = 20;

#[derive(Debug, Clone, Default)]
pub enum ComputationDevice {
    #[default]
    Cpu,
    #[cfg(feature = "cuda")]
    Gpu(Arc<CudaStream>),
}

/// Represents problems loading parameters. For example, if an atom is missing a force field type
/// or partial charge, or has a force field type that hasn't been loaded.
#[derive(Clone, Debug)]
pub struct ParamError {
    pub descrip: String,
}

impl ParamError {
    pub fn new(descrip: &str) -> Self {
        Self {
            descrip: descrip.to_owned(),
        }
    }
}

impl Display for ParamError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.descrip)
    }
}

impl Error for ParamError {}

impl From<io::Error> for ParamError {
    fn from(err: io::Error) -> Self {
        Self {
            descrip: format!("IO error: {err}"),
        }
    }
}

/// This is used to assign the correct force field parameters to a molecule.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum FfMolType {
    /// Protein or other construct of amino acids
    Peptide,
    /// E.g. a ligand.
    SmallOrganic,
    Dna,
    Rna,
    Lipid,
    Carbohydrate,
}

/// Packages information required to perform dynamics on a Molecule. This is used to initialize
/// the simulation with atoms and related; one or more of these is passed at init.
#[derive(Clone, Debug)]
pub struct MolDynamics {
    pub ff_mol_type: FfMolType,
    /// These must hold force field type and partial charge.
    pub atoms: Vec<AtomGeneric>,
    /// Separate from `atoms`; this may be more convenient than mutating the atoms
    /// as they may move! If None, we use the positions stored in the atoms.
    pub atom_posits: Option<Vec<Vec3F64>>,
    /// This may have uses if "shooting" a molecule into a docking position?
    pub atom_init_velocities: Option<Vec<Vec3>>,
    /// Not required if static.
    pub bonds: Vec<BondGeneric>,
    /// A fast lookup for finding atoms, by index, covalently bonded to each atom.
    /// If None, will be generated automatically from atoms and bonds. Use this
    /// if you wish to cache.
    pub adjacency_list: Option<Vec<Vec<usize>>>,
    /// If true, the atoms in the molecule don't move, but exert LJ and Coulomb forces
    /// on other atoms in the system.
    pub static_: bool,
    /// If present, any values here override molecule-type general parameters.
    pub mol_specific_params: Option<ForceFieldParams>,
    /// If true, this atom exerts and experiences non-bonded forces only.
    /// This may be useful for protein atoms that aren't near a docking site.
    pub bonded_only: bool,
}

/// This is mainly for overriding, while specifying atoms, bonds, posits, and mol type explicitly.
impl Default for MolDynamics {
    fn default() -> Self {
        Self {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: Vec::new(),
            atom_posits: None,
            atom_init_velocities: None,
            bonds: Vec::new(),
            adjacency_list: None,
            static_: false,
            mol_specific_params: None,
            bonded_only: false,
        }
    }
}

impl MolDynamics {
    // todo: from_mmcif?

    /// Load a molecule from a Mol2 file. Includes optional molecule-specific pararmeters.
    /// To work directly, this assumes that forcefield names, and partial charge are present
    /// in the `Mol2` struct for all atoms.
    ///
    /// You may wish to modify the `atom_posits` field after to position this relative to
    /// other molecules.
    pub fn from_mol2(mol: &Mol2, mol_specific_params: Option<ForceFieldParams>) -> Self {
        Self {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: mol.atoms.clone(),
            atom_posits: None,
            atom_init_velocities: None,
            bonds: mol.bonds.clone(),
            adjacency_list: None,
            static_: false,
            mol_specific_params,
            bonded_only: false,
        }
    }

    /// Load a molecule from a SDF file. Includes optional molecule-specific pararmeters.
    /// To work directly, this assumes that forcefield names, and partial charge are present
    /// in the `Mol2` struct for all atoms. Note that these are not present in
    /// SDF files that come from most online databases.
    ///
    /// You may wish to modify the `atom_posits` field after to position this relative to
    /// other molecules.
    pub fn from_sdf(mol: &Sdf, mol_specific_params: Option<ForceFieldParams>) -> Self {
        Self {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: mol.atoms.clone(),
            atom_posits: None,
            atom_init_velocities: None,
            bonds: mol.bonds.clone(),
            adjacency_list: None,
            static_: false,
            mol_specific_params,
            bonded_only: false,
        }
    }

    /// Load an Amber Geostd molecule from an online database, from its unique identifier. This
    /// includes molecule-specific parameters.
    ///
    /// You may wish to modify the `atom_posits` field after to position this relative to
    /// other molecules.
    #[cfg(feature = "network")]
    pub fn from_amber_geostd(ident: &str) -> io::Result<Self> {
        let data = bio_apis::amber_geostd::load_mol_files(ident)
            .map_err(|e| io::Error::other(format!("Error loading data: {e:?}")))?;

        let mol = Mol2::new(&data.mol2)?;
        let params = ForceFieldParams::from_frcmod(&data.frcmod.unwrap())?;

        Ok(Self {
            ff_mol_type: FfMolType::SmallOrganic,
            atoms: mol.atoms,
            atom_posits: None,
            atom_init_velocities: None,
            bonds: mol.bonds,
            adjacency_list: None,
            static_: false,
            mol_specific_params: Some(params),
            bonded_only: false,
        })
    }
}

/// A trimmed-down atom for use with molecular dynamics. Contains parameters for single-atom,
/// but we use ParametersIndex for multi-atom parameters.
#[derive(Clone, Debug, Default)]
pub struct AtomDynamics {
    pub serial_number: u32,
    /// Sources that affect atoms in the system, but are not themselves affected by it. E.g.
    /// in docking, this might be a rigid receptor. They serve as sources for Coulomb and LJ (non-bonded)
    /// interactions, and as anchors for bonded ones.
    pub static_: bool,
    /// If true, this atom exerts and experiences non-bonded forces only.
    /// This may be useful for protein atoms that aren't near a docking site.
    pub bonded_only: bool,
    pub force_field_type: String,
    pub element: Element,
    pub posit: Vec3,
    /// Å / ps
    pub vel: Vec3,
    /// Å / ps²
    pub accel: Vec3,
    /// Å • amu / ps²
    pub force: Vec3,
    /// Daltons or amu
    pub mass: f32,
    /// Amber charge units. This is not the elementary charge units found in amino19.lib and gaff2.dat;
    /// it's scaled by the electrostatic constant.
    pub partial_charge: f32,
    /// Å
    pub lj_sigma: f32,
    /// kcal/mol
    pub lj_eps: f32,
    /// 12-6-4 induction pair constant (Li-Merz OPC column, kcal/mol·Å⁴).
    /// Nonzero only for the divalent salt ions; the water kernels fold
    /// `U4 = −c4/r⁴` into the ion–water-oxygen pair before the force cap.
    pub lj_c4: f32,
}

impl Display for AtomDynamics {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Atom {}: {}, {}. ff: {}, q: {}",
            self.serial_number,
            self.element.to_letter(),
            self.posit,
            self.force_field_type,
            self.partial_charge,
        )?;

        if self.static_ {
            write!(f, ", Static")?;
        }

        Ok(())
    }
}

impl AtomDynamics {
    pub fn new(
        atom: &AtomGeneric,
        atom_posits: &[Vec3],
        i: usize,
        static_: bool,
        bonded_only: bool,
    ) -> Result<Self, ParamError> {
        let ff_type = match &atom.force_field_type {
            Some(ff_type) => ff_type.clone(),
            None => {
                return Err(ParamError::new(&format!(
                    "Atom missing FF type; can't run dynamics: {:?}",
                    atom
                )));
            }
        };

        let partial_charge = match atom.partial_charge {
            Some(p) => p * CHARGE_UNIT_SCALER,
            None => return Err(ParamError::new("Missing partial charge on atom {i}")),
        };

        Ok(Self {
            serial_number: atom.serial_number,
            static_,
            bonded_only,
            element: atom.element,
            posit: atom_posits[i],
            force_field_type: ff_type,
            partial_charge,
            ..Default::default()
        })
    }

    /// Populate atom-specific parameters.
    /// E.g. we use this workflow if creating the atoms prior to the indexed FF.
    pub(crate) fn assign_data_from_params(
        &mut self,
        ff_params: &ForceFieldParamsIndexed,
        i: usize,
    ) {
        self.mass = ff_params.mass[&i].mass;
        self.lj_sigma = ff_params.lennard_jones[&i].sigma;
        self.lj_eps = ff_params.lennard_jones[&i].eps;
    }
}

impl bio_files::md_params::AtomFfSource for AtomDynamics {
    fn ff_type(&self) -> Option<&str> {
        Some(&self.force_field_type)
    }
    fn element(&self) -> na_seq::Element {
        self.element
    }
    fn serial_number(&self) -> u32 {
        self.serial_number
    }
}

#[allow(unused)]
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Debug)]
pub(crate) struct AtomDynamicsx8 {
    pub serial_number: [u32; 8],
    pub static_: [bool; 8],
    pub bonded_only: [bool; 8],
    pub force_field_type: [String; 8],
    pub element: [Element; 8],
    pub posit: Vec3x8,
    pub vel: Vec3x8,
    pub accel: Vec3x8,
    pub mass: f32x8,
    pub partial_charge: f32x8,
    pub lj_sigma: f32x8,
    pub lj_eps: f32x8,
}

#[allow(unused)]
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Debug)]
pub(crate) struct AtomDynamicsx16 {
    pub serial_number: [u32; 16],
    pub static_: [bool; 16],
    pub bonded_only: [bool; 16],
    pub force_field_type: [String; 16],
    pub element: [Element; 16],
    pub posit: Vec3x16,
    pub vel: Vec3x16,
    pub accel: Vec3x16,
    pub mass: f32x16,
    pub partial_charge: f32x16,
    pub lj_sigma: f32x16,
    pub lj_eps: f32x16,
}

// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// impl AtomDynamicsx4 {
//     pub fn from_array(bodies: [AtomDynamics; 4]) -> Self {
//         let mut posits = [Vec3::new_zero(); 4];
//         let mut vels = [Vec3::new_zero(); 4];
//         let mut accels = [Vec3::new_zero(); 4];
//         let mut masses = [0.0; 4];
//         // Replace `Element::H` (for example) with some valid default for your `Element` type:
//         let mut elements = [Element::Hydrogen; 4];
//
//         for (i, body) in bodies.into_iter().enumerate() {
//             posits[i] = body.posit;
//             vels[i] = body.vel;
//             accels[i] = body.accel;
//             masses[i] = body.mass;
//             elements[i] = body.element;
//         }
//
//         Self {
//             posit: Vec3x4::from_array(posits),
//             vel: Vec3x4::from_array(vels),
//             accel: Vec3x4::from_array(accels),
//             mass: f64x4::from_array(masses),
//             element: elements,
//         }
//     }
// }

// todo: FIgure out how to apply this to python.
/// Note: The shortest edge should be > 2(r_cutoff + r_skin), to prevent atoms
/// from interacting with their own image in the real-space component.
#[cfg_attr(feature = "encode", derive(Encode, Decode))]
#[derive(Debug, Clone, PartialEq)]
pub enum SimBoxInit {
    /// Distance in Å from the edge to the molecule, at init.
    Pad(f32),
    /// Coordinate boundaries, at opposite corners
    Fixed((Vec3, Vec3)),
}

impl SimBoxInit {
    /// Centered at the origin; can be moved after, e.g. to center on molecules.
    pub fn new_cube(side_len: f32) -> Self {
        let l = side_len / 2.;
        Self::Fixed((Vec3::new(-l, -l, -l), Vec3::new(l, l, l)))
    }
}

impl Default for SimBoxInit {
    fn default() -> Self {
        Self::Pad(12.)
    }
}

#[derive(Clone, Default, Debug, PartialEq)]
#[cfg_attr(feature = "encode", derive(Encode, Decode))]
/// These are primarily used for debugging and testing, but may be used
/// for specific scenarios as well, e.g. if wishing to speed up computations for real-time use
/// by removing long range forces. These are not standard MD config parameters.
pub struct MdOverrides {
    /// Skips the initial solvent relaxation, where a simulation is run until
    /// hydrogen bonds are established, and temperature is initialized.
    pub skip_water_relaxation: bool,
    /// Leave counterion insertion to an external backend that will consume this
    /// state as an initial coordinate template.
    pub skip_counterion_insertion: bool,
    pub bonded_disabled: bool,
    pub coulomb_disabled: bool,
    pub lj_disabled: bool,
    pub long_range_recip_disabled: bool,
    /// Diagnostics: skip the Langevin thermostat's noise/friction on RIGID WATER
    /// (diagnose whether the per-atom 9-component water noise + SETTLE projection
    /// is responsible for the ~+70 K thermostat-equilibrium offset).
    pub skip_water_thermostat: bool,
    /// Run this block if we wish to, for dev purposes, take snapshots during the
    /// solvent equilibration phase, e.g. for tuning it.
    pub snapshots_during_equilibration: bool,
    /// Take snapshots during the energy minimization phase. (Not solvent equilibration)
    /// This can be used to visually QC this process.
    pub snapshots_during_energy_min: bool,
}

#[derive(Debug, Clone, Default)]
pub struct DistanceRestraint {
    pub atom_0_idx: usize,
    pub atom_1_idx: usize,
    pub r0: f32, // target equilibrium distance (Å)
    pub k: f32,  // force constant (kcal/mol/Å²)
}

#[derive(Default, Clone)]
pub struct MdState {
    // todo: Update how we handle mode A/R.
    // todo: You need to rework this state in light of arbitrary mol count.
    pub cfg: MdConfig,
    pub atoms: Vec<AtomDynamics>,
    #[allow(unused)]
    #[cfg(target_arch = "x86_64")]
    pub(crate) atoms_x8: Vec<AtomDynamicsx8>,
    #[allow(unused)]
    #[cfg(target_arch = "x86_64")]
    pub(crate) atoms_x16: Vec<AtomDynamicsx16>,
    pub water: Vec<WaterMolOpc>,
    #[allow(unused)]
    #[cfg(target_arch = "x86_64")]
    pub(crate) water_x8: Vec<WaterMolx8>,
    #[allow(unused)]
    #[cfg(target_arch = "x86_64")]
    pub(crate) water_x16: Vec<WaterMolx16>,
    /// Note: We don't use bond structs once the simulation is set up; the adjacency list is the
    /// source of this.
    pub adjacency_list: Vec<Vec<usize>>,
    pub force_field_params: ForceFieldParamsIndexed,
    pub distance_restraints: Vec<DistanceRestraint>,
    /// Current simulation time, in picoseconds.
    pub time: f64,
    pub step_count: usize, // increments.
    /// These are the snapshots we keep in memory, accumulating.
    pub snapshots: Vec<Snapshot>,
    pub cell: SimBox,
    pub neighbors_nb: NeighborsNb,
    /// Compact CPU nonbonded pair stream rebuilt with the neighbour list.
    pub(crate) cpu_pairs: Vec<CompactNonBondedPair>,
    /// Number of numeric pair records eligible for the SIMD path after the last rebuild.
    pub(crate) simd_pair_count: usize,
    /// Number of pair records kept on the scalar path after the last rebuild.
    pub(crate) scalar_pair_count: usize,
    // max_disp_sq: f64,           // track atom displacements²
    /// K
    barostat: Barostat,
    /// Exclusions of non-bonded forces for atoms connected by 1, or 2 covalent bonds.
    /// I can't find this in the RM, but ChatGPT is confident of it, and references an Amber file
    /// called 'prmtop', which I can't find. Fishy, but we're going with it.
    pairs_excluded_12_13: HashSet<(usize, usize)>,
    /// See Amber RM, sectcion 15, "1-4 Non-Bonded Interaction Scaling"
    /// These are indices of atoms separated by three consecutive bonds
    pairs_14_scaled: HashSet<(usize, usize)>,
    lj_tables: LjTables,
    pme_recip: Option<PmeRecip>,
    /// kcal/mol
    pub kinetic_energy: f64,
    pub potential_energy: f64,
    /// A newer, simpler approach for energy between molecules, compared to `potential_energy_between_mols`.
    /// This is simply the potential energy from non-bonded interactions, and excludes that from bonded.
    pub potential_energy_nonbonded: f64,
    /// E.g. energy in covalent bonds, as modelled as oscillators.
    pub potential_energy_bonded: f64,
    /// Count of atoms whose acceleration hit the MAX_ACCEL clamp (or a non-finite
    /// guard) on the last integration step, and the largest pre-clamp magnitude
    /// (Å/ps²). Stored so applications can expose them as observables: sustained
    /// clamps mean force spikes (e.g. thermal kicks at high T) are being swallowed
    /// by the clamp instead of relaxing through bonded geometry.
    pub last_clamped_count: usize,
    pub last_clamped_mag: f32,
    /// Instantaneous kinetic temperature (K) at the end of the last step:
    /// T = (2/3)·KE/(N·kB). Lets the caller verify the thermostat actually reaches
    /// the target temperature — a too-weak Langevin coupling keeps T_kin ≈ 298 K
    /// even when `temp_target` is 380 K.
    pub last_temperature_k: f32,
    /// Instantaneous pressure (bar) measured on the last step's forces —
    /// exactly the value that fed `Barostat::scale_factor` for the NEXT
    /// step's drive term. The barostat is linear response in it: sustained
    /// tens-of-kbar readings (clash virials, torn solvation shells) are what
    /// inflate a box during relaxation; apps can watch it to tell a real
    /// overpressure from a numerical phantom.
    pub last_pressure_bar: f64,
    /// Every so many snapshots, write these to file, then clear from memory.
    /// Used to track which molecule each atom is associated with in our flattened structures.
    /// This is the potential energy between every pair of molecules.
    pub potential_energy_between_mols: Vec<f64>,
    snapshot_queue_for_dcd: Vec<Snapshot>,
    snapshot_queue_for_trr: Vec<Snapshot>,
    snapshot_queue_for_xtc: Vec<Snapshot>,
    #[cfg(feature = "cuda")]
    gpu_kernels: Option<GpuKernels>,
    /// These store handles to data structures on the GPU. We pass them to the kernel each
    /// step, but don't transfer. Init to None. Populated during the run.
    #[cfg(feature = "cuda")]
    forces_posits_gpu: Option<ForcesPositsGpu>,
    #[cfg(feature = "cuda")]
    per_neighbor_gpu: Option<PerNeighborGpu>,
    pub neighbor_rebuild_count: usize,
    /// A cache of accel_factor / mass, per atom. Built once, at init.
    mass_accel_factor: Vec<f32>,
    pub computation_time: ComputationTimeSums,
    /// Whether the most recent step rebuilt the neighbor list.
    pub(crate) last_step_neighbor_rebuild: bool,
    /// Whether the most recent step evaluated reciprocal-space PME.
    pub(crate) last_step_pme: bool,
    /// Preclassified ordinary std-std pairs for the CPU SIMD path, with
    /// indices and pair parameters already expanded out of enum/table lookups.
    pub(crate) simd_pairs: Vec<NumericSimdPair>,
    /// Pairs that must remain on the scalar path.
    pub(crate) scalar_pairs: Vec<CompactNonBondedPair>,
    /// Compact water-solute SIMD batches; consumed by the CPU dispatch when
    /// `WATER_SIMD_ACTIVE` holds and the env kill-switch is unset.
    pub(crate) water_simd_pairs: Vec<WaterSoluteBatch8>,
    /// Compact water-water SIMD batches; same gating as `water_simd_pairs`.
    pub(crate) water_water_simd_pairs: Vec<WaterWaterBatch8>,
    pub(crate) water_water_candidate_count: usize,
    pub(crate) water_water_scalar_tail_count: usize,

    /// A cache. We don't run SPME every step; store the previous step's per-atom
    /// force values (Flattened; non-solvent, then solvent M, H0, H1), and apply them
    /// on the steps where we don't re-calculate. (Force, potential energy, virial energy)
    spme_force_prev: Option<(Vec<Vec3>, f64, f64)>,
    // /// Cached at init; used for kinetic energy calculations.
    // _num_static_atoms: usize,
    // todo: Sub-struct for ambient cache like num_static atoms and thermo_dof
    /// Degrees of freedom, used in temperature and kinetic energy calculations.
    thermo_dof: usize,
    /// Count of solute atoms stored at the front of `atoms`; any later atoms belong to
    /// explicit solvent molecules such as octanol or custom solvent templates.
    pub solute_atom_count: usize,
    // todo: Deprecate this if you deprecate per-atom posits and vels in snapshots?
    /// Used to track which molecule each atom is associated with in our flattened structures.
    pub mol_start_indices: Vec<usize>,
    /// A flag we set to disable certain things like snapshots during this MD phase.
    solvent_only_sim_at_init: bool,
    pub alchemical: StateAlchemical,
    /// Index assigned at the start of each MD run. Trajectory files are named
    /// `traj_N.dcd`, `traj_N.trr`, etc. so that successive runs never overwrite
    /// each other.  Chosen as the lowest N for which no such files exist yet.
    /// `None` until the first call to `handle_ss_file_writes`.
    pub run_index: Option<usize>,
}

impl MdState {
    /// Also returns any explicit solvent molecules added. This may be needed by applications
    /// in order to create molecule sets for rendering the trajectories. These are placed, in the
    /// trajectory, after all solute atoms.
    pub fn new(
        dev: &ComputationDevice,
        cfg: &MdConfig,
        mols: &[MolDynamics],
        param_set: &FfParamSet,
    ) -> Result<(Self, Vec<MolDynamics>), ParamError> {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let mxcsr = std::arch::x86_64::_mm_getcsr();
            // 同时开启 FTZ (bit 15) 和 DAZ (bit 6)
            std::arch::x86_64::_mm_setcsr(mxcsr | (1 << 15) | (1 << 6));
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let mut fpcr: u64;
            std::arch::asm!("mrs {}, fpcr", out(reg) fpcr);
            fpcr |= 1 << 24; // FZ bit
            std::arch::asm!("msr fpcr, {}", in(reg) fpcr);
        }

        // We combine all molecule general and specific params into this set, then
        // create Indexed params from it.
        let mut params = ForceFieldParams::default();

        // Used for updating indices for tracking purposes.
        let mut atom_ct_prior_to_this_mol = 0;

        let mut mol_start_indices = Vec::new();

        let posits_solute = {
            let mut p = Vec::new();

            // todo: This is DRY /extra computation with the general atom posits building below, but may be required
            // todo when dealing with a custom solute.
            for mol in mols {
                let atom_posits: Vec<Vec3> = match &mol.atom_posits {
                    Some(a) => a.iter().map(|p| (*p).into()).collect(),
                    None => mol.atoms.iter().map(|a| a.posit.into()).collect(),
                };
                p.extend(atom_posits);
            }

            p
        };

        let octanol_water_template = if cfg.solvent == Solvent::OctanolWithWater {
            Some(Gro::new(OCTANOL_WATER_TEMPLATE).map_err(|e| {
                ParamError::new(&format!("Problem loading the octanol/water GRO file: {e}"))
            })?)
        } else {
            None
        };

        let cell = if let Some(gro) = &octanol_water_template {
            const NM_TO_ANGSTROM: f32 = 10.0;
            SimBox::new(
                (-gro.box_vec * NM_TO_ANGSTROM as f64 / 2.0).into(),
                (gro.box_vec * NM_TO_ANGSTROM as f64 / 2.0).into(),
            )
        } else {
            SimBox::from_solute_atoms(&posits_solute, &cfg.sim_box)
        };

        // let custom_solvent_packed = match &cfg.solvent {
        //     Solvent::Custom((mols_solvent, water_count)) => {
        //         let (mols, snaps) = pack_solvent_with_shrinking_box(
        //             dev,
        //             // todo: Handle cases where len of custom solvent isn't 1.
        //             &mols_solvent[0].0,
        //             mols_solvent[0].1,
        //             *water_count,
        //             cell,
        //             param_set,
        //         )?;
        //
        //         mols
        //     }
        //     _ => Vec::new(),
        // };

        let custom_solvent_packed = match &octanol_water_template {
            Some(gro) => octanol_mols_from_gro(gro)?,
            None => Vec::new(),
        };

        // Build a combined slice: caller-supplied mols first, then packed custom solvents.
        let combined_mols: Vec<MolDynamics>;
        let all_mols: &[MolDynamics] = if custom_solvent_packed.is_empty() {
            mols
        } else {
            combined_mols = mols
                .iter()
                .cloned()
                .chain(custom_solvent_packed.clone())
                .collect();
            &combined_mols
        };

        // We create a flattened atom list, which simplifies our workflow, and is conducive to
        // parallel operations.
        // These Vecs all share indices, and all include all molecules.
        let mut atoms_md = Vec::new();
        let mut adjacency_list = Vec::new();
        let mut solute_atom_count = 0;

        for (mol_i, mol) in all_mols.iter().enumerate() {
            if !mol.atoms.is_empty() {
                mol_start_indices.push(atoms_md.len());
            }

            // Filter out hetero atoms in proteins. These are often example ligands that we do
            // not wish to model.
            // We must perform this filter prior to most of the other steps in this function.
            let mut atoms: Vec<AtomGeneric> = match mol.ff_mol_type {
                FfMolType::Peptide => mol.atoms.iter().filter(|a| !a.hetero).cloned().collect(),
                _ => mol.atoms.to_vec(),
            };

            let mut mol_specific_params = mol.mol_specific_params.clone();

            // Update partial charge, FF names, and param overrides A/R.
            if mol.ff_mol_type == FfMolType::SmallOrganic {
                let mut needs_ff_type_or_q = false;
                for atom in &atoms {
                    if atom.force_field_type.is_none() || atom.partial_charge.is_none() {
                        needs_ff_type_or_q = true;
                        break;
                    }
                }

                if needs_ff_type_or_q {
                    // Note: This invalidates any passed by the user.
                    mol_specific_params = Some(
                        update_small_mol_params(
                            &mut atoms,
                            &mol.bonds,
                            Some(&adjacency_list),
                            param_set.small_mol.as_ref().unwrap(),
                        )
                        .map_err(|_| ParamError {
                            descrip: "Problem inferring params".to_string(),
                        })?,
                    );
                }
            }

            // If the atoms list isn't already filtered by Hetero, and a manual
            // adjacency list or atom posits is passed, this will get screwed up.
            if mol.ff_mol_type == FfMolType::Peptide
                && atoms.len() != mol.atoms.len()
                && (mol.adjacency_list.is_some() || mol.atom_posits.is_some())
            {
                return Err(ParamError::new(
                    "Unable to perform MD on this peptide: If passing atom positions or an adjacency list,\
                 you must already have filtered out hetero atoms. We found one or more hetero atoms in the input.",
                ));
            }

            {
                let params_general = match mol.ff_mol_type {
                    FfMolType::Peptide => &param_set.peptide,
                    FfMolType::SmallOrganic => &param_set.small_mol,
                    FfMolType::Dna => &param_set.dna,
                    FfMolType::Rna => &param_set.rna,
                    FfMolType::Lipid => &param_set.lipids,
                    FfMolType::Carbohydrate => &param_set.carbohydrates,
                };

                let Some(params_general) = params_general else {
                    return Err(ParamError::new(&format!(
                        "Missing general parameters for {:?}",
                        mol.ff_mol_type
                    )));
                };

                // todo: If there are multiple molecules of a given type, this is unnecessary.
                // todo: Make sure overrides from one individual molecule don't affect others, todo,
                // todo and don't affect general params.
                params = merge_params(&params, params_general);

                if let Some(p) = &mol_specific_params {
                    params = merge_params(&params, p);
                }
            }

            let atom_posits: Vec<Vec3> = match &mol.atom_posits {
                Some(a) => a.iter().map(|p| (*p).into()).collect(),
                None => mol.atoms.iter().map(|a| a.posit.into()).collect(),
            };

            for (i, atom) in atoms.iter().enumerate() {
                let mut atom =
                    AtomDynamics::new(atom, &atom_posits, i, mol.static_, mol.bonded_only)?;

                if let Some(vel) = &mol.atom_init_velocities {
                    if i >= vel.len() {
                        return Err(ParamError::new(
                            "Initial velocities passed, but don't match atom len.",
                        ));
                    }
                    atom.vel = vel[i];
                }

                atoms_md.push(atom);
            }

            // Use the included adjacency list if available. If not, construct it.
            let adjacency_list_ = match &mol.adjacency_list {
                Some(a) => a,
                None => &build_adjacency_list(&atoms, &mol.bonds)?,
            };

            // Update indices based on atoms from previously added molecules.
            for aj in adjacency_list_ {
                let mut updated = aj.clone();
                for neighbor in &mut updated {
                    *neighbor += atom_ct_prior_to_this_mol;
                }

                adjacency_list.push(updated);
            }

            atom_ct_prior_to_this_mol += atoms.len();

            if mol_i + 1 == mols.len() {
                solute_atom_count = atoms_md.len();
            }
        }

        // Compute net charge from all solute atoms (internal Amber charge units → elementary).
        let net_q_e: f32 =
            atoms_md.iter().map(|a| a.partial_charge).sum::<f32>() / CHARGE_UNIT_SCALER;
        let n_ions = net_q_e.abs().round() as usize;

        let h_constrained = !matches!(cfg.hydrogen_constraint, HydrogenConstraint::Flexible);
        let force_field_params =
            ForceFieldParamsIndexed::new(&params, &atoms_md, &adjacency_list, h_constrained)
                .map_err(|e| ParamError::new(&e.to_string()))?;

        let mut mass_accel_factor = Vec::with_capacity(atoms_md.len());

        // Assign mass, LJ params, etc.
        for (i, atom) in atoms_md.iter_mut().enumerate() {
            atom.assign_data_from_params(&force_field_params, i);
            mass_accel_factor.push(KCAL_TO_NATIVE / atom.mass);
        }

        // let num_static_atoms = atoms_md.iter().filter(|a| !a.static_).count();

        // Dense molecule-energy analysis is disabled for large solvated systems.
        const MAX_DENSE_MOLECULES: usize = 256;
        let potential_energy_between_mols = if mol_start_indices.len() <= MAX_DENSE_MOLECULES {
            vec![0.; mol_start_indices.len() * mol_start_indices.len()]
        } else {
            Vec::new()
        };

        let mut result = Self {
            cfg: cfg.clone(),
            atoms: atoms_md,
            adjacency_list: adjacency_list.to_vec(),
            cell,
            pairs_excluded_12_13: HashSet::new(),
            pairs_14_scaled: HashSet::new(),
            force_field_params,
            mass_accel_factor,
            // _num_static_atoms: num_static_atoms,
            solute_atom_count,
            mol_start_indices,
            potential_energy_between_mols,
            ..Default::default()
        };

        // Validate atom positions BEFORE recentering so we check against the box that was
        // used for placement (add_copies places atoms relative to the original Fixed bounds).
        // Recentering shifts bounds_low/bounds_high to the atom centroid, which can move
        // atoms that were legitimately near a wall to just outside the new bounds.
        //
        // `OctanolWithWater` is loaded from a pre-equilibrated GRO template whose atoms can
        // legitimately sit very close to the cell faces, so we skip this generic placement
        // validation and preserve the template box as-is.
        if cfg.solvent != Solvent::OctanolWithWater {
            result.check_for_overlaps_oob()?;
            if cfg.recenter_sim_box {
                result.cell.recenter(&result.atoms);
            }
        }

        // Set up our LJ cache. Do this prior to building neighbors for the first time,
        // as that also sets up the GPU-struct LJ data.
        result.lj_tables = LjTables::new(&result.atoms);
        result.neighbors_nb = NeighborsNb::new(result.cfg.neighbor_skin, result.cfg.coulomb_cutoff);

        // Custom solvent molecules were pre-packed and added to `all_mols` before the atom-
        // processing loop above, so their atoms are already in `result.atoms` and their
        // parameters are already included in `force_field_params`.  Water (below) will avoid
        // them automatically because `make_water_mols` checks against `result.atoms`.

        result.water = match &cfg.solvent {
            Solvent::None => Vec::new(),
            Solvent::WaterOpc | Solvent::WaterOpcSpecifyMolCount(_) => {
                let count = match &cfg.solvent {
                    Solvent::WaterOpc => None,
                    Solvent::WaterOpcSpecifyMolCount(c) => Some(*c),
                    _ => unreachable!(),
                };

                water_mols_from_template(
                    &result.cell,
                    &result.atoms,
                    count,
                    &cfg.solvent_template_type,
                    cfg.skip_water_pbc_filter,
                )
            }
            Solvent::WaterOpcCustomRegions(regions) => {
                let mut water_mols = Vec::new();
                let region_offset = result.cell.center() - cell.center();

                for region in regions {
                    let region = region.translated(region_offset);

                    if !result.cell.contains_region(&region) {
                        return Err(ParamError::new(&format!(
                            "Water region sim box {region:?} is out of cell bounds {:?}",
                            cfg.sim_box
                        )));
                    }

                    let region_water = water_mols_from_template_in_region_avoiding(
                        &result.cell,
                        &region,
                        &result.atoms,
                        &water_mols,
                        None,
                        &cfg.solvent_template_type,
                        cfg.skip_water_pbc_filter,
                    )?;
                    water_mols.extend(region_water);
                }

                water_mols
            }
            Solvent::OctanolWithWater => {
                let Some(gro) = &octanol_water_template else {
                    return Err(ParamError::new(
                        "Missing octanol/water GRO template during initialization.",
                    ));
                };

                water_mols_from_gro(gro)?
            }
            // Custom solvents carry their own molecule specs (see
            // `Solvent::Custom`); this function yields only OPC water
            // templates, so custom solvent contributes none here.
            Solvent::Custom(_) => Vec::new(),
        };

        let mut n_ions_total = n_ions;
        if !cfg.overrides.skip_counterion_insertion {
            add_ions(&mut result, net_q_e, n_ions);
        }

        // SPICE: add salt (Na⁺/Cl⁻ pairs) to reach the target ionic strength.
        if let Some(conc_m) = cfg.salt_concentration_m {
            if conc_m > 0.0 {
                let vol_l = f64::from(result.cell.volume()) * 1.0e-27;
                let n_pairs = ((f64::from(conc_m)) * vol_l * AVOGADRO).round() as usize;
                n_ions_total += 2 * n_pairs;
                result.add_salt_ions(n_pairs);
            }
        }
        // Divalent salts (MgCl2-style formula units) on top of the monovalent
        // background; same c·V·N_A counting, three waters per unit.
        for (salt, conc_m) in &cfg.divalent_salts {
            if *conc_m > 0.0 {
                let vol_l = f64::from(result.cell.volume()) * 1.0e-27;
                let n_units = ((f64::from(*conc_m)) * vol_l * AVOGADRO).round() as usize;
                n_ions_total += 3 * n_units;
                result.add_divalent_salt(*salt, n_units);
            }
        }
        // General electrolyte channel (v1.3.2): any charge-balanced
        // cation/anion pair; stoichiometry from the species charges, count
        // per formula unit at c·V·N_A exactly like the named knobs.
        for salt in &cfg.salts {
            if salt.molarity > 0.0 {
                let vol_l = f64::from(result.cell.volume()) * 1.0e-27;
                let n_units = ((f64::from(salt.molarity)) * vol_l * AVOGADRO).round() as usize;
                n_ions_total += salt.stoichiometry().map_or(0, |(c, a)| (c + a) as usize) * n_units;
                result.add_salt(salt, n_units);
            }
        }
        // User-provided cosolvents (urea-style denaturants etc. — mechanism
        // here, parameter data from the caller). Note the exclusion flags are
        // (re)built inside add_cosolvent from bonds_topology.
        for cospec in &cfg.cosolvents {
            if cospec.molarity > 0.0 {
                let vol_l = f64::from(result.cell.volume()) * 1.0e-27;
                let n_mols = ((f64::from(cospec.molarity)) * vol_l * AVOGADRO).round() as usize;
                result.add_cosolvent(cospec, n_mols);
            }
        }
        validate_mol_start_indices(result.atoms.len(), &result.mol_start_indices)
            .map_err(|e| ParamError::new(&e))?;

        // Neutralizing path may have run (add_ions calls this itself only for
        // salt changes); check once for the counterion-only build too.
        if !cfg.overrides.skip_counterion_insertion && cfg.salt_concentration_m.is_none() {
            warn_if_not_neutral(&result);
        }

        // Report the EFFECTIVE total ionic strength (½Σcᵢzᵢ²) of the built box —
        // it can exceed the NaCl `salt_concentration_m` knob once divalent salts
        // or `salts_json` electrolytes are present, so surface the real number.
        if n_ions_total > 0 {
            let ionic = result.effective_ionic_strength_m();
            if ionic > 0.0 {
                eprintln!(
                    "System ionic strength: {ionic:.4} M (effective I; NaCl knob alone would \
                     understate it with divalent/extra salts)."
                );
            }
        }

        // Rebuild the LJ table to include any ions that were appended after the initial build.
        if n_ions_total > 0 {
            result.lj_tables = LjTables::new(&result.atoms);
        }

        // Calc DOF only after all atoms and solvent are initialized.
        result.thermo_dof = result.dof_for_thermo();

        result.setup_nonbonded_exclusion_scale_flags();

        result.build_all_neighbors(dev);

        // Initializes the FFT planner[s], among other things.
        result.regen_pme(dev);

        // Allocate force buffers on the GPU, and store a handle. Used for the entire run.
        // Initialize the per-neighbor data as well; we will do this again every time
        // we compute neighbors.
        #[cfg(feature = "cuda")]
        if let ComputationDevice::Gpu(stream) = dev {
            let ctx = CudaContext::new(0).unwrap();
            let module = ctx.load_module(Ptx::from_src(PTX)).unwrap();

            result.gpu_kernels = Some(GpuKernels {
                primary: module.load_function("nonbonded_force_kernel").unwrap(),
                alchemical: module
                    .load_function("nonbonded_force_alchemical_kernel")
                    .unwrap(),
                zero_f32: module.load_function("zero_f32").unwrap(),
                zero_f64: module.load_function("zero_f64").unwrap(),
            });

            result.forces_posits_gpu = Some(ForcesPositsGpu::new(
                stream,
                result.atoms.len(),
                result.water.len(),
                result.cfg.coulomb_cutoff,
                result.cfg.spme_alpha,
            ));

            result.per_neighbor_gpu = Some(PerNeighborGpu::new(
                stream,
                &result.nb_pairs,
                &result.atoms,
                &result.water,
                &result.lj_tables,
            ));
        }

        // todo: Move this AR
        // Pack SIMD once at init.
        #[cfg(target_arch = "x86_64")]
        result.pack_atoms();

        if !result.cfg.overrides.skip_water_relaxation {
            result.md_on_solute_only(dev);
        }

        if let Some(max_iters) = result.cfg.max_init_relaxation_iters {
            result.minimize_energy(dev, max_iters, None);
        }

        // Reset computation time to negate anything that was applied by minimization, initial
        // neighbor rebuild, and anything else done here that may affect it.
        result.computation_time = Default::default();

        Ok((result, custom_solvent_packed))
    }

    /// This way of returning a Result isn't great semantically, but it works.
    fn check_for_overlaps_oob(&mut self) -> Result<(), ParamError> {
        const MIN_DIST_FROM_EDGE: f32 = 0.5; // Å
        const MIN_INTER_MOL_DIST: f32 = 0.5; // Å

        let lo = self.cell.bounds_low;
        let hi = self.cell.bounds_high;

        for (i, atom) in self.atoms.iter().enumerate() {
            let p = atom.posit;

            if p.x < lo.x || p.y < lo.y || p.z < lo.z || p.x > hi.x || p.y > hi.y || p.z > hi.z {
                return Err(ParamError::new(&format!(
                    "Atom index {i} is outside the sim box. \
                         Pos ({:.3}, {:.3}, {:.3}), box [{:.3}..{:.3}, {:.3}..{:.3}, {:.3}..{:.3}]",
                    p.x, p.y, p.z, lo.x, hi.x, lo.y, hi.y, lo.z, hi.z,
                )));
            }

            let dist_to_edge = (p.x - lo.x)
                .min(hi.x - p.x)
                .min(p.y - lo.y)
                .min(hi.y - p.y)
                .min(p.z - lo.z)
                .min(hi.z - p.z);

            if dist_to_edge < MIN_DIST_FROM_EDGE {
                return Err(ParamError::new(&format!(
                    "Atom index {i} is too close to the sim box edge ({dist_to_edge:.3} Å). \
                         Pos ({:.3}, {:.3}, {:.3})",
                    p.x, p.y, p.z,
                )));
            }
        }

        // Inter-molecular minimum-image overlap check.
        // This catches the periodic-image case: molecules that appear far apart in direct
        // space but whose images wrap to overlap on the other side of the cell.
        if self.mol_start_indices.len() > 1 {
            let mut atom_mol = vec![0usize; self.atoms.len()];
            for (mol_i, &start) in self.mol_start_indices.iter().enumerate() {
                let end = self
                    .mol_start_indices
                    .get(mol_i + 1)
                    .copied()
                    .unwrap_or(self.atoms.len());
                for idx in start..end {
                    atom_mol[idx] = mol_i;
                }
            }

            for i in 0..self.atoms.len() {
                for j in (i + 1)..self.atoms.len() {
                    if atom_mol[i] == atom_mol[j] {
                        continue;
                    }
                    let diff = self
                        .cell
                        .min_image(self.atoms[i].posit - self.atoms[j].posit);
                    let dist = diff.magnitude();
                    if dist < MIN_INTER_MOL_DIST {
                        let pi = self.atoms[i].posit;
                        let pj = self.atoms[j].posit;
                        eprintln!(
                            "check_for_overlaps_oob FAIL: atom {i} pos=({:.3},{:.3},{:.3}) \
                             atom {j} pos=({:.3},{:.3},{:.3}) direct_dist={:.3} Å  \
                             min_image_dist={dist:.3} Å  cell_extent=({:.3},{:.3},{:.3})",
                            pi.x,
                            pi.y,
                            pi.z,
                            pj.x,
                            pj.y,
                            pj.z,
                            (self.atoms[i].posit - self.atoms[j].posit).magnitude(),
                            self.cell.extent.x,
                            self.cell.extent.y,
                            self.cell.extent.z,
                        );
                        return Err(ParamError::new(&format!(
                            "Atoms from different molecules (indices {i} and {j}) are too \
                                 close in minimum-image distance ({dist:.3} Å). This would cause \
                                 immediate simulation malfunction.",
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    pub fn computation_time(&self) -> io::Result<ComputationTime> {
        self.computation_time.time_per_step(self.step_count)
    }

    pub fn nb_pair_count(&self) -> usize {
        self.cpu_pairs.len()
    }

    pub fn simd_pair_count(&self) -> usize {
        self.simd_pair_count
    }

    /// Number of std-water pair records evaluated by the water SIMD path.
    pub fn water_simd_pair_count(&self) -> usize {
        self.water_simd_pairs.len() * 8
    }

    /// Number of water-water pair records evaluated by the water SIMD path.
    pub fn water_water_simd_pair_count(&self) -> usize {
        self.water_water_simd_pairs.len() * 8
    }

    pub fn water_water_candidate_count(&self) -> usize {
        self.water_water_candidate_count
    }

    pub fn water_water_scalar_tail_count(&self) -> usize {
        self.water_water_scalar_tail_count
    }

    pub fn scalar_pair_count(&self) -> usize {
        self.scalar_pair_count
    }

    pub fn last_step_neighbor_rebuilt(&self) -> bool {
        self.last_step_neighbor_rebuild
    }

    pub fn last_step_used_pme(&self) -> bool {
        self.last_step_pme
    }

    pub fn step_kind(&self) -> &'static str {
        match (self.last_step_neighbor_rebuild, self.last_step_pme) {
            (true, true) => "pme+rebuild",
            (true, false) => "rebuild",
            (false, true) => "pme",
            (false, false) => "ordinary",
        }
    }

    pub fn virial_components(&self) -> (f64, f64, f64, f64) {
        let v = self.barostat.virial.to_kcal_mol();
        (
            v.bonded,
            v.nonbonded_short_range,
            v.nonbonded_long_range,
            v.constraints,
        )
    }

    /// Topology-exclusion bookkeeping, for diagnostics: sizes of the 1-2/1-3
    /// excluded set, the 1-4 scaled set, the FF bond/angle/dihedral tables the
    /// exclusions were built from, and total std-std neighbor pairs. If
    /// `excluded`/`scaled14` look tiny next to `bonds`/`angles`, bonded pairs
    /// are leaking into the nonbonded sums (huge positive short-range virial).
    pub fn exclusion_diagnostics(&self) -> (usize, usize, usize, usize, usize, usize) {
        let ff = &self.force_field_params;
        (
            self.pairs_excluded_12_13.len(),
            self.pairs_14_scaled.len(),
            ff.bonds_topology.len(),
            ff.angle.len(),
            ff.dihedral.len(),
            self.cpu_pairs.len(),
        )
    }

    /// Reset acceleration, force, potential energy, and virial. Do this each step after the first half-step and drift, and
    /// shaking the fixed hydrogens.
    /// We must reset the virial pair prior to accumulating it, which we do when calculating non-bonded
    /// forces. Also reset forces on solvent.
    pub(crate) fn reset_f_acc_pe_virial(&mut self) {
        for a in &mut self.atoms {
            a.accel = Vec3::new_zero();
            a.force = Vec3::new_zero();
        }
        for mol in &mut self.water {
            mol.o.accel = Vec3::new_zero();
            mol.m.accel = Vec3::new_zero();
            mol.h0.accel = Vec3::new_zero();
            mol.h1.accel = Vec3::new_zero();

            mol.o.force = Vec3::new_zero();
            mol.m.force = Vec3::new_zero();
            mol.h0.force = Vec3::new_zero();
            mol.h1.force = Vec3::new_zero();
        }

        self.barostat.virial = Default::default();

        self.potential_energy = 0.;
        self.potential_energy_nonbonded = 0.;
        self.potential_energy_bonded = 0.;
        if self.mol_start_indices.len() <= 256 && !self.potential_energy_between_mols.is_empty() {
            self.potential_energy_between_mols.fill(0.0);
        } else {
            self.potential_energy_between_mols.clear();
        }

        self.alchemical.dh_dl = 0.0;
    }

    /// Entry point for force application. This includes bonded, non-bonded (LJ and Coulomb/SPME), and
    /// optionally external forces. (Indexed by atom).
    /// Set all atom positions and rebuild the neighbor list — used to restart
    /// independent MD segments from a clean reference conformation without
    /// leaving a stale neighbor list behind.
    pub fn set_positions_rebuild(&mut self, dev: &ComputationDevice, pos: &[Vec3]) {
        debug_assert_eq!(pos.len(), self.atoms.len());
        for (a, p) in self.atoms.iter_mut().zip(pos) {
            a.posit = *p;
        }
        self.build_all_neighbors(dev);
        // Externally changing positions invalidates cached force-derived state:
        // the SPME reciprocal-space cache was computed at the old positions and
        // the accumulated forces/accelerations belong to the previous
        // conformation. Clear them so the next step recomputes everything at
        // the restored positions — otherwise a restart can inherit a force /
        // energy spike and crash (observed systematically when reusing a
        // template engine at a new temperature).
        self.spme_force_prev = None;
        self.reset_f_acc_pe_virial();
    }

    /// Restore the full reference state for an independent MD segment/point:
    /// box, positions, neighbor list, and all position/box-derived caches
    /// (forces, SPME, PME reciprocal lattice). Reusing an engine across
    /// temperatures/points while only restoring positions leaves the box at the
    /// build temperature's volume — at a colder temperature the density is
    /// wrong and the pressure spike crashes the restart (fresh builds at that
    /// temperature are stable). Restoring the box too makes every restart
    /// start from exactly the reference state, matching a fresh build.
    pub fn set_state_rebuild(&mut self, dev: &ComputationDevice, pos: &[Vec3], cell: &SimBox) {
        debug_assert_eq!(pos.len(), self.atoms.len());
        self.cell = *cell;
        for (a, p) in self.atoms.iter_mut().zip(pos) {
            a.posit = *p;
        }
        self.build_all_neighbors(dev);
        self.reset_f_acc_pe_virial();
        self.spme_force_prev = None;
        self.regen_pme(dev);
    }

    pub(crate) fn apply_all_forces(
        &mut self,
        dev: &ComputationDevice,
        external_force: Option<&[Vec3]>,
    ) {
        let mut start = clock::Mono::now();
        let log_time = self.step_count.is_multiple_of(COMPUTATION_TIME_RATIO);

        if !self.cfg.overrides.bonded_disabled {
            self.apply_bonded_forces();
        }

        if log_time {
            let elapsed = start.elapsed().as_micros() as u64;
            self.computation_time.bonded_sum += elapsed;
            start = clock::Mono::now();
        }

        self.apply_nonbonded_forces(dev);

        if log_time {
            let elapsed = start.elapsed().as_micros() as u64;
            self.computation_time.non_bonded_short_range_sum += elapsed;
        }

        // Note: We currently set to skip these on energy minimization, but this
        // check may fail at step 0 anyway?
        // todo: When skipping long range forces, you may wish to use naive coulomb instead
        // todo of the short-range part of the recip. This depends on the application.
        if !self.cfg.overrides.long_range_recip_disabled {
            let compute_spme_every_step = self.alchemical.mol_idx.is_some();

            if compute_spme_every_step
                || self.step_count.is_multiple_of(SPME_RATIO)
                || self.spme_force_prev.is_none()
            {
                // Compute SPME recip forces as usual, and cache for use in steps where we don't.

                // Note: This relies on SPME_RATIO being divisible by COMPUTATION_TIME_RATIO.
                // It will produce inaccurate results otherwise.
                if log_time {
                    start = clock::Mono::now();
                }

                self.last_step_pme = true;
                let data = self.handle_spme_recip(dev);

                if !compute_spme_every_step && SPME_RATIO != 1 {
                    self.spme_force_prev = Some(data);
                }

                if log_time {
                    let elapsed = start.elapsed().as_micros() as u64;
                    self.computation_time.ewald_long_range_sum += elapsed;
                }
            } else {
                // Use the previously-cached SPME forces.
                match &self.spme_force_prev {
                    Some((forces, potential_e, virial_e)) => {
                        // Unpack; forces were applied to flattened solvent and non-solvent
                        // due to how our GPU pipeline works.

                        // self.unpack_apply_pme_forces(forces, &[]);
                        // todo: This is a C+P from the unpack fn! We are getting a borrow error otherwise.
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

                        self.potential_energy += potential_e;
                        self.potential_energy_nonbonded += potential_e;

                        self.barostat.virial.nonbonded_long_range += virial_e;
                    }
                    None => {
                        eprintln!(
                            "Error! Attempting to use cached previous SPME forces, but it's not set"
                        );
                    }
                }
            }
        }

        if let Some(f_ext) = external_force {
            for (i, f) in f_ext.iter().enumerate() {
                self.atoms[i].force += *f;
            }
        }

        self.apply_efield();
    }

    /// Uniform (optionally cosine-oscillating) external electric field.
    /// F_i += q_i·E(t) on every charged site including rigid-water M/H, and
    /// the field potential U = −E·Σq_i r_i is folded into `potential_energy`
    /// (same accounting as LAMMPS `fix efield`). No virial contribution: a
    /// homogeneous external field is not a pair interaction.
    ///
    /// `cfg.efield` is stored in kcal·mol⁻¹·e⁻¹·Å⁻¹; forces here use the
    /// engine's charge scaling (CHARGE_UNIT_SCALER per elementary charge),
    /// hence the division. Energy bookkeeping stays in kcal/mol.
    fn apply_efield(&mut self) {
        let e0 = self.cfg.efield;
        if e0 == [0.0; 3] {
            return;
        }
        let scale = if self.cfg.efield_omega != 0.0 {
            (self.cfg.efield_omega * self.time as f32).cos()
        } else {
            1.0
        };
        // Pre-scaled so `efield_term` can multiply the engine-scaled charge
        // directly: q_scaled·(E/SCALER) = q_e·E.
        let e = [
            e0[0] * scale / CHARGE_UNIT_SCALER,
            e0[1] * scale / CHARGE_UNIT_SCALER,
            e0[2] * scale / CHARGE_UNIT_SCALER,
        ];

        let mut field_potential = 0.0_f64;
        for a in &mut self.atoms {
            let (f, u) = efield_term(a.partial_charge, a.posit, e);
            a.force += f;
            field_potential += u;
        }
        for w in &mut self.water {
            for site in [&mut w.o, &mut w.m, &mut w.h0, &mut w.h1] {
                let (f, u) = efield_term(site.partial_charge, site.posit, e);
                site.force += f;
                field_potential += u;
            }
        }
        self.potential_energy += field_potential;
    }
}

/// Pure per-charge efield contribution: force (engine units) on a site with
/// engine-scaled charge `q_scaled` at `posit` under field `e_scaled`
/// (= E/(kcal·mol⁻¹·e⁻¹·Å⁻¹) / CHARGE_UNIT_SCALER), and the potential
/// energy contribution −q_e·E·r in kcal/mol (derived as
/// −q_scaled·e_scaled·r, equal to −q_e·(E_raw)·r by construction).
fn efield_term(q_scaled: f32, posit: Vec3, e_scaled: [f32; 3]) -> (Vec3, f64) {
    let f = Vec3::new(
        q_scaled * e_scaled[0],
        q_scaled * e_scaled[1],
        q_scaled * e_scaled[2],
    );
    let u = -f64::from(q_scaled)
        * (f64::from(e_scaled[0]) * f64::from(posit.x)
            + f64::from(e_scaled[1]) * f64::from(posit.y)
            + f64::from(e_scaled[2]) * f64::from(posit.z));
    (f, u)
}

impl Display for MdState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MdState. # Snapshots: {}. # steps: {}  Current time: {}. # of dynamic atoms: {}. # Water mols: {}",
            self.snapshots.len(),
            self.step_count,
            self.time,
            self.atoms.len(),
            self.water.len()
        )
    }
}

/// Mutable aliasing helpers.
pub(crate) fn split2_mut<T>(v: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    assert!(i != j);

    let (low, high) = if i < j { (i, j) } else { (j, i) };
    let (left, right) = v.split_at_mut(high);
    (&mut left[low], &mut right[0])
}

fn split3_mut<T>(v: &mut [T], i: usize, j: usize, k: usize) -> (&mut T, &mut T, &mut T) {
    let len = v.len();
    assert!(i < len && j < len && k < len, "index out of bounds");
    assert!(i != j && i != k && j != k, "indices must be distinct");

    // SAFETY: we just asserted that 0 <= i,j,k < v.len() and that they're all different.
    let ptr = v.as_mut_ptr();
    unsafe {
        let a = &mut *ptr.add(i);
        let b = &mut *ptr.add(j);
        let c = &mut *ptr.add(k);
        (a, b, c)
    }
}

pub(crate) fn split4_mut<T>(
    slice: &mut [T],
    i0: usize,
    i1: usize,
    i2: usize,
    i3: usize,
) -> (&mut T, &mut T, &mut T, &mut T) {
    // Safety gates
    let len = slice.len();
    assert!(
        i0 < len && i1 < len && i2 < len && i3 < len,
        "index out of bounds"
    );
    assert!(
        i0 != i1 && i0 != i2 && i0 != i3 && i1 != i2 && i1 != i3 && i2 != i3,
        "indices must be pair-wise distinct"
    );

    unsafe {
        let base = slice.as_mut_ptr();
        (
            &mut *base.add(i0),
            &mut *base.add(i1),
            &mut *base.add(i2),
            &mut *base.add(i3),
        )
    }
}

/// Set up with no solvent molecules or relaxation. Run one step to compute energies, then return
/// the snapshot taken.
pub fn compute_energy_snapshot(
    dev: &ComputationDevice,
    mols: &[MolDynamics],
    param_set: &FfParamSet,
) -> Result<Snapshot, ParamError> {
    let cfg = MdConfig {
        integrator: Integrator::VerletVelocity { thermostat: None },
        hydrogen_constraint: HydrogenConstraint::Flexible,
        max_init_relaxation_iters: None,
        solvent: Solvent::None,
        ..Default::default()
    };

    let (mut md_state, _) = MdState::new(dev, &cfg, mols, param_set)?;

    // dt is arbitrary?
    let dt = 0.001;
    md_state.step(dev, dt, None);

    if md_state.snapshots.is_empty() {
        return Err(ParamError {
            descrip: String::from("Snapshots empty on energy compuptation"),
        });
    }

    Ok(md_state.snapshots[0].clone())
}

fn water_mols_from_gro(gro: &Gro) -> Result<Vec<WaterMolOpc>, ParamError> {
    const NM_TO_ANGSTROM: f32 = 10.0;

    #[derive(Default)]
    struct WaterGroMol {
        o: Option<(Vec3, Vec3)>,
        h0: Option<(Vec3, Vec3)>,
        h1: Option<(Vec3, Vec3)>,
    }

    let mut water_by_mol_id: BTreeMap<u32, WaterGroMol> = BTreeMap::new();

    for atom in &gro.atoms {
        if atom.mol_name != "SOL" {
            continue;
        }

        let Some(vel) = atom.velocity else {
            return Err(ParamError::new(&format!(
                "Missing velocity on water atom {} in octanol/water GRO template.",
                atom.serial_number
            )));
        };

        let posit: Vec3 = atom.posit.into();
        let vel: Vec3 = vel.into();
        let posit = posit * NM_TO_ANGSTROM;
        let vel = vel * NM_TO_ANGSTROM;

        let entry = water_by_mol_id.entry(atom.mol_id).or_default();

        match atom.atom_type.as_str() {
            "OW" => entry.o = Some((posit, vel)),
            "HW1" => entry.h0 = Some((posit, vel)),
            "HW2" => entry.h1 = Some((posit, vel)),
            _ => (),
        }
    }

    let mut water = Vec::with_capacity(water_by_mol_id.len());

    for (mol_id, sites) in water_by_mol_id {
        let Some((o_posit, o_vel)) = sites.o else {
            return Err(ParamError::new(&format!(
                "Water molecule {mol_id} is missing OW in octanol/water GRO template."
            )));
        };
        let Some((h0_posit, h0_vel)) = sites.h0 else {
            return Err(ParamError::new(&format!(
                "Water molecule {mol_id} is missing HW1 in octanol/water GRO template."
            )));
        };
        let Some((h1_posit, h1_vel)) = sites.h1 else {
            return Err(ParamError::new(&format!(
                "Water molecule {mol_id} is missing HW2 in octanol/water GRO template."
            )));
        };

        // Orientation comes from the template's raw site positions, but the
        // internal geometry (O–H bond, HOH angle, O–EP offset) is re-placed at
        // the canonical OPC constants — never copied from the GRO template,
        // whose TIP3P-distance H sites would over-deepen every H-bond under
        // OPC charges (see solvent/init.rs water-placement note).
        let v_h0 = h0_posit - o_posit;
        let v_h1 = h1_posit - o_posit;
        let mut mol = WaterMolOpc::from_axes(o_posit, o_vel, v_h0 + v_h1, v_h0);
        mol.h0.vel = h0_vel;
        mol.h1.vel = h1_vel;
        mol.update_virtual_site();

        water.push(mol);
    }

    Ok(water)
}

// todo: Move this A/R
pub(crate) fn validate_mol_start_indices(
    n_atoms: usize,
    mol_start_indices: &[usize],
) -> Result<(), String> {
    if mol_start_indices.is_empty() {
        if n_atoms == 0 {
            return Ok(());
        }

        return Err(format!(
            "missing molecule start indices for {n_atoms} non-solvent atom(s)"
        ));
    }

    if mol_start_indices[0] != 0 {
        return Err(format!(
            "first molecule starts at atom {}, but it must start at 0",
            mol_start_indices[0]
        ));
    }

    let mut prev = mol_start_indices[0];
    if prev >= n_atoms {
        return Err(format!(
            "molecule 0 starts at atom {prev}, but there are only {n_atoms} atom(s)"
        ));
    }

    for (mol_idx, &start) in mol_start_indices.iter().enumerate().skip(1) {
        if start >= n_atoms {
            return Err(format!(
                "molecule {mol_idx} starts at atom {start}, but valid atom indices are 0..{}",
                n_atoms.saturating_sub(1)
            ));
        }
        if start <= prev {
            return Err(format!(
                "molecule starts must be strictly increasing; molecule {mol_idx} starts at {start} after {prev}"
            ));
        }
        prev = start;
    }

    Ok(())
}

/// Add one ion atom by displacing the water molecule at `w_idx` (does NOT remove the
/// water; the caller must remove all displaced waters afterwards, in descending order).
/// Joung–Cheatham parameters tuned for OPC water (Amber frcmod.ionsjc_opc).
pub(crate) fn insert_ion(
    state: &mut MdState,
    w_idx: usize,
    ff_type: &str,
    elem: Element,
    mass: f32,
    q: f32,
    sigma: f32,
    eps: f32,
    c4: f32,
) {
    let atom_idx = state.atoms.len();
    let posit = state.water[w_idx].o.posit;

    state.atoms.push(AtomDynamics {
        serial_number: atom_idx as u32,
        force_field_type: ff_type.to_string(),
        element: elem,
        posit,
        mass,
        partial_charge: q,
        lj_sigma: sigma,
        lj_eps: eps,
        lj_c4: c4,
        ..Default::default()
    });
    state.force_field_params.mass.insert(
        atom_idx,
        MassParams {
            atom_type: ff_type.to_string(),
            mass,
            comment: None,
        },
    );
    state.force_field_params.lennard_jones.insert(
        atom_idx,
        LjParams {
            atom_type: ff_type.to_string(),
            sigma,
            eps,
        },
    );
    state.adjacency_list.push(Vec::new());
    state.mass_accel_factor.push(KCAL_TO_NATIVE / mass);
    state.mol_start_indices.push(atom_idx);
}

/// Remove displaced water molecules. Indices may be in any order; they are deduplicated
/// and removed from highest to lowest to avoid index shifting.
pub(crate) fn remove_waters(state: &mut MdState, mut w_indices: Vec<usize>) {
    w_indices.sort_unstable_by(|a, b| b.cmp(a));
    w_indices.dedup();
    for idx in w_indices {
        state.water.remove(idx);
    }
}

/// Write ONE instance of a [`SpeciesSpec`] into displaced water slot
/// `slot_w` — the single insertion algorithm behind every path (counterions,
/// salts, cosolutes; v1.3.2). The caller owns slot selection
/// (`pick_ion_slots`), water removal, and table/DOF refresh, so the
/// historical tail order (append ions, then delete waters) is unchanged.
///
/// Atomic species (single site, zero offset, no bonds) defer entirely to
/// `insert_ion`, which is what makes them *bit-identical* to the v1.3.1
/// output: the position is taken verbatim from the slot's water O (no
/// `center + offset` float arithmetic, which could flip signed zeros), the
/// `MassParams` comment stays `None` (the ion path never wrote one), and
/// `mass_accel_factor` keeps its un-clamped division.
///
/// Multi-site instances reproduce the old `add_cosolvent` loop line for
/// line, with one deliberate change: each site's `c4` is now threaded into
/// `lj_c4`. In v1.3.1 it silently fell to the `Default` (0.0), so a
/// cosolute-shaped species carrying 12-6-4 coefficients lost them; shipped
/// CGenFF presets are pure 12-6 (c4 = 0), so *their* output is unaffected.
pub(crate) fn insert_instance(state: &mut MdState, sp: &SpeciesSpec, slot_w: usize) {
    if sp.is_atomic() {
        let s = &sp.sites[0];
        insert_ion(
            state,
            slot_w,
            &s.ff_type,
            s.element,
            s.mass,
            s.charge_scaled,
            s.sigma,
            s.eps,
            s.c4,
        );
        return;
    }
    for &(a, b, _, _) in &sp.bonds {
        assert!(
            a < sp.sites.len() && b < sp.sites.len() && a != b,
            "species {}: bond ({a},{b}) out of site range",
            sp.name
        );
    }
    let center = state.water[slot_w].o.posit;
    let first = state.atoms.len();
    for site in &sp.sites {
        let atom_idx = state.atoms.len();
        state.atoms.push(AtomDynamics {
            serial_number: atom_idx as u32,
            force_field_type: site.ff_type.clone(),
            element: site.element,
            posit: center + site.offset,
            mass: site.mass,
            partial_charge: site.charge_scaled,
            lj_sigma: site.sigma,
            lj_eps: site.eps,
            lj_c4: site.c4,
            ..Default::default()
        });
        state.force_field_params.mass.insert(
            atom_idx,
            MassParams {
                atom_type: site.ff_type.clone(),
                mass: site.mass,
                comment: Some(sp.name.clone()),
            },
        );
        state.force_field_params.lennard_jones.insert(
            atom_idx,
            LjParams {
                atom_type: site.ff_type.clone(),
                sigma: site.sigma,
                eps: site.eps,
            },
        );
        state.adjacency_list.push(Vec::new());
        state
            .mass_accel_factor
            .push(KCAL_TO_NATIVE / site.mass.max(1e-6));
    }
    // One instance = one mol group (first site index).
    state.mol_start_indices.push(first);
    for &(a, b, k, r0) in &sp.bonds {
        let (ia, ib) = (first + a, first + b);
        // Harmonic connectivity (bonded force comes from restraints).
        state.distance_restraints.push(DistanceRestraint {
            atom_0_idx: ia,
            atom_1_idx: ib,
            r0,
            k,
        });
        // 1-2 exclusion source + adjacency symmetry.
        state
            .force_field_params
            .bonds_topology
            .insert((ia.min(ib), ia.max(ib)));
        state.adjacency_list[ia].push(ib);
        state.adjacency_list[ib].push(ia);
    }
}

/// Amber/CHARMM parameter files tabulate **R_min/2** ("R*"); the engine's
/// `lj_sigma` field is a true σ (used in the 4ε[(σ/r)¹²−(σ/r)⁶] form, paired
/// by arithmetic mean). The conversion is σ = 2·R*/2^(1/6), i.e. this factor.
/// bio_files' `LjParams::from_line` applies the same factor to every protein
/// and water heavy atom — ions must use it too, or their LJ minima land
/// ~44% too close (the bug the v1.3.1 provenance audit caught).
/// Consequence of matching factors: an (ion, water-O) pair's LJ minimum in
/// the engine equals exactly R*_ion + R*_O, i.e. Amber's Rmin,ij rule.
pub const R_STAR_TO_SIGMA: f32 = 2.0 / 1.122_462_048_309_373;

/// One nonbonded ion model: ff type, element, mass, charge (engine-scaled),
/// LJ (**true σ**, kept as `r_star * R_STAR_TO_SIGMA` so the published
/// R_min/2 value is visible in the source), and the 12-6-4 induction
/// constant `c4` (0 = plain 12-6). Single source for counterions *and*
/// explicit salt of every valence.
///
/// All values below are the **OPC columns** of the Li/Merz ion series,
/// chosen so every nonbonded ion parameter shares one water model, one
/// combining convention, and one lab:
/// - Monovalent: Sengupta, Li, Song, Li & Merz, J. Chem. Inf. Model. 2021,
///   61, 869 (doi:10.1021/acs.jcim.0c01390), Table 2 (12-6, HFE-optimized).
///   A 12-6-4 variant also exists (their Table 4: Na 1.450/0.02545423/C4 0,
///   Cl 2.143/0.51564233/C4 −69) — left unused because the engine applies
///   anisotropic C4 to cation–water-O pairs only (negative anion C4 has no
///   consumer; revisit with a hydrogen-side path).
/// - Divalent: Li, Song, Li & Merz, J. Chem. Theory Comput. 2020, 16, 4429
///   (doi:10.1021/acs.jctc.0c00194), Table 5 (12-6-4). Their Eq. 5 defines
///   the pair term `U = ... − C4_MW/r⁴` for ion–water-oxygen; ion–H C4 = 0.
///   Against other solute atoms C4 now follows the PGY extension: pairs of
///   two c4-carrying atoms combine by geometric mean (see
///   `combine_lj_params`) — that is how Panteva–Giambasu–York site
///   corrections are meant to ride on the Li–Merz ions ([`panteva`]).
///
/// Since v1.3.2 this is the *atomic* member of the insertable-species family:
/// `IonParams: Insertable` normalizes into a single-site [`SpeciesSpec`]
/// (see the `species` module), so the same `add_salt`/`SaltSpec` channel
/// that takes these constants also takes multi-site species.
#[derive(Clone, Copy, Debug)]
pub struct IonParams {
    pub ff_type: &'static str,
    pub element: Element,
    pub mass: f32,
    pub charge_scaled: f32,
    pub sigma: f32,
    pub eps: f32,
    pub c4: f32,
}

pub const ION_NA: IonParams = IonParams {
    ff_type: "Na+",
    element: Element::Sodium,
    mass: 22.99,
    charge_scaled: CHARGE_UNIT_SCALER,
    sigma: 1.467 * R_STAR_TO_SIGMA,
    eps: 0.029_603_43,
    c4: 0.0,
};
pub const ION_CL: IonParams = IonParams {
    ff_type: "Cl-",
    element: Element::Chlorine,
    mass: 35.45,
    charge_scaled: -CHARGE_UNIT_SCALER,
    sigma: 2.360 * R_STAR_TO_SIGMA,
    eps: 0.678_788_7,
    c4: 0.0,
};
/// K⁺ — same Sengupta Table 2 OPC 12-6 HFE family as Na⁺/Cl⁻.
/// Provenance audit (2026-09-18): the paper's Erratum (doi:10.1021/
/// acs.jcim.1c00576) revises only Table 1's Rb⁺/Cs⁺ *coordination-number
/// targets* and one reference number — it touches no LJ parameter — so the
/// PMC-hosted Table 2 row (1.702 Å / 0.13953816 kcal/mol) is the shipped,
/// corrected value (author confirmed on the AMBER list that the frcmod
/// files rebase on this publication).
pub const ION_K: IonParams = IonParams {
    ff_type: "K+",
    element: Element::Potassium,
    mass: 39.0983,
    charge_scaled: CHARGE_UNIT_SCALER,
    sigma: 1.702 * R_STAR_TO_SIGMA,
    eps: 0.139_538_16,
    c4: 0.0,
};
pub const ION_MG: IonParams = IonParams {
    ff_type: "Mg2+",
    element: Element::Magnesium,
    mass: 24.305,
    charge_scaled: 2.0 * CHARGE_UNIT_SCALER,
    sigma: 1.405 * R_STAR_TO_SIGMA,
    eps: 0.016_529_39,
    c4: 127.0,
};
pub const ION_CA: IonParams = IonParams {
    ff_type: "Ca2+",
    element: Element::Calcium,
    mass: 40.078,
    charge_scaled: 2.0 * CHARGE_UNIT_SCALER,
    sigma: 1.602 * R_STAR_TO_SIGMA,
    eps: 0.080_342_31,
    c4: 86.0,
};
pub const ION_SR: IonParams = IonParams {
    ff_type: "Sr2+",
    // na_seq 0.3.15 (crates.io, latest published) has no Strontium variant;
    // `Other` is its documented catch-all. Identity is carried by ff_type
    // "Sr2+" — filter on that (not element) when counting Sr. Switch once
    // upstream na_seq adds Strontium; monitored passively, nothing here
    // blocks on it.
    element: Element::Other,
    mass: 87.62,
    charge_scaled: 2.0 * CHARGE_UNIT_SCALER,
    sigma: 1.738 * R_STAR_TO_SIGMA,
    eps: 0.165_002_96,
    c4: 87.0,
};
pub const ION_BA: IonParams = IonParams {
    ff_type: "Ba2+",
    element: Element::Barium,
    mass: 137.327,
    charge_scaled: 2.0 * CHARGE_UNIT_SCALER,
    sigma: 1.898 * R_STAR_TO_SIGMA,
    eps: 0.297_186_82,
    c4: 78.0,
};

/// Selectable divalent salts; each formula unit is electroneutral and
/// displaces that many whole waters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DivalentSalt {
    /// MgCl2: 1 Mg²⁺ + 2 Cl⁻ (3 waters displaced per formula unit)
    Mg,
    /// CaCl2
    Ca,
    /// SrCl2 — na_seq::Element has no Strontium variant, so the ion carries
    /// `Element::Other` and identity lives in its ff_type "Sr2+".
    Sr,
    /// BaCl2
    Ba,
}

impl DivalentSalt {
    pub fn cation(self) -> IonParams {
        match self {
            DivalentSalt::Mg => ION_MG,
            DivalentSalt::Ca => ION_CA,
            DivalentSalt::Sr => ION_SR,
            DivalentSalt::Ba => ION_BA,
        }
    }
}

/// Provenance lock for the ion table (v1.3.1). These tests fail if anyone
/// edits `ION_*` without re-deriving from the published Li–Merz/Sengupta
/// OPC tables, and they encode the r_star→σ semantics that the engine's
/// atom table uses (bio_files' conversion) — the convention bug the audit
/// caught is exactly what the Rmin-sum test below would have caught.
#[cfg(test)]
mod ion_provenance_tests {
    use super::{
        CHARGE_UNIT_SCALER, ION_BA, ION_CA, ION_CL, ION_K, ION_MG, ION_NA, ION_SR, IonParams,
        R_STAR_TO_SIGMA,
    };
    use crate::engine::md_core::solvent::{O_RSTAR, O_SIGMA};

    /// (label, params, published R_min/2, eps, c4) — Sengupta JCIM 2021
    /// Table 2 (Na, K, Cl) and Li-Merz JCTC 2020 Table 5 (divalent), OPC.
    const TABLE: &[(&str, IonParams, f32, f32, f32)] = &[
        ("Na+", ION_NA, 1.467, 0.029_603_43, 0.0),
        ("K+", ION_K, 1.702, 0.139_538_16, 0.0),
        ("Cl-", ION_CL, 2.360, 0.678_788_7, 0.0),
        ("Mg2+", ION_MG, 1.405, 0.016_529_39, 127.0),
        ("Ca2+", ION_CA, 1.602, 0.080_342_31, 86.0),
        ("Sr2+", ION_SR, 1.738, 0.165_002_96, 87.0),
        ("Ba2+", ION_BA, 1.898, 0.297_186_82, 78.0),
    ];

    #[test]
    fn sigma_is_exactly_published_rstar_scaled() {
        for (name, p, r, _, _) in TABLE {
            assert_eq!(p.sigma, r * R_STAR_TO_SIGMA, "{name} sigma drift");
        }
    }

    #[test]
    fn eps_and_c4_match_the_papers() {
        for (name, p, _, eps, c4) in TABLE {
            assert_eq!(p.eps, *eps, "{name} eps drift");
            assert_eq!(p.c4, *c4, "{name} c4 drift");
        }
        // Charges in units of e (engine-scaled): monovalent ±1, divalent +2.
        assert_eq!(ION_NA.charge_scaled, CHARGE_UNIT_SCALER);
        assert_eq!(ION_K.charge_scaled, CHARGE_UNIT_SCALER);
        assert_eq!(ION_CL.charge_scaled, -CHARGE_UNIT_SCALER);
        for i in [ION_MG, ION_CA, ION_SR, ION_BA] {
            assert_eq!(i.charge_scaled, 2.0 * CHARGE_UNIT_SCALER, "{}", i.ff_type);
        }
    }

    #[test]
    fn ion_water_lj_minimum_equals_amber_rmin_sum() {
        // The engine's σ-form pair minimum sits at 2^(1/6)·(σi+σj)/2; with
        // correct conversions this MUST equal Amber's Rmin,ij = R*_i + R*_j.
        let two_one_sixth = 1.122_462_048_309_373f32;
        for (name, p, r, _, _) in TABLE {
            let engine_min = two_one_sixth * (p.sigma + O_SIGMA) * 0.5;
            let amber_min = r + O_RSTAR;
            assert!(
                (engine_min - amber_min).abs() < 2e-4,
                "{name}: engine pair minimum {engine_min:.4} vs Amber Rmin,ij {amber_min:.4}"
            );
        }
    }
}

/// Minimum distance an ion must keep from every non-solvent atom (protein,
/// ligands, previously placed ions) — mirrors `gmx genion -rmin` (0.6 nm).
pub const ION_EXCLUSION_RADIUS_ANGSTROM: f32 = 6.0;

/// Fixed seed so ion sites — like everything else about a build — are
/// reproducible run to run (genion uses a fresh seed by default; we don't).
const ION_SHUFFLE_SEED: u64 = 0x5EED_1017;

/// Choose `count` distinct water indices to displace with ions, genion-style:
/// seeded random permutation of all waters, accepted in order if the water's
/// O sits at least [`ION_EXCLUSION_RADIUS_ANGSTROM`] (PBC min-image) from
/// every non-solvent atom. The previous fixed-stride walk laid ions on a
/// lattice-regular subset of the water array and allowed them to sit directly
/// on buried waters touching protein atoms. If the box is so salt-loaded that
/// too few sites honor the exclusion, the remainder is taken anyway with one
/// warning — a buildable system beats a fatal error.
///
/// This is a *placement-time* guarantee: the solvation-relaxation and
/// initial-minimization passes that follow deliberately let Coulomb
/// interactions pull ions toward (counter-ions) or away from (co-ions) the
/// protein — settling into solvation shells is the physics we want, and the
/// `rmin`-style floor only exists to keep the starting configuration from
/// burying an ion inside the solute.
///
/// Placement quality was assessed once (v1.3.5 audit) and ACCEPTED as-is;
/// three independent reasons, so the question need not be reopened:
/// (1) the protocol mirrors `gmx genion` (seeded shuffle + greedy 6 Å floor),
/// the placement used by the whole Amber/CHARMM user base;
/// (2) post-build relaxation, not the placement, is what forms solvation
/// shells — over-engineering slots would pre-judge the physics;
/// (3) any "smarter" in-place rearrangement would make the deterministic
/// insertion tail (tests/ion_layout_golden.rs) disagree with itself between
/// code paths, sacrificing the one bit-identity guard we have.
fn pick_ion_slots(state: &MdState, count: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..state.water.len()).collect();
    let mut rng = StdRng::seed_from_u64(ION_SHUFFLE_SEED);
    order.shuffle(&mut rng);

    let excl2 = ION_EXCLUSION_RADIUS_ANGSTROM * ION_EXCLUSION_RADIUS_ANGSTROM;
    let cell = &state.cell;
    let allowed = |&w: &usize| {
        let pos = state.water[w].o.posit;
        !state.atoms.iter().any(|a| {
            let d = cell.min_image(a.posit - pos);
            d.x * d.x + d.y * d.y + d.z * d.z < excl2
        })
    };

    // Greedy accept in shuffled order. Each site is checked against the
    // solute *and* every ion already taken this batch, so the exclusion is
    // pairwise-complete, not just batch-vs-solute.
    let mut chosen: Vec<(usize, lin_alg::f32::Vec3)> = Vec::with_capacity(count);
    for &w in &order {
        if chosen.len() == count {
            break;
        }
        let pos = state.water[w].o.posit;
        let clear_of_solute = allowed(&w);
        if clear_of_solute
            && chosen.iter().all(|(_, p)| {
                let d = cell.min_image(pos - *p);
                d.x * d.x + d.y * d.y + d.z * d.z >= excl2
            })
        {
            chosen.push((w, pos));
        }
    }
    if chosen.len() < count {
        let taken: std::collections::HashSet<usize> = chosen.iter().map(|(w, _)| *w).collect();
        eprintln!(
            "Ion placement: only {}/{} sites honor the {ION_EXCLUSION_RADIUS_ANGSTROM}\u{00c5} exclusion; \
             placing the rest at nearest available waters.",
            chosen.len(),
            count
        );
        for &w in &order {
            if chosen.len() == count {
                break;
            }
            if taken.contains(&w) || !allowed(&w) {
                continue;
            }
            chosen.push((w, state.water[w].o.posit));
        }
    }
    chosen.into_iter().map(|(w, _)| w).collect()
}

// The general insertable-species vocabulary (v1.3.2). `SiteSpec` replaced the
// old private-shape `CosolventSite` (kept as an alias below for source
// compatibility); single-atom ions normalize into the same shape via
// `SpeciesSpec::from_ion`, and `insert_instance` is now the ONE place that
// writes species atoms into the state.
pub use species::{
    Insertable, SaltBalanceError, SaltSpec, SiteSpec, SpeciesSpec, balance_stoichiometry,
};

/// A [`CosolventSpec`] molecule's site: offset from the displaced water's
/// oxygen, charge in engine-scaled units, LJ in the engine's true-σ
/// convention, and the per-site 12-6-4 `c4`.
pub type CosolventSite = SiteSpec;

/// A user-parameterized cosolute molecule (denaturant, osmolyte, …).
/// `bonds` are (site_i, site_j, k kcal·mol⁻¹·Å⁻², r0 Å): they drive BOTH the
/// harmonic connectivity — applied through `distance_restraints`, the
/// parameter-carrying mechanism — and the 1-2 nonbonded exclusions (via
/// `bonds_topology`), so internal site pairs do not feel full Coulomb/LJ.
/// A cosolute carries no net implicit-solvent trickery; charges must sum to
/// the intended molecular charge (the neutrality warning will shout if the
/// overall system ends up fractional).
#[derive(Clone, Debug, PartialEq)]
pub struct CosolventSpec {
    pub name: String,
    /// Formula-unit molarity, mol/L; count = round(c·V·N_A) molecules.
    pub molarity: f32,
    pub sites: Vec<CosolventSite>,
    /// Internal connectivity as (site_i, site_j, k, r0) — NOTE the force
    /// constant comes BEFORE the equilibrium distance (the order `
    /// add_cosolvent` destructures and the v1.3 PRB probe test established).
    /// Each bond becomes a harmonic distance restraint (E = ½kΔr²) and a 1-2
    /// topology exclusion. Sites absent from every bond (e.g. GdmCl's Cl⁻)
    /// float free within the molecule's slot.
    pub bonds: Vec<(usize, usize, f32, f32)>,
}

impl MdState {
    /// Displace waters with `n_mols` copies of `spec` at genion-style seeded,
    /// exclusion-respecting sites (each molecule occupies ONE water slot; its
    /// sites hang off that water's O by `offset`). Rebuilds LJ tables, the
    /// molecule-energy matrix, and thermo DOF, and runs the neutrality
    /// diagnostic — same contract as the salt paths.
    pub fn add_cosolvent(&mut self, spec: &CosolventSpec, n_mols: usize) {
        if n_mols == 0 || spec.sites.is_empty() {
            return;
        }
        // One normalized template, N instances (v1.3.2: this and every other
        // insertion path now share `insert_instance`; bond-index validation
        // moved there, so the failure message says "species").
        let sp = spec.to_species();
        let slots = pick_ion_slots(self, n_mols);
        let per_mol = sp.sites.len();
        let mut placed = 0usize;
        for &slot in &slots {
            insert_instance(self, &sp, slot);
            placed += 1;
        }
        if placed < n_mols {
            eprintln!(
                "Cosolvent {}: only {placed}/{} molecules placed (solvent exhausted).",
                spec.name, n_mols
            );
        }
        remove_waters(self, slots);
        refresh_mol_energy_matrix(self);
        self.setup_nonbonded_exclusion_scale_flags();
        self.lj_tables = LjTables::new(&self.atoms);
        self.thermo_dof = self.dof_for_thermo();
        eprintln!(
            "Added {placed} cosolute molecule(s) of {} ({} sites, {:.2} M).",
            spec.name, per_mol, spec.molarity
        );
        warn_if_not_neutral(self);
    }
}

/// PME/Ewald assumes an electroneutral cell; a nonzero net charge is silently
/// treated as a uniform neutralizing background (GROMACS refuses such systems,
/// LAMMPS warns). Integer counterions cannot cancel fractional *e* exactly when
/// terminal-residue charge units assume backbone hydrogens the internal digit
/// map never adds, so we surface the residue instead of aborting a scan build.
fn warn_if_not_neutral(state: &MdState) {
    let net = state.net_charge_e();
    if net.abs() > 0.01 {
        eprintln!(
            "Warning: system net charge {net:+.3} e after ion placement; \
             PME will apply its implicit neutralizing background."
        );
    }
}

/// Keep the dense molecule-energy analysis matrix coherent after the
/// molecule list changed (ions displaced waters).
fn refresh_mol_energy_matrix(state: &mut MdState) {
    let n_mols = state.mol_start_indices.len();
    if n_mols <= 256 && !state.potential_energy_between_mols.is_empty() {
        state
            .potential_energy_between_mols
            .resize(n_mols * n_mols, 0.0);
    } else {
        state.potential_energy_between_mols.clear();
    }
}

pub(crate) fn add_ions(state: &mut MdState, net_q_e: f32, n_ions: usize) {
    // Add counter-ions to neutralize any net charge.
    // Positive net → add Cl⁻;  negative net → add Na⁺.
    if n_ions > 0 && !state.water.is_empty() {
        let ion = if net_q_e > 0.0 { &ION_CL } else { &ION_NA };

        let w_indices = pick_ion_slots(state, n_ions);

        for &w_idx in &w_indices {
            insert_ion(
                state,
                w_idx,
                ion.ff_type,
                ion.element,
                ion.mass,
                ion.charge_scaled,
                ion.sigma,
                ion.eps,
                ion.c4,
            );
        }
        refresh_mol_energy_matrix(state);
        remove_waters(state, w_indices);

        eprintln!(
            "Added {n_ions} {} ion(s) to neutralize net charge ({net_q_e:+.3}e).",
            ion.ff_type
        );
    } else if n_ions > 0 {
        eprintln!(
            "Warning: net charge {net_q_e:+.3}e detected but no solvent available to \
                 displace; skipping ion insertion."
        );
    }
}

impl MdState {
    /// Rebuild the derived per-atom tables after external atom-list surgery
    /// (the reuse-path ion repack strips ions before re-inserting them):
    /// LJ pair tables and thermostat DOF, identical to what the insertion
    /// helpers do at their tail (v1.3.2 single-refresh contract).
    pub fn refresh_species_tables(&mut self) {
        self.mass_accel_factor = self.atoms.iter().map(|a| KCAL_TO_NATIVE / a.mass).collect();
        self.lj_tables = LjTables::new(&self.atoms);
        self.thermo_dof = self.dof_for_thermo();
    }

    /// Add `n_pairs` Na⁺/Cl⁻ ion pairs to reach a target ionic strength, each pair
    /// displacing a water molecule. Rebuilds the LJ tables and thermo DOF so the new
    /// ions are fully accounted for. Safe to call again later (e.g. to change salt).
    pub fn add_salt_ions(&mut self, n_pairs: usize) {
        if n_pairs == 0 || self.water.is_empty() {
            return;
        }

        // One pick+insert+remove pass inside the helper (alternating
        // Na/Cl over the seeded shuffled order keeps every pair neutral).
        self.insert_formula_unit_salts(&ION_NA, &[ION_CL], n_pairs);
        self.lj_tables = LjTables::new(&self.atoms);
        self.thermo_dof = self.dof_for_thermo();
        eprintln!("Added {n_pairs} Na⁺/Cl⁻ pair(s) for ionic strength.");
        warn_if_not_neutral(self);
    }

    /// Insert `n_units` electroneutral formula units `cation + anions` by
    /// displacing whole waters at genion-style exclusion-respecting sites.
    /// Every call draws from the same seeded shuffle, but sites already used
    /// by previously inserted ions fail the exclusion and are skipped — so
    /// repeated calls never collide. Caller must refresh `lj_tables` (the
    /// batched env update in `add_divalent_salt` does).
    pub(crate) fn insert_formula_unit_salts<C: Insertable, A: Insertable>(
        &mut self,
        cation: &C,
        anions: &[A],
        n_units: usize,
    ) {
        // v1.3.2: normalize to species, then expand one formula unit in the
        // exact [cation, anions…] order the old `.cycle()` produced — the
        // slot draw, instance↔slot mapping and atom append order are
        // bit-identical to v1.3.1.
        let csp = cation.to_species();
        let asps: Vec<SpeciesSpec> = anions.iter().map(|a| a.to_species()).collect();
        let parts: Vec<&SpeciesSpec> = std::iter::once(&csp).chain(asps.iter()).collect();
        self.insert_formula_unit(&parts, n_units);
    }

    /// Insert `n_units` copies of an expanded formula unit: `parts` is the
    /// instance order *within one unit* (MgCl₂ → [Mg²⁺, Cl⁻, Cl⁻]), each
    /// instance occupying its own displaced water slot. Multi-site parts
    /// (a polyatomic anion, someday) hang their extra sites off their own
    /// slot's oxygen — same contract as `add_cosolvent`.
    /// Caller refreshes `lj_tables`/`thermo_dof`, as the salt paths always
    /// have.
    pub(crate) fn insert_formula_unit(&mut self, parts: &[&SpeciesSpec], n_units: usize) {
        let total = parts.len() * n_units;
        let w_indices = pick_ion_slots(self, total);
        let mut iter = parts.iter().copied().cycle().take(total);
        for &w_idx in &w_indices {
            insert_instance(self, iter.next().unwrap(), w_idx);
        }
        remove_waters(self, w_indices);
        refresh_mol_energy_matrix(self);
    }

    /// Insert `n_units` formula units of any balanced electrolyte (v1.3.2
    /// species channel — KCl, CsCl, a future Na₂SO₄, …). The per-unit
    /// instance order comes straight from [`SaltSpec::formula_unit_parts`];
    /// a pair that cannot be balanced (two anions, junk charges) is skipped
    /// with a warning rather than panicking a long scan build. Same tail
    /// contract as `add_divalent_salt` (tables, DOF, neutrality note).
    pub fn add_salt(&mut self, salt: &SaltSpec, n_units: usize) {
        if n_units == 0 {
            return;
        }
        let parts = match salt.formula_unit_parts() {
            Ok(parts) => parts,
            Err(e) => {
                eprintln!("Salt {}/{} skipped: {e}", salt.cation.name, salt.anion.name);
                return;
            }
        };
        self.insert_formula_unit(&parts, n_units);
        self.lj_tables = LjTables::new(&self.atoms);
        self.thermo_dof = self.dof_for_thermo();
        let (nc, na) = salt
            .stoichiometry()
            .unwrap_or((0, 0)) // parts succeeded above, so this cannot fail
            ;
        eprintln!(
            "Added {n_units} {}/{} formula unit(s) ({nc} cation + {na} anion instances).",
            salt.cation.name, salt.anion.name
        );
        warn_if_not_neutral(self);
    }

    /// Add `n_units` formula units (e.g. MgCl2) of a divalent salt, replacing
    /// waters. Table 5 (Li-Merz 2020, OPC column) values; see `IonParams`.
    pub fn add_divalent_salt(&mut self, salt: DivalentSalt, n_units: usize) {
        if n_units == 0 {
            return;
        }
        let cation = salt.cation();
        self.insert_formula_unit_salts(&cation, &[ION_CL, ION_CL], n_units);
        self.lj_tables = LjTables::new(&self.atoms);
        self.thermo_dof = self.dof_for_thermo();
        eprintln!(
            "Added {n_units} {}Cl2 formula unit(s) (1 {}, 2 Cl⁻ each).",
            cation.ff_type.trim_end_matches('+'),
            cation.ff_type
        );
        warn_if_not_neutral(self);
    }

    /// Net charge of the whole system in units of *e*: every solute/ion atom
    /// plus every explicit water site charge (OPC puts the charge at the M/EP
    /// site and the hydrogens; O itself is chargeless by design).
    pub fn net_charge_e(&self) -> f32 {
        let mut q = 0.0_f64;
        for a in &self.atoms {
            q += f64::from(a.partial_charge);
        }
        for w in &self.water {
            q += f64::from(w.o.partial_charge)
                + f64::from(w.h0.partial_charge)
                + f64::from(w.h1.partial_charge)
                + f64::from(w.m.partial_charge);
        }
        (q / f64::from(CHARGE_UNIT_SCALER)) as f32
    }

    /// Effective ionic strength of the built box, I = ½ Σᵢ cᵢ zᵢ² (mol/L), summed
    /// over every explicit salt ion present (monovalent Na⁺/K⁺/Cl⁻, divalent
    /// Mg²⁺/Ca²⁺/Sr²⁺/Ba²⁺, and any `salts_json` electrolyte whose site is a
    /// registered ion).
    ///
    /// This is DIFFERENT from `EnvParams::ionic_strength_m`, which only sets the
    /// NaCl background molarity — adding divalent salts or extra electrolytes
    /// raises the true I above that single number. Ions are matched by
    /// `force_field_type` (NOT element: Sr²⁺ is `Element::Other` upstream), and
    /// z is the ion's formal valence (|partial_charge| in e, rounded). A cosolute
    /// site that is itself a free chloride (e.g. GdmCl) carries the `Cl-` type and
    /// so correctly contributes to I.
    pub fn effective_ionic_strength_m(&self) -> f32 {
        let vol_l = f64::from(self.cell.volume()) * 1.0e-27;
        if vol_l <= 0.0 {
            return 0.0;
        }
        // Concentration of a single particle of one species: 1 / (V_L · N_A).
        let c_one = 1.0 / (vol_l * AVOGADRO);
        let mut i = 0.0_f64;
        for a in &self.atoms {
            let is_ion = matches!(
                a.force_field_type.as_str(),
                "Na+" | "K+" | "Cl-" | "Mg2+" | "Ca2+" | "Sr2+" | "Ba2+"
            );
            if !is_ion {
                continue;
            }
            let z = (f64::from(a.partial_charge) / f64::from(CHARGE_UNIT_SCALER))
                .round()
                .abs();
            if z > 0.5 {
                i += 0.5 * c_one * z * z;
            }
        }
        i as f32
    }

    // ------------------------------------------------------------------
    // Observability probes (analysis module). Read-only: none of these
    // mutate state or touch the MD hot path, so golden byte-identity and
    // the determinism of runs are unaffected.
    // ------------------------------------------------------------------

    /// Electrostatic potential at arbitrary probe points, in kcal/mol/e,
    /// using the engine's OWN PME conventions: the charge list is the
    /// production packer (`pack_pme_pos_q`: atoms, then water M/H0/H1), the
    /// real-space term is `erfc(alpha r)/r` minimum-image inside the Coulomb
    /// cutoff, the reciprocal term is the same half-spectrum mesh sum with
    /// the exact B-spline deconvolution (validated against the ewald fork's
    /// reciprocal energy in `analysis::tests`).
    ///
    /// Report POTENTIAL DIFFERENCES or values relative to a reference point.
    /// The PME constant offset (self-energy + neutralizing background) is
    /// position-independent and cancels in any difference; absolute values
    /// carry the arbitrary gauge of the mesh, so they are not meaningful.
    /// Catalytic-axis field strength = (phi(b) - phi(a)) / |b - a|; anion-hole
    /// potential and pH-dependent site potential are single points here.
    pub fn electrostatic_potential(&self, points: &[[f64; 3]]) -> Vec<f64> {
        let (pos, q) = self.pack_pme_pos_q();
        analysis::electrostatic_potential(
            points,
            &pos,
            &q,
            self.cell.extent,
            f64::from(self.cfg.spme_alpha),
            f64::from(self.cfg.coulomb_cutoff),
            f64::from(self.cfg.spme_mesh_spacing),
        )
    }

    /// Electrostatic FIELD **E = −∇φ** (kcal/mol·e⁻¹·Å⁻¹) at probe points, one PME
    /// pass (real erfc-gradient + reciprocal k-gradient) — see
    /// `analysis::electrostatic_field`. This is the B2 functional-probe primitive
    /// (the site's catalytic field), computed ANALYTICALLY so it does NOT inherit
    /// the subtractive-cancellation noise of finite-differencing the gauge-
    /// wandering potential.
    ///
    /// `positions = Some(p)` feeds a *time-averaged structure* (length must equal
    /// `pme_positions()`) so thermal motion averages out: snapshot `pme_positions`
    /// over a window, take the running mean, pass it back here. `None` uses the
    /// current frame.
    pub fn electrostatic_field(
        &self,
        points: &[[f64; 3]],
        positions: Option<&[[f64; 3]]>,
    ) -> Vec<[f64; 3]> {
        let (base_pos, q) = self.pack_pme_pos_q();
        let owned: Vec<Vec3>;
        let pos: &[Vec3] = match positions {
            Some(ov) => {
                assert_eq!(
                    ov.len(),
                    base_pos.len(),
                    "electrostatic_field positions override length {} != pme particles {}",
                    ov.len(),
                    base_pos.len()
                );
                owned = ov
                    .iter()
                    .map(|p| Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32))
                    .collect();
                &owned
            }
            None => &base_pos,
        };
        analysis::electrostatic_field(
            points,
            pos,
            &q,
            self.cell.extent,
            f64::from(self.cfg.spme_alpha),
            f64::from(self.cfg.coulomb_cutoff),
            f64::from(self.cfg.spme_mesh_spacing),
        )
    }

    /// PME particle positions (the SAME order/index space as
    /// `electrostatic_field`'s `positions` override): atoms + water (O/M/H) +
    /// ions/cosolvent. Snapshot these across a trajectory window and average to
    /// build a time-averaged structure for the site-field query.
    pub fn pme_positions(&self) -> Vec<[f64; 3]> {
        let (pos, _q) = self.pack_pme_pos_q();
        pos.iter()
            .map(|p| [p.x as f64, p.y as f64, p.z as f64])
            .collect()
    }

    /// Site layout WITH explicit solvent, for occlusion-style queries: all
    /// non-solvent atoms first (indices `0..atoms.len()`, the same index
    /// space as `SpiceEngine::select_atoms`), then per water its O, H0, H1.
    /// The M/EP site is never included: it is a virtual charge placement
    /// with no LJ and no vdW radius. Used by `bottleneck_profile` with
    /// `include_water=true` (a hydrated pore radius really does have water
    /// lining it); NOT used by SASA (see `atom_sasa`).
    pub fn real_sites_pos_radii(&self) -> (Vec<Vec3>, Vec<f64>) {
        let mut pos = Vec::with_capacity(self.atoms.len() + 3 * self.water.len());
        let mut radii = Vec::with_capacity(pos.capacity());
        for a in &self.atoms {
            pos.push(a.posit);
            radii.push(analysis::vdw_radius(&a.element, f64::from(a.lj_sigma)));
        }
        for w in &self.water {
            for site in [&w.o, &w.h0, &w.h1] {
                pos.push(site.posit);
                radii.push(analysis::vdw_radius(
                    &site.element,
                    f64::from(site.lj_sigma),
                ));
            }
        }
        (pos, radii)
    }

    /// Per-atom solvent-accessible surface area (Angstrom^2, Shrake-Rupley
    /// with `n_sphere` Fibonacci lattice points): exactly one value per
    /// `state.atoms` entry, the same index space as `select_atoms`. Summing
    /// over a selection gives MM/PBSA nonpolar terms and per-site burial;
    /// core atoms come out near zero, exposed surface atoms near 4 pi r^2.
    ///
    /// Explicit water is deliberately NOT part of this calculation, as both
    /// target and occluder. The 1.4 A probe IS the solvent model: a surface
    /// atom already loses the cap the probe cannot reach. Feeding the first
    /// hydration shell in as occluder spheres instead measures the
    /// solute-water CONTACT area, which is nearly zero for a solvated
    /// protein (this collapsed 2LYZ from ~1.8e4 A^2 to ~30 A^2 when tried; `tests/observables.rs` prints the kept side).
    /// Ions/cosolvent sites live in `state.atoms` and do occlude, matching
    /// an "all non-solvent atoms" SASA from standard tools.
    pub fn atom_sasa(&self, probe: f64, n_sphere: usize) -> Vec<f64> {
        let pos: Vec<Vec3> = self.atoms.iter().map(|a| a.posit).collect();
        let radii: Vec<f64> = self
            .atoms
            .iter()
            .map(|a| analysis::vdw_radius(&a.element, f64::from(a.lj_sigma)))
            .collect();
        analysis::atom_sasa(&pos, &radii, self.cell.extent, probe, n_sphere.max(50))
    }
}

#[cfg(test)]
mod efield_tests {
    use super::*;

    #[test]
    fn efield_term_matches_manual_dipole_energy() {
        // One +1 e charge at x=2 with field (E0,0,0): F = +qE·x̂, U = −q·E·x.
        let raw = 0.5_f32; // kcal·mol⁻¹·e⁻¹·Å⁻¹
        let e = [raw / CHARGE_UNIT_SCALER, 0.0, 0.0];
        let q = CHARGE_UNIT_SCALER; // +1 e in engine-scaled units
        let (f_pos, u_pos) = efield_term(q, Vec3::new(2.0, 0.0, 0.0), e);
        assert!((f_pos.x - raw).abs() < 1e-6, "force must equal q_e·E");
        assert!(f_pos.y == 0.0 && f_pos.z == 0.0);
        assert!((u_pos + 1.0).abs() < 1e-6, "U = −(+1)(0.5)(2) = −1");
    }

    #[test]
    fn efield_net_force_on_neutral_pair_is_zero() {
        let e = [0.3 / CHARGE_UNIT_SCALER, -0.1 / CHARGE_UNIT_SCALER, 0.0];
        let (f1, _) = efield_term(CHARGE_UNIT_SCALER, Vec3::new(1.0, 2.0, 3.0), e);
        let (f2, _) = efield_term(-CHARGE_UNIT_SCALER, Vec3::new(-1.0, 0.0, 4.0), e);
        assert!((f1 + f2).magnitude() < 1e-6);
    }
}
