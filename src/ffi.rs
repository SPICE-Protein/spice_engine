//! Python bindings (PyO3 + rust-numpy) — the P4 FFI surface.
//!
//! Python side feeds in-memory atom arrays (`PyStructure`) built from its data
//! pipeline (`spice_protein/` Parquet → numpy), then drives `PyEngine`:
//! step (optional bias-force action), metrics (five-dimensional `M`),
//! pseudo-labels (time-averaged Cα), reset / set_temperature.
//!
//! Build with maturin: `maturin develop --features python` (or `--release`).

use std::sync::OnceLock;

use na_seq::Element;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::actions::ForceAction;
use crate::builder::BuildOptions;
use crate::engine::SpiceEngine;
use crate::env::EnvParams;
use crate::equilibrate::EquilConfig;
use crate::metrics::{Metrics, MetricsConfig};
use crate::structure::{AtomInput, StructureInput, build_from_input};

/// Lazily loaded Amber parameter set (load once per process).
fn param_set() -> &'static dynamics::params::FfParamSet {
    static PS: OnceLock<dynamics::params::FfParamSet> = OnceLock::new();
    PS.get_or_init(|| dynamics::params::FfParamSet::new_amber().expect("load amber params"))
}

fn err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyValueError::new_err(msg.into()))
}

/// In-memory molecular structure (atom arrays), fed by Python.
#[pyclass(name = "Structure", module = "spice_engine", skip_from_py_object)]
#[derive(Clone)]
pub struct PyStructure {
    pub inner: StructureInput,
}

#[pymethods]
impl PyStructure {
    #[new]
    fn new() -> Self {
        Self {
            inner: StructureInput::default(),
        }
    }

    /// Build from parallel numpy arrays (all length N):
    ///   atom_names: [N] str   (e.g. "CA", "N", "O", "CB")
    ///   elements:   [N] str   (e.g. "C", "N", "O", "S")
    ///   res_seq:    [N] int
    ///   res_names:  [N] str   (3-letter, e.g. "ALA")
    ///   coords:     [N, 3] f32  (Å)
    ///   occupancy:  [N] f32 (optional; default 1.0). MUST be provided for
    ///               altloc-heavy structures — `dedup_altloc` keeps the
    ///               highest-occupancy conformer; dropping occupancy mixes
    ///               overlapping conformers (hard clash that blows up MD).
    #[staticmethod]
    #[pyo3(signature = (atom_names, elements, res_seq, res_names, coords, occupancy=None))]
    fn from_atoms(
        atom_names: Vec<String>,
        elements: Vec<String>,
        res_seq: Vec<i32>,
        res_names: Vec<String>,
        coords: PyReadonlyArray2<'_, f32>,
        occupancy: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<Self> {
        let shape = coords.as_array().shape().to_vec();
        if shape[1] != 3 {
            return err(format!("coords must be [N,3], got {shape:?}"));
        }
        let n = shape[0];
        if atom_names.len() != n
            || elements.len() != n
            || res_seq.len() != n
            || res_names.len() != n
        {
            return err(format!("array length mismatch: atoms={} elms={} res_seq={} res_names={} coords={n}", atom_names.len(), elements.len(), res_seq.len(), res_names.len()));
        }
        let occ = occupancy.as_ref().map(|o| {
            let a = o.as_array();
            if a.len() != n {
                return Err(format!("occupancy length {} != coords {n}", a.len()));
            }
            Ok(a.iter().copied().collect::<Vec<f32>>())
        });
        let occ = match occ {
            Some(Ok(v)) => v,
            Some(Err(e)) => return err(e),
            None => vec![1.0f32; n],
        };
        let c = coords.as_array();
        let mut input = StructureInput::default();
        for i in 0..n {
            let element = na_seq::Element::from_letter(&elements[i])
                .map_err(|_| PyValueError::new_err(format!("bad element '{}'", elements[i])))?;
            input.push(AtomInput {
                chain_id: "A".to_string(),
                res_seq: res_seq[i],
                res_name: res_names[i].clone(),
                atom_name: atom_names[i].clone(),
                element,
                x: c[[i, 0]],
                y: c[[i, 1]],
                z: c[[i, 2]],
                occupancy: occ[i],
            });
        }
        Ok(Self { inner: input })
    }

    /// Convenience: load an mmCIF from disk (testing / ad-hoc; the production
    /// path is Python's own pipeline feeding `from_atoms`).
    #[staticmethod]
    fn from_mmcif(path: &str) -> PyResult<Self> {
        let mm = bio_files::MmCif::load(std::path::Path::new(path))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut input = StructureInput::default();
        for r in &mm.residues {
            if matches!(r.res_type, bio_files::ResidueType::Water) {
                continue;
            }
            let res_name = match &r.res_type {
                bio_files::ResidueType::AminoAcid(aa) => {
                    aa.to_str(na_seq::AaIdent::ThreeLetters).to_string()
                }
                _ => continue,
            };
            for sn in &r.atom_sns {
                let Some(a) = mm.atoms.iter().find(|a| &a.serial_number == sn) else {
                    continue;
                };
                input.push(AtomInput {
                    chain_id: "A".to_string(),
                    res_seq: r.serial_number as i32,
                    res_name: res_name.clone(),
                    atom_name: a
                        .type_in_res
                        .as_ref()
                        .map(|t| t.to_string())
                        .or_else(|| a.type_in_res_general.clone())
                        .unwrap_or_default(),
                    element: a.element,
                    x: a.posit.x as f32,
                    y: a.posit.y as f32,
                    z: a.posit.z as f32,
                    occupancy: a.occupancy.unwrap_or(1.0),
                });
            }
        }
        Ok(Self { inner: input })
    }

    fn sequence(&self) -> PyResult<String> {
        self.inner.sequence().map_err(|e| PyValueError::new_err(e))
    }

    fn residue_count(&self) -> usize {
        self.inner.residue_count()
    }
}

/// The MD engine wrapper exposed to Python.
#[pyclass(name = "Engine", module = "spice_engine")]
pub struct PyEngine {
    pub engine: SpiceEngine,
    pub force: ForceAction,
    pub metrics: Metrics,
}

#[pymethods]
impl PyEngine {
    /// Build a system from an in-memory `Structure` plus environment.
    ///
    /// `strict_incomplete=True` (default) rejects structures with residues
    /// missing their sidechain heavy atoms (disordered crystal sidechains),
    /// failing with a clear error listing them. Set `False` to build them
    /// truncated (backbone + Cα H only, warning) — physics is wrong for those
    /// residues, so only use it to explore.
    #[staticmethod]
    #[pyo3(signature = (structure, ph, temp, pressure, ionic_strength_m, relax_iters, tolerance, strict_incomplete = true))]
    fn build(
        structure: &Bound<'_, PyStructure>,
        ph: f32,
        temp: f32,
        pressure: f32,
        ionic_strength_m: f32,
        relax_iters: usize,
        tolerance: f32,
        strict_incomplete: bool,
    ) -> PyResult<Self> {
        let dev = dynamics::ComputationDevice::Cpu;
        let opts = BuildOptions {
            env: EnvParams::new(ph, temp, pressure, ionic_strength_m),
            relax_iters: Some(relax_iters),
            energy_minimization_tolerance: tolerance,
            strict_incomplete_residues: strict_incomplete,
            ..Default::default()
        };
        let structure = structure.borrow();
        let engine = build_from_input(&dev, param_set(), &structure.inner, &opts)
            .map_err(|e| PyValueError::new_err(e))?;
        let n_res = engine.topology.sequence.len();
        let metrics = Metrics::new(&engine, MetricsConfig::default());
        let force = ForceAction::new(n_res, 16, 0.5, 20);
        Ok(Self {
            engine,
            force,
            metrics,
        })
    }

    /// Create a new engine with a mutated structure by reusing the solvent box and ions
    /// of this engine. This is extremely fast (<0.5 seconds vs 30 seconds for complete build)
    /// and avoids CPU thermal throttling.
    #[pyo3(signature = (structure, ph, temp, pressure, ionic_strength_m, relax_iters, tolerance, strict_incomplete = true))]
    fn mutate_with_solvent_reuse(
        &self,
        structure: &Bound<'_, PyStructure>,
        ph: f32,
        temp: f32,
        pressure: f32,
        ionic_strength_m: f32,
        relax_iters: usize,
        tolerance: f32,
        strict_incomplete: bool,
    ) -> PyResult<Self> {
        let opts = BuildOptions {
            env: EnvParams::new(ph, temp, pressure, ionic_strength_m),
            relax_iters: Some(relax_iters),
            energy_minimization_tolerance: tolerance,
            strict_incomplete_residues: strict_incomplete,
            ..Default::default()
        };
        let structure = structure.borrow();
        let engine = crate::builder::build_mutant_by_solvent_reuse(&self.engine, param_set(), &structure.inner, &opts)
            .map_err(|e| PyValueError::new_err(e))?;
        let n_res = engine.topology.sequence.len();
        let metrics = Metrics::new(&engine, MetricsConfig::default());
        let force = ForceAction::new(n_res, 16, 0.5, 20);
        Ok(Self {
            engine,
            force,
            metrics,
        })
    }

    /// Advance one step. `action` is an optional `[M=16]` bias-force coefficient
    /// vector; `None` runs unbiased. Returns a dict with U, Cα coords, step/time,
    /// crash flag and the five metrics.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.step_impl(py, action, true)
    }

    /// Advance one MD step WITHOUT computing the five metrics. The full-metric
    /// `step` spends ~20-90 ms/step in O(N²) clash + surface + DSSP-lite work;
    /// `step_md` is just the integrator (~ms/step). Use this in tight loops
    /// (benchmarks, stability scans, long production runs) and call `metrics()`
    /// at checkpoints. Returns U, coords, step/time and crash flag only.
    fn step_md<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.step_impl(py, None, false)
    }

    /// The five physical metrics at the current state.
    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let m = self.metrics.compute(&self.engine);
        let d = PyDict::new(py);
        d.set_item("m1", m.m1)?;
        d.set_item("m2", m.m2)?;
        d.set_item("m3", m.m3)?;
        d.set_item("m4", m.m4)?;
        d.set_item("m5", m.m5)?;
        d.set_item("rg", m.rg)?;
        d.set_item("u_t_kcal", m.u_t_kcal)?;
        d.set_item("n_ss_ref", m.n_ss_ref)?;
        d.set_item("n_ss_kept", m.n_ss_kept)?;
        d.set_item("n_surface_charged", m.n_surface_charged)?;
        d.set_item("stability_margin", m.stability_margin)?;
        d.set_item("rmsf", m.rmsf)?;
        Ok(d)
    }

    /// Time-averaged Cα coordinates `[L, 3]` (the pseudo-label source).
    fn pseudo_labels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let avg: Vec<Vec<f32>> = self
            .engine
            .time_averaged_ca()
            .iter()
            .map(|c| vec![c[0], c[1], c[2]])
            .collect();
        PyArray2::from_vec2(py, &avg).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Current Cα coordinates `[L, 3]`.
    fn coords_ca<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let coords: Vec<Vec<f32>> = self
            .topology()
            .ca_indices
            .iter()
            .map(|&i| {
                let p = self.engine.state.atoms[i].posit;
                vec![p.x, p.y, p.z]
            })
            .collect();
        PyArray2::from_vec2(py, &coords).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn sequence(&self) -> String {
        self.topology().sequence.clone()
    }

    fn n_residues(&self) -> usize {
        self.topology().sequence.len()
    }

    fn u_t_kcal(&self) -> f64 {
        self.engine.state.potential_energy
    }

    /// Diagnostic: exact DOF / temperature bookkeeping. Returns atom & water
    /// counts vs the cached `thermo_dof`, and the temperature it implies for
    /// the current kinetic energy — lets us check whether `t_kin` is
    /// miscalibrated (e.g. thermo_dof cached before H's/ions were finalized).
    fn thermo_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        const R_KCAL: f64 = 0.001_987_204_1;
        let s = &self.engine.state;
        let n_atoms = s.atoms.len();
        let n_static = s.atoms.iter().filter(|a| a.static_).count();
        let n_h = s
            .atoms
            .iter()
            .filter(|a| a.element == Element::Hydrogen && !a.static_)
            .count();
        let n_water = s.water.len();
        let thermo_dof = s.thermo_dof();
        let dof_now = s.dof_for_thermo_now();
        let ke = s.kinetic_energy;
        let t_implied = if thermo_dof > 0 {
            2.0 * ke / (thermo_dof as f64 * R_KCAL)
        } else {
            0.0
        };
        let d = PyDict::new(py);
        d.set_item("n_atoms", n_atoms)?;
        d.set_item("n_static", n_static)?;
        d.set_item("n_hydrogens", n_h)?;
        d.set_item("n_water", n_water)?;
        d.set_item("thermo_dof", thermo_dof)?;
        d.set_item("dof_for_thermo_now", dof_now)?;
        d.set_item("dof_water_6n", 6 * n_water)?;
        d.set_item("dof_solute_3n", 3 * (n_atoms - n_static))?;
        d.set_item("kinetic_energy_kcal", ke)?;
        d.set_item("t_implied_k", t_implied)?;
        Ok(d)
    }

    /// Per-species temperature split (solute vs water), for thermostat
    /// calibration: tells us WHICH species the thermostat over-heats. DOF:
    /// solute = 3·non-static atoms, water = 6·n_water (rigid). KE in kcal/mol.
    fn species_temperatures<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        const NATIVE_TO_KCAL: f64 = 1.0 / 418.4;
        const R_KCAL: f64 = 0.001_987_204_1;
        let s = &self.engine.state;

        let mut solute_ke_native = 0.0f64;
        let mut n_solute = 0usize;
        let mut n_solute_h = 0usize;
        for a in &s.atoms {
            if a.static_ {
                continue;
            }
            n_solute += 1;
            if a.element == Element::Hydrogen {
                n_solute_h += 1;
            }
            let v2 = a.vel.magnitude_squared() as f64;
            solute_ke_native += (a.mass as f64) * v2;
        }
        let solute_ke = 0.5 * solute_ke_native * NATIVE_TO_KCAL;

        let mut water_ke_native = 0.0f64;
        for w in &s.water {
            for atom in [&w.o, &w.h0, &w.h1] {
                let v2 = atom.vel.magnitude_squared() as f64;
                water_ke_native += (atom.mass as f64) * v2;
            }
        }
        let water_ke = 0.5 * water_ke_native * NATIVE_TO_KCAL;

        // Solute DOF: 3 per non-static atom, minus 1 per constrained H (LINCS/SHAKE
        // both remove one H–heavy-bond DOF; rattle projects the bond velocity before
        // the KE is measured). Mirrors dynamics' `dof_for_thermo`.
        let solute_dof = (3 * n_solute - n_solute_h) as f64;
        let water_dof = (6 * s.water.len()) as f64;

        let d = PyDict::new(py);
        d.set_item("solute_ke_kcal", solute_ke)?;
        d.set_item("water_ke_kcal", water_ke)?;
        d.set_item("solute_dof", solute_dof)?;
        d.set_item("water_dof", water_dof)?;
        d.set_item(
            "solute_t_k",
            if solute_dof > 0.0 {
                2.0 * solute_ke / (solute_dof * R_KCAL)
            } else {
                0.0
            },
        )?;
        d.set_item(
            "water_t_k",
            if water_dof > 0.0 {
                2.0 * water_ke / (water_dof * R_KCAL)
            } else {
                0.0
            },
        )?;
        Ok(d)
    }

    /// Diagnostic: split the water kinetic energy into rigid-body (COM
    /// translation + rotation) vs internal (bond-stretch / angle) parts. If the
    /// internal part is significant, SETTLE is NOT removing the water's internal
    /// DOF — then `water_t` (computed with 6 DOF) is inflated ~1.5× and the real
    /// system temperature is LOWER than reported (thermostat under-injecting).
    fn water_rigid_split<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        use lin_alg::f32::{Mat3 as Mat3F32, Vec3};
        const NATIVE_TO_KCAL: f64 = 1.0 / 418.4;
        const R_KCAL: f64 = 0.001_987_204_1;
        let s = &self.engine.state;

        let mut total_accum = 0.0f64; // Σ m·v² over all 9 components (no ½)
        let mut rigid_accum = 0.0f64; // M·V² + L·ω  (no ½)
        let mut n_water = 0usize;
        for w in &s.water {
            n_water += 1;
            let m_total = w.o.mass + w.h0.mass + w.h1.mass;
            let r_com = (w.o.posit * w.o.mass + w.h0.posit * w.h0.mass + w.h1.posit * w.h1.mass)
                / m_total;
            let v_com = (w.o.vel * w.o.mass + w.h0.vel * w.h0.mass + w.h1.vel * w.h1.mass)
                / m_total;

            for atom in [&w.o, &w.h0, &w.h1] {
                let v2 = atom.vel.magnitude_squared() as f64;
                total_accum += (atom.mass as f64) * v2;
            }

            let (r_o, r_h0, r_h1) = (w.o.posit - r_com, w.h0.posit - r_com, w.h1.posit - r_com);
            let (v_o, v_h0, v_h1) = (w.o.vel - v_com, w.h0.vel - v_com, w.h1.vel - v_com);
            let l = r_o.cross(v_o) * w.o.mass
                + r_h0.cross(v_h0) * w.h0.mass
                + r_h1.cross(v_h1) * w.h1.mass;

            let inertia = |r: Vec3, mass: f32| {
                let r2 = r.dot(r);
                [
                    [mass * (r2 - r.x * r.x), -mass * r.x * r.y, -mass * r.x * r.z],
                    [-mass * r.y * r.x, mass * (r2 - r.y * r.y), -mass * r.y * r.z],
                    [-mass * r.z * r.x, -mass * r.z * r.y, mass * (r2 - r.z * r.z)],
                ]
            };
            let mut i_arr = inertia(r_o, w.o.mass);
            for add in [inertia(r_h0, w.h0.mass), inertia(r_h1, w.h1.mass)] {
                for i in 0..3 {
                    for j in 0..3 {
                        i_arr[i][j] += add[i][j];
                    }
                }
            }
            let i_mat = Mat3F32::from_arr(i_arr);
            let omega = i_mat.solve_system(l); // ω = I⁻¹L

            rigid_accum += (m_total as f64) * (v_com.magnitude_squared() as f64)
                + l.dot(omega) as f64; // M·V² + L·ω = 2·(COM_KE + rot_KE)
        }

        let total_ke = 0.5 * total_accum * NATIVE_TO_KCAL;
        let rigid_ke = 0.5 * rigid_accum * NATIVE_TO_KCAL;
        let internal_ke = total_ke - rigid_ke;

        let d = PyDict::new(py);
        d.set_item("water_total_ke_kcal", total_ke)?;
        d.set_item("water_rigid_ke_kcal", rigid_ke)?;
        d.set_item("water_internal_ke_kcal", internal_ke)?;
        d.set_item("n_water", n_water)?;
        d.set_item(
            "water_rigid_t_k",
            if n_water > 0 {
                2.0 * rigid_ke / ((6 * n_water) as f64 * R_KCAL)
            } else {
                0.0
            },
        )?;
        d.set_item(
            "water_9dof_t_k",
            if n_water > 0 {
                2.0 * total_ke / ((9 * n_water) as f64 * R_KCAL)
            } else {
                0.0
            },
        )?;
        d.set_item(
            "water_internal_t_k",
            if n_water > 0 {
                2.0 * internal_ke / ((3 * n_water) as f64 * R_KCAL)
            } else {
                0.0
            },
        )?;
        Ok(d)
    }

    /// Instantaneous kinetic energy in kcal/mol (matches `state.kinetic_energy`,
    /// the same quantity `t_kin` is derived from). Lets callers compute the
    /// total energy E = U + KE and check conservation without needing DOF.
    fn kinetic_energy_kcal(&self) -> f64 {
        self.engine.state.kinetic_energy
    }

    fn step_count(&self) -> usize {
        self.engine.state.step_count
    }

    fn time_ps(&self) -> f64 {
        self.engine.state.time
    }

    fn set_temperature(&mut self, k: f32) {
        self.engine.set_temperature(k);
    }

    fn reset_velocities(&mut self) {
        self.engine.reset_velocities();
    }

    /// Switch the production integrator (diagnostics / thermostat tuning):
    ///   "langevin_middle" -> default LangevinMiddle gamma=0.5
    ///   "langevin_strong" -> LangevinMiddle gamma=10 (well-damped, settle-like)
    ///   "nve"             -> VerletVelocity with NO thermostat (energy-conservation probe)
    fn set_integrator(&mut self, mode: &str) -> PyResult<()> {
        let integrator = match mode {
            "langevin_middle" => dynamics::Integrator::LangevinMiddle { gamma: 0.5 },
            "langevin_strong" => dynamics::Integrator::LangevinMiddle { gamma: 10.0 },
            "nve" => dynamics::Integrator::VerletVelocity { thermostat: None },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown integrator mode '{other}' (expected langevin_middle | langevin_strong | nve)"
                )))
            }
        };
        self.engine.state.cfg.integrator = integrator;
        Ok(())
    }

    /// Toggle individual force classes (diagnostics). Each arg is a boolean:
    /// `true` DISABLES that class. Used to bisect the NVE energy leak — we
    /// disable one class at a time and see which one stops the ~7 kcal/mol/step
    /// spurious heating.
    fn set_force_overrides(&mut self, bonded: bool, coulomb: bool, lj: bool, long_range: bool) {
        let o = &mut self.engine.state.cfg.overrides;
        o.bonded_disabled = bonded;
        o.coulomb_disabled = coulomb;
        o.lj_disabled = lj;
        o.long_range_recip_disabled = long_range;
    }

    /// Diagnostics: skip the Langevin thermostat on rigid WATER (to bisect the
    /// ~+70 K thermostat-equilibrium offset: per-atom 9-component water noise +
    /// SETTLE projection vs the rest).
    fn set_skip_water_thermostat(&mut self, skip: bool) {
        self.engine.state.cfg.overrides.skip_water_thermostat = skip;
    }

    fn reset_pseudo_labels(&mut self) {
        self.engine.reset_pseudo_labels();
    }

    /// Per-protein-atom diagnostic labels: (element, one-letter residue,
    /// residue seq_id, mmCIF serial number) for every atom of the built system,
    /// in `MdState.atoms` index order — lets you identify the atoms that hit
    /// the accel clamp (the index in the `Warn: N atom(s) hit accel clamp ...`
    /// line is this same index).
    fn atom_labels<'py>(&self, _py: Python<'py>) -> PyResult<Vec<(String, char, i32, u32)>> {
        let n = self.engine.state.atoms.len();
        let mut res_of: Vec<(char, i32)> = vec![('?', 0); n];
        for r in &self.engine.topology.residues {
            for &i in &r.atom_indices {
                if i < n {
                    res_of[i] = (r.one_letter, r.seq_id);
                }
            }
        }
        Ok(self
            .engine
            .state
            .atoms
            .iter()
            .enumerate()
            .map(|(i, a)| {
                (
                    format!("{:?}", a.element),
                    res_of[i].0,
                    res_of[i].1,
                    a.serial_number,
                )
            })
            .collect())
    }

    /// Per-residue max |force| (kcal/mol/Å) over the residue's atoms — the
    /// per-residue strain map for targeted mutation (path B: which residue to
    /// mutate). [L] aligned with `topology.residues`. O(N), cheap to call each
    /// step. Complemented by `clash_report` / `atom_labels` for atom-level
    /// detail.
    fn per_residue_max_force(&self) -> Vec<f32> {
        let n = self.engine.state.atoms.len();
        let mut max_f = vec![0.0f32; self.engine.topology.residues.len()];
        for (ri, r) in self.engine.topology.residues.iter().enumerate() {
            let mut m = 0.0f32;
            for &i in &r.atom_indices {
                if i < n {
                    let fmag = self.engine.state.atoms[i].force.magnitude();
                    if fmag > m {
                        m = fmag;
                    }
                }
            }
            max_f[ri] = m;
        }
        max_f
    }

    /// Diagnostic: report every hydrogen whose current force magnitude exceeds
    /// `min_force`, along with its minimum distance to a NON-bonded atom (any
    /// atom in a different residue). The min distance is the quantity that
    /// `add_hydrogens::resolve_h_clashes` uses to decide whether to remove a
    /// clashing H (current threshold 1.2 Å), so this shows exactly which H's
    /// are hard clashes that slipped through. Returns
    /// (element, residue, seq_id, serial, |force| kcal/mol/Å, min_d Å).
    fn clash_report<'py>(
        &self,
        _py: Python<'py>,
        min_force: f32,
    ) -> PyResult<Vec<(String, char, i32, u32, f32, f32)>> {
        let n = self.engine.state.atoms.len();
        let mut res_of: Vec<(char, i32)> = vec![('?', 0); n];
        let mut atom_res: Vec<usize> = vec![0; n];
        for (ri, r) in self.engine.topology.residues.iter().enumerate() {
            for &i in &r.atom_indices {
                if i < n {
                    res_of[i] = (r.one_letter, r.seq_id);
                    atom_res[i] = ri;
                }
            }
        }
        let atoms = &self.engine.state.atoms;
        let mut out = Vec::new();
        for i in 0..n {
            let a = &atoms[i];
            if a.element != Element::Hydrogen {
                continue;
            }
            let fmag = a.force.magnitude();
            if fmag < min_force {
                continue;
            }
            let mut min_d = f32::INFINITY;
            for j in 0..n {
                if j == i || atom_res[j] == atom_res[i] {
                    continue;
                }
                let d = (atoms[j].posit - a.posit).magnitude();
                if d < min_d {
                    min_d = d;
                }
            }
            out.push((
                format!("{:?}", a.element),
                res_of[i].0,
                res_of[i].1,
                a.serial_number,
                fmag,
                min_d,
            ));
        }
        out.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        Ok(out)
    }

    /// Diagnostic: report EVERY atom (any element) whose current force
    /// magnitude exceeds `min_force`, with its minimum distance to a
    /// NON-bonded atom (different residue). Returns
    /// (element, residue, seq_id, serial, |force| kcal/mol/Å, min_d Å).
    fn force_report<'py>(
        &self,
        _py: Python<'py>,
        min_force: f32,
    ) -> PyResult<Vec<(String, char, i32, u32, f32, f32)>> {
        let n = self.engine.state.atoms.len();
        let mut res_of: Vec<(char, i32)> = vec![('?', 0); n];
        let mut atom_res: Vec<usize> = vec![0; n];
        for (ri, r) in self.engine.topology.residues.iter().enumerate() {
            for &i in &r.atom_indices {
                if i < n {
                    res_of[i] = (r.one_letter, r.seq_id);
                    atom_res[i] = ri;
                }
            }
        }
        let atoms = &self.engine.state.atoms;
        let mut out = Vec::new();
        for i in 0..n {
            let a = &atoms[i];
            let fmag = a.force.magnitude();
            if fmag < min_force {
                continue;
            }
            let mut min_d = f32::INFINITY;
            for j in 0..n {
                if j == i || atom_res[j] == atom_res[i] {
                    continue;
                }
                let d = (atoms[j].posit - a.posit).magnitude();
                if d < min_d {
                    min_d = d;
                }
            }
            out.push((
                format!("{:?}", a.element),
                res_of[i].0,
                res_of[i].1,
                a.serial_number,
                fmag,
                min_d,
            ));
        }
        out.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        Ok(out)
    }

    /// Post-build equilibration: NVT settle with strong friction + gentle
    /// temperature ramp.
    ///
    /// Releases the residual step-0 strain that minimization cannot remove —
    /// mostly added-HYDROGEN clashes sitting 1.5-2.0 Å from a non-bonded atom
    /// (a static minimizer can't fix them; the H's are pinned by their bonds).
    ///
    /// Validated: on 2LYZ, max |force| drops from ~10⁴ (step 0) to ~107 by
    /// step 100, and post-equilibration production has ZERO accel clamps (vs
    /// 13-19/step before). Positional restraints (`k_restraint>0`) are an
    /// opt-in and are OFF by default — they freeze the heavy skeleton and stop
    /// the H's from relaxing, which stores strain and causes a mid-ramp crash.
    ///
    /// Optional — `Engine.build` stays fast without it. Typical: build, call
    /// `equilibrate()`, then run production MD. Returns an error if the system
    /// blows up mid-ramp (caller can treat as build_failed).
    #[pyo3(signature = (ramp_steps=300, t_start_k=100.0, k_restraint=0.0, hold_steps=100, restrain_hydrogens=false, friction_gamma=10.0))]
    fn equilibrate(
        &mut self,
        ramp_steps: usize,
        t_start_k: f32,
        k_restraint: f32,
        hold_steps: usize,
        restrain_hydrogens: bool,
        friction_gamma: f32,
    ) -> PyResult<()> {
        let cfg = EquilConfig {
            ramp_steps,
            t_start_k,
            k_restraint,
            hold_steps,
            restrain_hydrogens,
            friction_gamma,
        };
        crate::equilibrate::equilibrate(&mut self.engine, &cfg).map_err(PyValueError::new_err)
    }

    /// Engine-internal per-category timing sums (µs), sampled every 20 steps.
    /// Returns a dict of the accumulated cost per MD phase — the profile that
    /// tells us where a step's time actually goes (bonded / nonbonded / ewald /
    /// neighbors / integrate / ambient / snapshots / total).
    fn computation_time<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let ct = &self.engine.state.computation_time;
        let d = PyDict::new(py);
        d.set_item("bonded_us", ct.bonded_sum)?;
        d.set_item("nonbonded_short_us", ct.non_bonded_short_range_sum)?;
        d.set_item("ewald_long_us", ct.ewald_long_range_sum)?;
        d.set_item("neighbor_all_us", ct.neighbor_all_sum)?;
        d.set_item("neighbor_rebuild_us", ct.neighbor_rebuild_sum)?;
        d.set_item("neighbor_rebuild_count", ct.neighbor_rebuild_count)?;
        d.set_item("integration_us", ct.integration_sum)?;
        d.set_item("ambient_us", ct.ambient_sum)?;
        d.set_item("snapshot_us", ct.snapshot_sum)?;
        d.set_item("total_us", ct.total)?;
        d.set_item("steps", self.engine.state.step_count)?;
        Ok(d)
    }

    /// Fraction of residues currently receiving bias force.
    fn mask_fraction(&self) -> f32 {
        self.force.mask.fraction()
    }
}

impl PyEngine {
    /// Shared step driver. `want_metrics=false` skips the expensive
    /// `Metrics::compute` (O(N²) clash/surface + DSSP-lite) — the hot path for
    /// tight MD loops; metrics are then available via the `metrics()` method.
    fn step_impl<'py>(
        &mut self,
        py: Python<'py>,
        action: Option<PyReadonlyArray1<'_, f32>>,
        want_metrics: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = match action {
            Some(a) => {
                let v = a.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
                self.force.step(&mut self.engine, v)
            }
            None => self.engine.step(None),
        };
        let m = want_metrics.then(|| self.metrics.compute(&self.engine));

        let d = PyDict::new(py);
        d.set_item("u_t_kcal", result.u_t_kcal)?;
        d.set_item("u_t_kj", result.u_t_kj)?;
        let coords_vec: Vec<Vec<f32>> = result
            .coords_ca
            .iter()
            .map(|c| vec![c[0], c[1], c[2]])
            .collect();
        d.set_item("coords_ca", PyArray2::from_vec2(py, &coords_vec)?)?;
        d.set_item("step_count", result.step_count)?;
        d.set_item("time_ps", result.time_ps)?;
        d.set_item("crashed", result.crashed)?;
        d.set_item("crash_reason", result.crash_reason.clone())?;
        // Clamp / thermostat observables: n_clamped + max_accel_clamped are the
        // "temperature instability" signal (sustained clamps = force spikes being
        // swallowed by MAX_ACCEL), t_kin verifies the thermostat reached target T.
        d.set_item("n_clamped", self.engine.state.last_clamped_count)?;
        d.set_item("max_accel_clamped", self.engine.state.last_clamped_mag)?;
        d.set_item("t_kin", self.engine.state.last_temperature_k)?;
        if let Some(m) = m {
            d.set_item("m1", m.m1)?;
            d.set_item("m2", m.m2)?;
            d.set_item("m3", m.m3)?;
            d.set_item("m4", m.m4)?;
            d.set_item("m5", m.m5)?;
            d.set_item("rg", m.rg)?;
            d.set_item("stability_margin", m.stability_margin)?;
            d.set_item("rmsf", m.rmsf)?;
        }
        Ok(d)
    }

    fn topology(&self) -> &crate::topology::ProteinTopology {
        &self.engine.topology
    }
}

/// Apply a point mutation to a sequence and return the new sequence.
#[pyfunction]
fn mutate_sequence(seq: &str, position: usize, to: char) -> PyResult<String> {
    crate::mutate::apply_mutations(seq, &[crate::mutate::Mutation::new(position, to)])
        .map_err(|e| PyValueError::new_err(e))
}

/// Validate that a sequence contains only standard amino acids.
#[pyfunction]
fn validate_sequence(seq: &str) -> PyResult<()> {
    crate::mutate::validate_sequence(seq).map_err(|e| PyValueError::new_err(e))
}

/// Shared scan pipeline: build each point's system under `grid`, run `n_steps`
/// MD steps, report stability. Runs in parallel (one build per worker).
fn scan_impl<'py>(
    py: Python<'py>,
    structure: &Bound<'py, PyStructure>,
    grid: crate::domain::EnvGrid,
    n_steps: usize,
    equil_steps: usize,
    repeats: usize,
    relax_iters: Option<usize>,
    tolerance: f32,
    prune_crashed: bool,
    anchor_temp: f32,
    adaptive_repeats: bool,
    trend_detector: bool,
    trend_window: usize,
    trend_z_threshold: f64,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let cfg = crate::domain::StabilityConfig {
        n_steps,
        equil_steps,
        repeats,
        relax_iters,
        tolerance,
        prune_crashed,
        anchor_temp,
        adaptive_repeats,
        trend: crate::domain::TrendConfig {
            enabled: trend_detector,
            window: trend_window,
            z_threshold: trend_z_threshold,
            ..Default::default()
        },
        ..Default::default()
    };
    let opts = BuildOptions::default();
    let structure = structure.borrow();
    let pts = crate::domain::scan_stability(
        &dynamics::ComputationDevice::Cpu,
        param_set(),
        &structure.inner,
        &grid,
        &opts,
        &cfg,
    );
    let mut out = Vec::with_capacity(pts.len());
    for p in pts {
        let d = PyDict::new(py);
        d.set_item("temp", p.env.temp_k)?;
        d.set_item("ph", p.env.ph)?;
        d.set_item("pressure", p.env.pressure_bar)?;
        d.set_item("ionic", p.env.ionic_strength_m)?;
        d.set_item("stable", p.stable)?;
        d.set_item("crashed", p.crashed)?;
        d.set_item("build_failed", p.build_failed)?;
        d.set_item("terminated_reason", p.terminated_reason.clone())?;
        match &p.metrics {
            Some(m) => {
                d.set_item("m1", m.m1)?;
                d.set_item("m2", m.m2)?;
                d.set_item("m3", m.m3)?;
                d.set_item("m4", m.m4)?;
                d.set_item("m5", m.m5)?;
                d.set_item("rg", m.rg)?;
            }
            None => {
                for k in ["m1", "m2", "m3", "m4", "m5", "rg"] {
                    d.set_item(k, f64::NAN)?;
                }
            }
        }
        out.push(d);
    }
    Ok(out)
}

/// Batch-scan the stability domain over an explicit (temp × ph) point grid.
/// Each point builds the system, runs `n_steps` MD steps and reports whether the
/// protein stayed folded. Runs in parallel (one build per worker).
#[pyfunction]
#[pyo3(signature = (
    structure,
    temps,
    phs,
    pressures = None,
    ionics = None,
    n_steps = 20,
    equil_steps = 10,
    repeats = 3,
    relax_iters = None,
    tolerance = 2.0,
    prune_crashed = true,
    anchor_temp = 310.0,
    adaptive_repeats = true,
    trend_detector = true,
    trend_window = 100,
    trend_z_threshold = 3.0
))]
fn scan_stability<'py>(
    py: Python<'py>,
    structure: &Bound<'py, PyStructure>,
    temps: Vec<f32>,
    phs: Vec<f32>,
    pressures: Option<Vec<f32>>,
    ionics: Option<Vec<f32>>,
    n_steps: usize,
    equil_steps: usize,
    repeats: usize,
    relax_iters: Option<usize>,
    tolerance: f32,
    prune_crashed: bool,
    anchor_temp: f32,
    adaptive_repeats: bool,
    trend_detector: bool,
    trend_window: usize,
    trend_z_threshold: f64,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let grid = crate::domain::EnvGrid {
        temps,
        phs,
        pressures: pressures.unwrap_or_else(|| vec![1.0]),
        ionics: ionics.unwrap_or_else(|| vec![0.0]),
    };
    scan_impl(
        py,
        structure,
        grid,
        n_steps,
        equil_steps,
        repeats,
        relax_iters,
        tolerance,
        prune_crashed,
        anchor_temp,
        adaptive_repeats,
        trend_detector,
        trend_window,
        trend_z_threshold,
    )
}

/// Batch-scan the stability domain over per-axis `(start, end, step)` ranges,
/// so each SPICE dimension uses its own resolution — e.g. fine temperature
/// steps (5 K) but coarse pH steps (1.0, since protonation is discrete).
#[pyfunction]
#[pyo3(signature = (
    structure,
    temp_range,
    ph_range,
    pressure_range = None,
    ionic_range = None,
    n_steps = 20,
    equil_steps = 10,
    repeats = 3,
    relax_iters = None,
    tolerance = 2.0,
    prune_crashed = true,
    anchor_temp = 310.0,
    adaptive_repeats = true,
    trend_detector = true,
    trend_window = 100,
    trend_z_threshold = 3.0
))]
fn scan_stability_ranges<'py>(
    py: Python<'py>,
    structure: &Bound<'py, PyStructure>,
    temp_range: (f32, f32, f32),
    ph_range: (f32, f32, f32),
    pressure_range: Option<(f32, f32, f32)>,
    ionic_range: Option<(f32, f32, f32)>,
    n_steps: usize,
    equil_steps: usize,
    repeats: usize,
    relax_iters: Option<usize>,
    tolerance: f32,
    prune_crashed: bool,
    anchor_temp: f32,
    adaptive_repeats: bool,
    trend_detector: bool,
    trend_window: usize,
    trend_z_threshold: f64,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let grid =
        crate::domain::EnvGrid::from_ranges(temp_range, ph_range, pressure_range, ionic_range);
    if grid.is_empty() {
        return Ok(Vec::new());
    }
    scan_impl(
        py,
        structure,
        grid,
        n_steps,
        equil_steps,
        repeats,
        relax_iters,
        tolerance,
        prune_crashed,
        anchor_temp,
        adaptive_repeats,
        trend_detector,
        trend_window,
        trend_z_threshold,
    )
}

/// Dict of an environment point.
fn env_dict<'py>(py: Python<'py>, e: &EnvParams) -> Bound<'py, PyDict> {
    let d = PyDict::new(py);
    d.set_item("temp", e.temp_k).unwrap();
    d.set_item("ph", e.ph).unwrap();
    d.set_item("pressure", e.pressure_bar).unwrap();
    d.set_item("ionic", e.ionic_strength_m).unwrap();
    d
}

/// Bidirectional stability-domain probe: from an (assumed stable) anchor
/// environment, walk each axis outward in both + and − directions until the
/// point is judged unstable, a build fails, or `max_steps` is reached. Each ray
/// runs in parallel. Pass `step=None` to skip an axis.
#[pyfunction]
#[pyo3(signature = (
    structure,
    anchor_ph,
    anchor_temp,
    anchor_pressure = 1.0,
    anchor_ionic = 0.0,
    temp_step = Some(10.0),
    temp_max = 10,
    temp_precision = None,
    ph_step = Some(0.5),
    ph_max = 10,
    ph_precision = None,
    pressure_step = None,
    pressure_max = 10,
    pressure_precision = None,
    ionic_step = None,
    ionic_max = 10,
    ionic_precision = None,
    n_steps = 20,
    equil_steps = 10,
    repeats = 3,
    relax_iters = None,
    tolerance = 2.0
))]
fn scan_radial<'py>(
    py: Python<'py>,
    structure: &Bound<'py, PyStructure>,
    anchor_ph: f32,
    anchor_temp: f32,
    anchor_pressure: f32,
    anchor_ionic: f32,
    temp_step: Option<f32>,
    temp_max: usize,
    temp_precision: Option<f32>,
    ph_step: Option<f32>,
    ph_max: usize,
    ph_precision: Option<f32>,
    pressure_step: Option<f32>,
    pressure_max: usize,
    pressure_precision: Option<f32>,
    ionic_step: Option<f32>,
    ionic_max: usize,
    ionic_precision: Option<f32>,
    n_steps: usize,
    equil_steps: usize,
    repeats: usize,
    relax_iters: Option<usize>,
    tolerance: f32,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    use crate::domain::{Axis, AxisProbe, Direction};

    let anchor = EnvParams::new(anchor_ph, anchor_temp, anchor_pressure, anchor_ionic);
    let mut probes = Vec::new();
    if let Some(ts) = temp_step {
        probes.push(AxisProbe {
            axis: Axis::Temp,
            step: ts,
            max_steps: temp_max,
            precision: Some(temp_precision.unwrap_or(ts)),
        });
    }
    if let Some(ps) = ph_step {
        probes.push(AxisProbe {
            axis: Axis::Ph,
            step: ps,
            max_steps: ph_max,
            precision: Some(ph_precision.unwrap_or(ps)),
        });
    }
    if let Some(s) = pressure_step {
        probes.push(AxisProbe {
            axis: Axis::Pressure,
            step: s,
            max_steps: pressure_max,
            precision: Some(pressure_precision.unwrap_or(s)),
        });
    }
    if let Some(s) = ionic_step {
        probes.push(AxisProbe {
            axis: Axis::Ionic,
            step: s,
            max_steps: ionic_max,
            precision: Some(ionic_precision.unwrap_or(s)),
        });
    }
    let cfg = crate::domain::StabilityConfig {
        n_steps,
        equil_steps,
        repeats,
        relax_iters,
        tolerance,
        ..Default::default()
    };
    let opts = BuildOptions::default();
    let structure = structure.borrow();
    let rays = crate::domain::scan_radial(
        &dynamics::ComputationDevice::Cpu,
        param_set(),
        &structure.inner,
        anchor,
        &probes,
        &opts,
        &cfg,
    );

    let mut out = Vec::with_capacity(rays.len());
    for r in rays {
        let d = PyDict::new(py);
        d.set_item("axis", r.axis.name())?;
        d.set_item(
            "direction",
            if r.direction == Direction::Positive { "+" } else { "-" },
        )?;
        d.set_item("anchor", env_dict(py, &r.anchor))?;
        match r.boundary_stable() {
            Some(p) => d.set_item("boundary_stable", env_dict(py, &p.env))?,
            None => d.set_item("boundary_stable", py.None())?,
        }
        match r.first_unstable() {
            Some(p) => d.set_item("first_unstable", env_dict(py, &p.env))?,
            None => d.set_item("first_unstable", py.None())?,
        }
        d.set_item("n_stable", r.points.iter().filter(|p| p.stable).count())?;
        d.set_item("n_probed", r.points.len())?;

        let pts: Vec<Bound<'py, PyDict>> = r
            .points
            .iter()
            .map(|p| {
                let pd = PyDict::new(py);
                pd.set_item("temp", p.env.temp_k).unwrap();
                pd.set_item("ph", p.env.ph).unwrap();
                pd.set_item("stable", p.stable).unwrap();
                pd.set_item("crashed", p.crashed).unwrap();
                pd.set_item("build_failed", p.build_failed).unwrap();
                match &p.metrics {
                    Some(m) => {
                        pd.set_item("m1", m.m1).unwrap();
                        pd.set_item("m2", m.m2).unwrap();
                        pd.set_item("m3", m.m3).unwrap();
                        pd.set_item("m4", m.m4).unwrap();
                        pd.set_item("m5", m.m5).unwrap();
                    }
                    None => {}
                }
                pd
            })
            .collect();
        d.set_item("points", pts)?;
        out.push(d);
    }
    Ok(out)
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// `spice_engine` Python module.
#[pymodule]
fn spice_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStructure>()?;
    m.add_class::<PyEngine>()?;
    m.add_function(wrap_pyfunction!(mutate_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(validate_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(scan_stability, m)?)?;
    m.add_function(wrap_pyfunction!(scan_stability_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(scan_radial, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
