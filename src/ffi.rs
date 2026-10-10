//! Python bindings (PyO3 + rust-numpy) — the P4 FFI surface.
//!
//! Python side feeds in-memory atom arrays (`PyStructure`) built from its data
//! pipeline (Parquet → numpy), then drives `PyEngine`:
//! step (optional bias-force action), metrics (five-dimensional `M`),
//! pseudo-labels (time-averaged Cα), reset / set_temperature.
//!
//! v1.3.7 ergonomics: `se.Env` is a reusable build-recipe object — pass it as
//! `env=` to `Engine.build` / `Engine.mutate_with_solvent_reuse` instead of
//! re-typing the environment scalars on every call (the legacy positional
//! style remains fully supported). Observability getters:
//! `effective_ionic_strength_m`, `pressure_bar`, `env_info`,
//! `exclusion_diagnostics`; `set_trend_monitor(preset="rl_fail_fast")`.
//! v1.3.8 observability probes: `electrostatic_potential`, `atom_sasa`,
//! `atom_names` / `select_atoms` / `contact_count` / `bottleneck_radius`,
//! and mutate's optional `repack_ions=True` salt mode.
//! See `docs/capabilities.md` for the full surface.
//!
//! Build with maturin: `maturin develop --features python` (or `--release`).

use std::sync::OnceLock;
use std::time::Instant;

use na_seq::Element;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyDeprecationWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::actions::ForceAction;
use crate::builder::BuildOptions;
use crate::engine::SpiceEngine;
use crate::env::{BuildScalars, EnvParams, EnvSpec, ResolvedBuild, resolve_build_args};
use crate::equilibrate::EquilConfig;
use crate::metrics::{Metrics, MetricsConfig};
use crate::structure::{AtomInput, StructureInput, build_from_input};

/// Lazily loaded Amber parameter set (load once per process).
fn param_set() -> &'static crate::engine::md_core::params::FfParamSet {
    static PS: OnceLock<crate::engine::md_core::params::FfParamSet> = OnceLock::new();
    PS.get_or_init(|| {
        crate::engine::md_core::params::FfParamSet::new_amber().expect("load amber params")
    })
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
            return err(format!(
                "array length mismatch: atoms={} elms={} res_seq={} res_names={} coords={n}",
                atom_names.len(),
                elements.len(),
                res_seq.len(),
                res_names.len()
            ));
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
        self.inner.sequence().map_err(PyValueError::new_err)
    }

    fn residue_count(&self) -> usize {
        self.inner.residue_count()
    }
}

/// Build-environment recipe (v1.3.7). ONE `Env` drives both
/// `Engine.build(structure, env=e)` and
/// `Engine.mutate_with_solvent_reuse(structure, env=e)`: the long scalar
/// chain is defined once and can never drift between the two calls. All
/// fields are plain attributes — build once, tweak one knob, rebuild:
///
/// ```python
/// e = se.Env(ph=7.0, temp_k=310.0, pressure=0.0, ionic_strength_m=0.15)
/// eng = se.Engine.build(structure, env=e)
/// e.temp_k = 350.0                     # attribute is `temp_k`, not `temp`
/// hot = eng.mutate_with_solvent_reuse(structure, env=e)
/// ```
///
/// Passing `env=` together with any individual scalar raises (no silent
/// override ambiguity). The legacy positional style
/// `(structure, ph, temp, pressure, ionic_strength_m, relax_iters, tolerance)`
/// stays fully supported and un-deprecated.
#[pyclass(name = "Env", module = "spice_engine", skip_from_py_object)]
#[derive(Clone)]
pub struct PyEnv {
    #[pyo3(get, set)]
    ph: f32,
    #[pyo3(get, set)]
    temp_k: f32,
    #[pyo3(get, set)]
    pressure_bar: f32,
    #[pyo3(get, set)]
    ionic_strength_m: f32,
    /// Iteration cap of the build-tail L-BFGS minimization.
    #[pyo3(get, set)]
    relax_iters: usize,
    /// Minimization convergence tolerance.
    #[pyo3(get, set)]
    tolerance: f32,
    #[pyo3(get, set)]
    strict_incomplete: bool,
    #[pyo3(get, set)]
    mg_cl2_m: f32,
    #[pyo3(get, set)]
    ca_cl2_m: f32,
    #[pyo3(get, set)]
    sr_cl2_m: f32,
    #[pyo3(get, set)]
    ba_cl2_m: f32,
    #[pyo3(get, set)]
    redox_reducing: f32,
    /// JSON channel to the cosolvent presets/custom-spec parser.
    #[pyo3(get, set)]
    cosolvents_json: String,
    /// JSON channel to the general-salt registry parser.
    #[pyo3(get, set)]
    salts_json: String,
    /// Solvent box padding (Å) around the solute; 0 = engine default (10 Å,
    /// folded-state-sized). Denature/refold (PFDE) runs must raise this so
    /// the unfolded contour fits inside the cell (see env::EnvSpec::box_pad_a).
    #[pyo3(get, set)]
    box_pad_a: f32,
}

#[pymethods]
impl PyEnv {
    #[new]
    #[pyo3(signature = (ph = 7.0, temp_k = 310.0, pressure_bar = 1.0, ionic_strength_m = 0.0,
                        relax_iters = 2000, tolerance = 2.0, strict_incomplete = true,
                        mg_cl2_m = 0.0, ca_cl2_m = 0.0, sr_cl2_m = 0.0, ba_cl2_m = 0.0,
                        redox_reducing = 0.0, cosolvents_json = "", salts_json = "",
                        box_pad_a = 0.0))]
    // Python constructor mirrors every EnvSpec field; clippy's Rust-idiom
    // arg-count advice does not apply to a kwargs surface.
    #[allow(clippy::too_many_arguments)]
    fn new(
        ph: f32,
        temp_k: f32,
        pressure_bar: f32,
        ionic_strength_m: f32,
        relax_iters: usize,
        tolerance: f32,
        strict_incomplete: bool,
        mg_cl2_m: f32,
        ca_cl2_m: f32,
        sr_cl2_m: f32,
        ba_cl2_m: f32,
        redox_reducing: f32,
        cosolvents_json: &str,
        salts_json: &str,
        box_pad_a: f32,
    ) -> Self {
        Self {
            ph,
            temp_k,
            pressure_bar,
            ionic_strength_m,
            relax_iters,
            tolerance,
            strict_incomplete,
            mg_cl2_m,
            ca_cl2_m,
            sr_cl2_m,
            ba_cl2_m,
            redox_reducing,
            cosolvents_json: cosolvents_json.to_string(),
            salts_json: salts_json.to_string(),
            box_pad_a,
        }
    }

    /// Convenience alias: `Env(pressure=...)` reads like every other call
    /// site in the project; the canonical attribute stays `pressure_bar`.
    #[getter]
    fn pressure(&self) -> f32 {
        self.pressure_bar
    }

    #[setter]
    fn set_pressure(&mut self, v: f32) {
        self.pressure_bar = v;
    }

    /// The recipe as a plain dict (keys = attribute names).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("ph", self.ph)?;
        d.set_item("temp_k", self.temp_k)?;
        d.set_item("pressure_bar", self.pressure_bar)?;
        d.set_item("ionic_strength_m", self.ionic_strength_m)?;
        d.set_item("relax_iters", self.relax_iters)?;
        d.set_item("tolerance", self.tolerance)?;
        d.set_item("strict_incomplete", self.strict_incomplete)?;
        d.set_item("mg_cl2_m", self.mg_cl2_m)?;
        d.set_item("ca_cl2_m", self.ca_cl2_m)?;
        d.set_item("sr_cl2_m", self.sr_cl2_m)?;
        d.set_item("ba_cl2_m", self.ba_cl2_m)?;
        d.set_item("redox_reducing", self.redox_reducing)?;
        d.set_item("cosolvents_json", &self.cosolvents_json)?;
        d.set_item("salts_json", &self.salts_json)?;
        d.set_item("box_pad_a", self.box_pad_a)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        let salt = match (self.mg_cl2_m, self.ca_cl2_m, self.sr_cl2_m, self.ba_cl2_m) {
            (0.0, 0.0, 0.0, 0.0) => String::new(),
            (mg, ca, sr, ba) => format!(" mg={mg} ca={ca} sr={sr} ba={ba}"),
        };
        let cosolv = if self.cosolvents_json.is_empty() {
            String::new()
        } else {
            " +cosolvents".to_string()
        };
        let salts = if self.salts_json.is_empty() {
            String::new()
        } else {
            " +salts_json".to_string()
        };
        let boxp = if self.box_pad_a > 0.0 {
            format!(" box_pad_a={}", self.box_pad_a)
        } else {
            String::new()
        };
        format!(
            "<Env ph={} temp_k={} pressure_bar={} ionic_strength_m={} relax_iters={} tolerance={} strict={}{}{}{}{}>",
            self.ph,
            self.temp_k,
            self.pressure_bar,
            self.ionic_strength_m,
            self.relax_iters,
            self.tolerance,
            self.strict_incomplete,
            salt,
            cosolv,
            salts,
            boxp,
        )
    }
}

impl PyEnv {
    fn to_spec(&self) -> EnvSpec {
        EnvSpec {
            ph: self.ph,
            temp_k: self.temp_k,
            pressure_bar: self.pressure_bar,
            ionic_strength_m: self.ionic_strength_m,
            relax_iters: self.relax_iters,
            tolerance: self.tolerance,
            strict_incomplete: self.strict_incomplete,
            mg_cl2_m: self.mg_cl2_m,
            ca_cl2_m: self.ca_cl2_m,
            sr_cl2_m: self.sr_cl2_m,
            ba_cl2_m: self.ba_cl2_m,
            redox_reducing: self.redox_reducing,
            cosolvents_json: self.cosolvents_json.clone(),
            salts_json: self.salts_json.clone(),
            box_pad_a: self.box_pad_a,
        }
    }
}

/// Shared "scalars OR Env" front-door for build & mutate_with_solvent_reuse
/// (v1.3.7). Resolution rules live in the pure kernel
/// (`env::resolve_build_args`); this only marshals Python types and raises
/// the DeprecationWarning for explicitly-passed trailing scalars.
#[allow(clippy::too_many_arguments)]
fn resolve_env(
    py: Python<'_>,
    env: Option<PyRef<'_, PyEnv>>,
    ph: Option<f32>,
    temp: Option<f32>,
    pressure: Option<f32>,
    ionic_strength_m: Option<f32>,
    relax_iters: Option<usize>,
    tolerance: Option<f32>,
    strict_incomplete: Option<bool>,
    mg_molar: Option<f32>,
    ca_molar: Option<f32>,
    sr_molar: Option<f32>,
    ba_molar: Option<f32>,
    redox_reducing: Option<f32>,
    cosolvents_json: Option<&str>,
    salts_json: Option<&str>,
) -> PyResult<ResolvedBuild> {
    let scalars = BuildScalars {
        ph,
        temp,
        pressure,
        ionic_strength_m,
        relax_iters,
        tolerance,
        strict_incomplete,
        mg_molar,
        ca_molar,
        sr_molar,
        ba_molar,
        redox_reducing,
        cosolvents_json: cosolvents_json.map(str::to_owned),
        salts_json: salts_json.map(str::to_owned),
    };
    let spec = env.map(|e| e.to_spec());
    let resolved = resolve_build_args(&scalars, spec.as_ref()).map_err(PyValueError::new_err)?;
    warn_deprecated_scalars(py, &resolved)?;
    Ok(resolved)
}

/// Turn resolved scalars into `BuildOptions` (shared by build & mutate).
fn build_options(r: &ResolvedBuild) -> PyResult<BuildOptions> {
    let d = BuildOptions::default();
    Ok(BuildOptions {
        env: r.env,
        cosolvents: crate::engine::md_core::cosolvent_presets::parse_cosolvents_json(
            &r.cosolvents_json,
        )
        .map_err(PyValueError::new_err)?,
        // v1.3.2 general electrolyte channel, e.g.
        // `[{"cation":"K+","anion":"Cl-","molarity":0.15}]` — names
        // resolve through the provenance-locked ion registry and the
        // stoichiometry is derived from their charges.
        salts: crate::engine::md_core::species::parse_salts_json(&r.salts_json)
            .map_err(PyValueError::new_err)?,
        relax_iters: Some(r.relax_iters),
        energy_minimization_tolerance: r.tolerance,
        strict_incomplete_residues: r.strict_incomplete,
        // 0.0 sentinel = engine default (10 A). Env-only knob; see
        // env::EnvSpec::box_pad_a for why denature/refold needs a big box.
        box_padding_angstrom: if r.box_pad_a > 0.0 {
            r.box_pad_a
        } else {
            d.box_padding_angstrom
        },
        ..d
    })
}

/// v1.3.7: trailing env scalars passed straight to build/mutate still work
/// but earn a DeprecationWarning pointing at `se.Env(...)`.
fn warn_deprecated_scalars(py: Python<'_>, r: &ResolvedBuild) -> PyResult<()> {
    if r.deprecated_used.is_empty() {
        return Ok(());
    }
    let category = py.get_type::<PyDeprecationWarning>();
    let msg = std::ffi::CString::new(format!(
        "spice_engine: {} passed directly to build()/mutate_with_solvent_reuse() \
         is deprecated - fold it into a reusable se.Env(...) object and pass env=... \
         (see docs/capabilities.md)",
        r.deprecated_used.join(", ")
    ))
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    PyErr::warn(py, &category, msg.as_c_str(), 2)
}

/// The MD engine wrapper exposed to Python.
#[pyclass(name = "Engine", module = "spice_engine")]
pub struct PyEngine {
    pub engine: SpiceEngine,
    pub force: ForceAction,
    pub metrics: Metrics,
    /// Last-call timings in microseconds; MD excludes metrics computation.
    pub last_md_us: u64,
    pub last_metrics_us: u64,
    /// Reused row storage for NumPy coordinate conversion.
    pub coords_scratch: Vec<Vec<f32>>,
    pub flat_coords_scratch: Vec<f32>,
}

#[pymethods]
impl PyEngine {
    /// Build a system from an in-memory `Structure` plus environment.
    ///
    /// Two styles, never mixed (mixing raises):
    /// - v1.3.7 preferred: `Engine.build(structure, env=se.Env(...))` — one
    ///   reusable recipe object; tweak attributes (`e.temp_k = 350`) and
    ///   rebuild instead of re-typing scalars.
    /// - Legacy: `Engine.build(structure, ph, temp, pressure,
    ///   ionic_strength_m, relax_iters, tolerance, ...)` — the 7-slot
    ///   positional prefix is fully supported (spice_rl style) and NOT
    ///   deprecated; the trailing divalent/redox/cosolvents/salts scalars
    ///   still work but earn a DeprecationWarning (fold them into `Env`).
    ///
    /// `strict_incomplete=True` (default) rejects structures with residues
    /// missing any charge-lib sidechain heavy atom (disordered/truncated
    /// crystal sidechains — per-residue check, also catches "only CB
    /// survived"), failing with a clear error listing each residue's missing
    /// atoms. Set `False` to build them with just the atoms present (warning)
    /// — physics is wrong for those residues, so only use it to explore.
    #[staticmethod]
    #[pyo3(signature = (structure, ph = None, temp = None, pressure = None, ionic_strength_m = None, relax_iters = None, tolerance = None, strict_incomplete = None, mg_molar = None, ca_molar = None, sr_molar = None, ba_molar = None, redox_reducing = None, cosolvents_json = None, salts_json = None, env = None))]
    // Wide by contract: the legacy positional/keyword surface must accept
    // every parameter individually (compat), so no Rust-idiom narrowing here.
    #[allow(clippy::too_many_arguments)]
    fn build(
        py: Python<'_>,
        structure: &Bound<'_, PyStructure>,
        ph: Option<f32>,
        temp: Option<f32>,
        pressure: Option<f32>,
        ionic_strength_m: Option<f32>,
        relax_iters: Option<usize>,
        tolerance: Option<f32>,
        strict_incomplete: Option<bool>,
        mg_molar: Option<f32>,
        ca_molar: Option<f32>,
        sr_molar: Option<f32>,
        ba_molar: Option<f32>,
        redox_reducing: Option<f32>,
        cosolvents_json: Option<&str>,
        salts_json: Option<&str>,
        env: Option<PyRef<'_, PyEnv>>,
    ) -> PyResult<Self> {
        let resolved = resolve_env(
            py,
            env,
            ph,
            temp,
            pressure,
            ionic_strength_m,
            relax_iters,
            tolerance,
            strict_incomplete,
            mg_molar,
            ca_molar,
            sr_molar,
            ba_molar,
            redox_reducing,
            cosolvents_json,
            salts_json,
        )?;
        let dev = crate::engine::md_core::ComputationDevice::Cpu;
        let opts = build_options(&resolved)?;
        let structure = structure.borrow();
        let engine = build_from_input(&dev, param_set(), &structure.inner, &opts)
            .map_err(PyValueError::new_err)?;
        Ok(Self::from_engine(engine))
    }

    /// Create a new engine with a mutated structure by reusing the solvent box and ions
    /// of this engine. Unchanged residues ride the PARENT's relaxed coordinates
    /// (index-aligned copy; v1.3.6 — the reused box is at the parent's state, so
    /// a raw-position solute start would detonate the equilibration ramp).
    /// Cost routing: a pure environment / temperature change on an unchanged
    /// structure SKIPS minimization (~0.2 s on 2LYZ, ~1 s on HSFA2); a real
    /// single-point mutation runs a 6 Å LOCAL minimization (~1 s); both are far
    /// below a ~30 s cold build. A pH change relaxes only the (de)protonated
    /// residues. By default this path reuses the parent's ion
    /// population and does NOT re-pack salt for a changed `ionic_strength_m`
    /// / divalent concentration (decided v1.3.5; guard warns if you try) — vary those through the
    /// rebuild-based stability scan (`domain::scan_stability`), which keys a
    /// fresh solvated build on ionic strength.
    /// v1.3.7: accepts the same two styles as `Engine.build` — prefer
    /// `mutate_with_solvent_reuse(structure, env=e)` reusing the very recipe
    /// the parent was built with (single source of truth; the 7-slot
    /// positional legacy prefix keeps working for existing callers).
    /// v1.3.8: `repack_ions=True` opts INTO in-place salt re-packing for a
    /// changed salt request: the solvent box stays, the parent's monatomic
    /// ions are stripped and the requested composition re-inserted
    /// (counterions + background + divalent/generic salts), then the new
    /// sites enter the 6 Å relaxation shell. Ion counts and effective I
    /// match a fresh build; positions carry an accepted density drift and
    /// are NOT bitwise-comparable to one (salt scans at reuse cost).
    #[pyo3(signature = (structure, ph = None, temp = None, pressure = None, ionic_strength_m = None, relax_iters = None, tolerance = None, strict_incomplete = None, mg_molar = None, ca_molar = None, sr_molar = None, ba_molar = None, redox_reducing = None, cosolvents_json = None, salts_json = None, env = None, repack_ions = false))]
    // Same wide-by-contract rationale as `build` (kept signature-identical).
    #[allow(clippy::too_many_arguments)]
    fn mutate_with_solvent_reuse(
        &self,
        py: Python<'_>,
        structure: &Bound<'_, PyStructure>,
        ph: Option<f32>,
        temp: Option<f32>,
        pressure: Option<f32>,
        ionic_strength_m: Option<f32>,
        relax_iters: Option<usize>,
        tolerance: Option<f32>,
        strict_incomplete: Option<bool>,
        mg_molar: Option<f32>,
        ca_molar: Option<f32>,
        sr_molar: Option<f32>,
        ba_molar: Option<f32>,
        redox_reducing: Option<f32>,
        cosolvents_json: Option<&str>,
        salts_json: Option<&str>,
        env: Option<PyRef<'_, PyEnv>>,
        repack_ions: bool,
    ) -> PyResult<Self> {
        let resolved = resolve_env(
            py,
            env,
            ph,
            temp,
            pressure,
            ionic_strength_m,
            relax_iters,
            tolerance,
            strict_incomplete,
            mg_molar,
            ca_molar,
            sr_molar,
            ba_molar,
            redox_reducing,
            cosolvents_json,
            salts_json,
        )?;
        // NOTE: by default the mutant path reuses the PARENT's solvent box and
        // ions, so cosolvent molecules come along from the parent; the
        // cosolvents/salts JSON here (on either style) only records intent for
        // diagnostics and does NOT pack a second population into the reused
        // box. Exception: repack_ions=True (v1.3.8) re-packs the MONATOMIC
        // ion population to the requested salt; cosolvents still ride along.
        let mut opts = build_options(&resolved)?;
        opts.repack_ions = repack_ions;
        let structure = structure.borrow();
        let engine = crate::builder::build_mutant_by_solvent_reuse(
            &self.engine,
            param_set(),
            &structure.inner,
            &opts,
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self::from_engine(engine))
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
    /// at checkpoints. An optional action applies the same 16-dimensional bias
    /// force as `step`, without paying for metrics on the current step.
    #[pyo3(signature = (action = None))]
    fn step_md<'py>(
        &mut self,
        py: Python<'py>,
        action: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.step_impl(py, action, false)
    }

    /// Fast MD step with optional low-frequency metrics.
    ///
    /// `metrics_every=0` is equivalent to `step_md` and never computes the
    /// expensive physical metrics. For a positive interval, metrics are
    /// included only on steps whose absolute `step_count` is divisible by the
    /// interval; callers can use the returned `metrics_available` flag rather
    /// than probing dictionary keys. This keeps the hot loop allocation and
    /// O(N²) metric cost out of most steps while retaining a single-call API.
    #[pyo3(signature = (action = None, metrics_every = 0))]
    fn step_fast<'py>(
        &mut self,
        py: Python<'py>,
        action: Option<PyReadonlyArray1<'_, f32>>,
        metrics_every: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        // `step_impl` advances the state before exposing `step_count`; compute
        // the schedule for the step that is about to be produced.
        let metrics_available = metrics_due(self.engine.state.step_count + 1, metrics_every);
        let result = self.step_impl(py, action, metrics_available)?;
        result.set_item("metrics_available", metrics_available)?;
        Ok(result)
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

    /// Add a harmonic distance restraint between two atoms (e.g. for AlphaFold 3 ligand/ion coordination).
    fn add_distance_restraint(&mut self, atom_0_idx: usize, atom_1_idx: usize, r0: f32, k: f32) {
        self.engine
            .add_distance_restraint(atom_0_idx, atom_1_idx, r0, k);
    }

    /// Re-target restraint #idx (SMD ramp); false if out of range.
    fn update_distance_restraint(&mut self, idx: usize, r0: f32, k: f32) -> bool {
        self.engine.update_distance_restraint(idx, r0, k)
    }

    /// Release all restraints (unbiased window begins on the next step).
    fn clear_distance_restraints(&mut self) {
        self.engine.clear_distance_restraints();
    }

    /// Integration timestep in ps. Solute X–H bonds ride SHAKE
    /// (`HydrogenConstraint::BondRigidConstraints`) and water SETTLE is
    /// dt-agnostic, so 0.004–0.005 is the standard rigid-H protocol
    /// (AMBER/CHARMM): a legal 2–2.5× sampling-rate lever. Verify T/P/energy
    /// drift at the new dt before production (tests probe this).
    fn set_timestep(&mut self, dt_ps: f32) {
        self.engine.dt_ps = dt_ps;
    }

    /// Current Cα coordinates `[L, 3]`.
    fn coords_ca<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let ca_indices = self.topology().ca_indices.clone();
        self.coords_scratch.clear();
        self.coords_scratch.extend(ca_indices.iter().map(|&i| {
            let p = self.engine.state.atoms[i].posit;
            vec![p.x, p.y, p.z]
        }));
        PyArray2::from_vec2(py, &self.coords_scratch)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// All atom positions `[n, 3]` in `state.atoms` order. The engine appends
    /// solvent/ions AFTER the solute during build, so the caller's known solute
    /// atom count slices the topology back out (`coords_all()[:n_solute]`).
    /// Why: two-segment MD protocols (Kaggle CPU 12 h cap) checkpoint between
    /// segments; CA-only frames cannot refeed the mutant builder, which needs
    /// the N/CA/C backbone to orient sidechain placement — full solute
    /// coordinates make the restore exact, sidechain rotamers included.
    fn coords_all<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let n = self.engine.state.atoms.len();
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(n);
        for a in self.engine.state.atoms.iter() {
            let p = a.posit;
            out.push(vec![p.x, p.y, p.z]);
        }
        PyArray2::from_vec2(py, &out).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Low-allocation flat Cα coordinates `[x0,y0,z0,...]`.
    fn coords_ca_flat<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let ca_indices = self.topology().ca_indices.clone();
        self.flat_coords_scratch.clear();
        self.flat_coords_scratch
            .extend(ca_indices.iter().flat_map(|&i| {
                let p = self.engine.state.atoms[i].posit;
                [p.x, p.y, p.z]
            }));
        Ok(PyArray1::from_vec(py, self.flat_coords_scratch.clone()))
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

    /// Energy bookkeeping split: bonded vs nonbonded (real + reciprocal PME
    /// live inside `nonbonded`). PFDE diagnostics (2026-09-23): total U read
    /// ~10x deeper than physical while per-residue MAX forces stayed in the
    /// normal 40-110 kcal/mol/A band -> decompose to find the accounting gap.
    fn energy_terms<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = &self.engine.state;
        let d = PyDict::new(py);
        d.set_item("total", s.potential_energy)?;
        d.set_item("nonbonded", s.potential_energy_nonbonded)?;
        d.set_item("bonded", s.potential_energy_bonded)?;
        Ok(d)
    }

    /// Pressure audit (v1.3.8): rigidly dilate the system by `lam` and return
    /// (potential_energy, total_virial_kcal, pressure_bar) at the scaled state.
    /// Python finite-differences ±lam against −∂E/∂lnV to check the reported
    /// virial buckets. NOTE: mutates the state (call as probe(s), probe(1/s)
    /// to return; f32 round-trip drift is ~ULP).
    fn debug_rigid_scale_probe(&mut self, lam: f64) -> (f64, f64, f64) {
        let dev = crate::engine::md_core::ComputationDevice::Cpu;
        self.engine.state.debug_rigid_scale_probe(lam, &dev)
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
            let r_com =
                (w.o.posit * w.o.mass + w.h0.posit * w.h0.mass + w.h1.posit * w.h1.mass) / m_total;
            let v_com =
                (w.o.vel * w.o.mass + w.h0.vel * w.h0.mass + w.h1.vel * w.h1.mass) / m_total;

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
                    [
                        mass * (r2 - r.x * r.x),
                        -mass * r.x * r.y,
                        -mass * r.x * r.z,
                    ],
                    [
                        -mass * r.y * r.x,
                        mass * (r2 - r.y * r.y),
                        -mass * r.y * r.z,
                    ],
                    [
                        -mass * r.z * r.x,
                        -mass * r.z * r.y,
                        mass * (r2 - r.z * r.z),
                    ],
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

            rigid_accum +=
                (m_total as f64) * (v_com.magnitude_squared() as f64) + l.dot(omega) as f64; // M·V² + L·ω = 2·(COM_KE + rot_KE)
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

    /// EFFECTIVE ionic strength of the built box, mol/L: I = ½Σcᵢzᵢ² over all
    /// inserted ions — NaCl background PLUS neutralizing counterions PLUS
    /// divalent salts (v1.3.7 getter for the audit in v1.3.5/1.3.6). Contrast
    /// with the INPUT `ionic_strength_m`, which is only the NaCl background
    /// molarity; for pure monovalent NaCl the two coincide.
    fn effective_ionic_strength_m(&self) -> f64 {
        self.engine.state.effective_ionic_strength_m() as f64
    }

    /// Last instantaneous pressure reading, bar (pair-virial + kinetic, same
    /// quantity the barostat consumes). Under NVT (pressure=0 at build) this
    /// is a pure diagnostic — a cold box legitimately reads large positive
    /// values; under NPT it is the live control signal. Returns 0.0 before
    /// the first completed step.
    fn pressure_bar(&self) -> f64 {
        self.engine.state.last_pressure_bar
    }

    /// Echo of the environment vector this engine was built with (the values
    /// AFTER range-clamping — e.g. a requested pH of 99 builds as 14).
    fn env_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let e = &self.engine.env;
        let d = PyDict::new(py);
        d.set_item("ph", e.ph)?;
        d.set_item("temp_k", e.temp_k)?;
        d.set_item("pressure_bar", e.pressure_bar)?;
        d.set_item("ionic_strength_m", e.ionic_strength_m)?;
        d.set_item("mg_cl2_m", e.mg_cl2_m)?;
        d.set_item("ca_cl2_m", e.ca_cl2_m)?;
        d.set_item("sr_cl2_m", e.sr_cl2_m)?;
        d.set_item("ba_cl2_m", e.ba_cl2_m)?;
        d.set_item("redox_reducing", e.redox_reducing)?;
        d.set_item("efield", e.efield.to_vec())?;
        d.set_item("efield_omega", e.efield_omega)?;
        Ok(d)
    }

    /// Topology-exclusion bookkeeping (sizes): `excluded_12_13` (1-2/1-3
    /// bonded pairs removed from nonbonded), `scaled_1_4`, `bonds_topology`,
    /// `angles`, `dihedrals`, `neighbor_pairs` (total std-std streams). If
    /// the first two look tiny next to the bonded tables, bonded pairs are
    /// leaking into nonbonded (huge positive short-range virial) — same
    /// diagnostic as `MdState::exclusion_diagnostics`.
    fn exclusion_diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (e12, s14, bonds, angles, dihedrals, pairs) = self.engine.state.exclusion_diagnostics();
        let d = PyDict::new(py);
        d.set_item("excluded_12_13", e12)?;
        d.set_item("scaled_1_4", s14)?;
        d.set_item("bonds_topology", bonds)?;
        d.set_item("angles", angles)?;
        d.set_item("dihedrals", dihedrals)?;
        d.set_item("neighbor_pairs", pairs)?;
        Ok(d)
    }

    /// Pressure-audit dump (v1.3.8 dev): every PME-relevant site (solute, ions,
    /// water O/H0/H1/M with their runtime charges and LJ params), the cell,
    /// cutoffs, alpha, the exclusion / scaled-14 pair lists, and the live
    /// virial buckets. Purpose: a brute-force external recomputation of the
    /// direct-space energy + pair virial must match the engine bucket sums
    /// class-by-class — the definitive audit for the +40 kbar anomaly.
    /// Diagnostic-only: not part of the stable API.
    fn debug_state_dump<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = &self.engine.state;
        let d = PyDict::new(py);
        d.set_item("cell", [s.cell.extent.x, s.cell.extent.y, s.cell.extent.z])?;
        d.set_item("alpha", s.cfg.spme_alpha)?;
        d.set_item("coulomb_cutoff", s.cfg.coulomb_cutoff)?;
        d.set_item("lj_cutoff", s.cfg.lj_cutoff)?;
        // kinds: 0 = std (solute/ions), 1 = water O, 2 = H0, 3 = H1, 4 = M(EP)
        let mut sites: Vec<Vec<f64>> = Vec::new();
        for (i, a) in s.atoms.iter().enumerate() {
            sites.push(vec![
                a.posit.x as f64,
                a.posit.y as f64,
                a.posit.z as f64,
                a.partial_charge as f64,
                a.lj_sigma as f64,
                a.lj_eps as f64,
                0.0,
                i as f64,
            ]);
        }
        for (j, w) in s.water.iter().enumerate() {
            for (site, kind) in [(&w.o, 1.0f64), (&w.h0, 2.0), (&w.h1, 3.0), (&w.m, 4.0)] {
                sites.push(vec![
                    site.posit.x as f64,
                    site.posit.y as f64,
                    site.posit.z as f64,
                    site.partial_charge as f64,
                    site.lj_sigma as f64,
                    site.lj_eps as f64,
                    kind,
                    j as f64 + 1.0,
                ]);
            }
        }
        d.set_item("n_std", s.atoms.len())?;
        d.set_item("sites", sites)?;
        // Live force array in the SAME site order — for external A/B force
        // audits (e.g. LAMMPS single-point comparison on the identical frame).
        let mut forces: Vec<Vec<f64>> = Vec::new();
        for a in s.atoms.iter() {
            forces.push(vec![a.force.x as f64, a.force.y as f64, a.force.z as f64]);
        }
        for w in s.water.iter() {
            for site in [&w.o, &w.h0, &w.h1, &w.m] {
                forces.push(vec![
                    site.force.x as f64,
                    site.force.y as f64,
                    site.force.z as f64,
                ]);
            }
        }
        d.set_item("forces", forces)?;
        let (ex, sc) = s.pairs_lists();
        d.set_item(
            "excluded_12_13",
            ex.iter().map(|&(i, j)| [i, j]).collect::<Vec<_>>(),
        )?;
        d.set_item(
            "scaled_1_4",
            sc.iter().map(|&(i, j)| [i, j]).collect::<Vec<_>>(),
        )?;
        // mirrors non_bonded::types (module-private constants)
        d.set_item("scale_coul_14", 1.0f64 / 1.2)?;
        d.set_item("scale_lj_14", 0.5f64)?;
        let (bond, short, long, constr) = s.virial_components();
        d.set_item("w_bonded", bond)?;
        d.set_item("w_short", short)?;
        d.set_item("w_long", long)?;
        d.set_item("w_constr", constr)?;
        d.set_item("ke_full", s.kinetic_energy)?;
        d.set_item("pressure_bar", s.last_pressure_bar)?;
        Ok(d)
    }

    // -- observability probes (v1.3.8): ESP / SASA / subset geometry -------

    /// Electrostatic potential at probe points, kcal/mol/e, through the
    /// engine's own PME (real-space erfc + reciprocal mesh, same packing and
    /// constants as the forces). Points are absolute box coordinates (Angstrom,
    /// wrapped internally). REPORT DIFFERENCES (phi(p) - phi(ref)): the PME
    /// constant offset (self + background) is position-independent and cancels;
    /// absolute values are gauge-arbitrary. Catalytic-axis field strength =
    /// (phi(b)-phi(a))/|b-a|; anion-hole potential / pH-dependent site
    /// potential are single points. Cost: one mesh build + O(points x spectrum).
    fn electrostatic_potential(&self, points: Vec<[f64; 3]>) -> Vec<f64> {
        self.engine.state.electrostatic_potential(&points)
    }

    /// Electrostatic FIELD **E = −∇φ** (kcal/mol/e/Angstrom) at probe points,
    /// one analytic PME pass (NOT a finite difference of the potential, so the
    /// gauge/background drift of phi does not amplify). This is the site-field
    /// functional probe (B2). `positions=Some(...)` (length == `pme_positions()`)
    /// evaluates on a time-averaged structure; omit for the current frame.
    #[pyo3(signature = (points, positions = None))]
    fn electrostatic_field(
        &self,
        points: Vec<[f64; 3]>,
        positions: Option<Vec<[f64; 3]>>,
    ) -> Vec<[f64; 3]> {
        self.engine
            .state
            .electrostatic_field(&points, positions.as_deref())
    }

    /// PME particle positions (atoms + water O/M/H + ions/cosolvent), the index
    /// space `electrostatic_field(positions=...)` expects. Snapshot+average over a
    /// window to get a time-averaged structure for the site-field query.
    fn pme_positions(&self) -> Vec<[f64; 3]> {
        self.engine.state.pme_positions()
    }

    /// Per-atom solvent-accessible surface area, Angstrom^2 (Shrake-Rupley):
    /// one value per `atom_labels()` / `select_atoms()` entry, same index
    /// space. A selection's SASA is `sum(sasa[i] for i in sel)`; MM/PBSA
    /// nonpolar term `G_np = gamma*SASA + beta`. Explicit water is NOT an
    /// occluder (the 1.4 A probe is the solvent model; feeding hydration
    /// water in would measure solute-water contact area, ~zero for any
    /// solvated surface). Ions/cosolvent atoms do occlude. O(N x n_sphere)
    /// with a neighbor grid.
    #[pyo3(signature = (probe_radius = 1.4, n_sphere = 200))]
    fn atom_sasa(&self, probe_radius: f64, n_sphere: usize) -> PyResult<Vec<f64>> {
        if !(0.5..=5.0).contains(&probe_radius) {
            return Err(PyValueError::new_err(format!(
                "probe_radius {probe_radius} A out of sane range [0.5, 5.0]"
            )));
        }
        Ok(self.engine.state.atom_sasa(probe_radius, n_sphere))
    }

    /// Atom-name list in `state.atoms` order: true PDB-style names for
    /// protein atoms ("CA", "NE2", "OG", "HH11") taken from the mmCIF via
    /// the topology; ions/cosolvent (outside the topology) report their
    /// `force_field_type` species label ("Na+", "Cl-", ...). This is the
    /// name space `select_atoms(names=...)` matches. (The raw per-atom
    /// `force_field_type` is an Amber ATOM TYPE like "2C"/"O2", not a name.)
    fn atom_names(&self) -> Vec<String> {
        self.engine.atom_names()
    }

    /// Residue-level atom selection returning `state.atoms` indices:
    /// `res_seq` = residue sequence ids (list); `names` = optional exact
    /// allow-list from `atom_names()`; `sidechain_heavy=True` keeps only
    /// non-H, non-backbone (N/CA/C/O/OXT) atoms. Example catalytic-site
    /// query: `select_atoms([57, 102], sidechain_heavy=True)`.
    #[pyo3(signature = (res_seq, names = None, sidechain_heavy = false))]
    fn select_atoms(
        &self,
        res_seq: Vec<i32>,
        names: Option<Vec<String>>,
        sidechain_heavy: bool,
    ) -> Vec<usize> {
        self.engine
            .select_atoms(&res_seq, names.as_deref(), sidechain_heavy)
    }

    /// Number of (a_i, b_j) atom pairs closer than `cutoff` Angstrom (minimum
    /// image; indices from `select_atoms`). Same list for both sets -> each
    /// unordered pair once, self-pairs skipped (use that for Q_site contact
    /// retention: contacts(set, set) before vs after). Water cannot be
    /// indexed here (it is outside the topology).
    fn contact_count(&self, set_a: Vec<usize>, set_b: Vec<usize>, cutoff: f64) -> PyResult<usize> {
        let n = self.engine.state.atoms.len();
        if let Some(&bad) = set_a.iter().chain(set_b.iter()).find(|&&i| i >= n) {
            return Err(PyValueError::new_err(format!(
                "atom index {bad} out of range (n_atoms = {n})"
            )));
        }
        Ok(self.engine.contact_count(&set_a, &set_b, cutoff))
    }

    /// Clearance profile along a polyline path: resampled every `spacing`
    /// Angstrom, distance to the nearest vdW surface (minus radius, minimum
    /// image). Returns `(profile, bottleneck)` where `bottleneck = min(profile)`
    /// is the largest probe radius that traverses the path: channel/pore
    /// radius in one call. `exclude` skips atom indices (the substrate whose
    /// channel you measure); `include_water=True` lets solvent line the path
    /// too (hydrated bottleneck; noisy frame to frame, average over frames).
    #[pyo3(signature = (path, spacing = 0.5, exclude = None, include_water = false))]
    fn bottleneck_radius(
        &self,
        path: Vec<[f64; 3]>,
        spacing: f64,
        exclude: Option<Vec<usize>>,
        include_water: bool,
    ) -> PyResult<(Vec<f64>, f64)> {
        if path.len() < 2 {
            return Err(PyValueError::new_err(
                "path needs at least 2 points ([start, end] straight line is fine)",
            ));
        }
        let profile = self.engine.bottleneck_profile(
            &path,
            spacing,
            exclude.as_deref().unwrap_or(&[]),
            include_water,
        );
        let bottleneck = profile.iter().cloned().fold(f64::INFINITY, f64::min);
        Ok((profile, bottleneck))
    }

    fn __repr__(&self) -> String {
        let e = &self.engine.env;
        format!(
            "<Engine n_res={} ph={} temp_k={} pressure_bar={} ionic_strength_m={} step={} t={}ps>",
            self.engine.topology.sequence.len(),
            e.ph,
            e.temp_k,
            e.pressure_bar,
            e.ionic_strength_m,
            self.engine.state.step_count,
            self.engine.state.time,
        )
    }

    #[pyo3(signature = (k, solute_k = None))]
    /// Bath setpoint; `solute_k` (dual-bath, v1.3.8) gives the solute its own
    /// Langevin setpoint while water keeps `k` — needed for physical annealing
    /// (single-bath solute relaxation is slower than any stage: v8b).
    fn set_temperature(&mut self, k: f32, solute_k: Option<f32>) {
        self.engine.set_temperature(k);
        self.engine.set_solute_temperature(solute_k);
    }

    /// Re-tune Langevin friction at runtime (dual-bath companion, v1.3.8):
    /// γ is the solute setpoint's stiffness — against a hot bath, γ=0.5 lets
    /// collision pumping win (v8c: solute_k 360 → solT 476), γ≈2 pins the
    /// solute near its target. No-op on non-Langevin integrators.
    fn set_langevin_gamma(&mut self, gamma: f32) {
        self.engine.set_langevin_gamma(gamma);
    }

    fn reset_velocities(&mut self) {
        self.engine.reset_velocities();
    }

    /// Arm the fail-fast trend monitor (v1.3.4). The engine then samples
    /// U (free), Rg and the kept backbone-H-bond fraction on `check_every`
    /// steps and, once `window` observations show ≥2 of 3 signals trending the
    /// wrong way (energy rising / Rg expanding / SS dissolving), returns
    /// `crashed=True` with `trend_alarm="<signal>"` from the next `step()`.
    /// Call it AFTER the structure you want to protect is in place. Leave the
    /// defaults for RL episodes (kills a diverging rollout at ~0.6 ps, before
    /// numerical blow-up). `skip_steps` leaves N early steps unobserved (the
    /// scan uses this for its per-segment equilibration prefix).
    /// `preset` (v1.3.7) takes the tuned config verbatim and ignores the
    /// knobs: `"rl_fail_fast"` = the RL-episode preset (0.6 ps window, z=8,
    /// raised floors — `TrendConfig::rl_fail_fast`); `"default"` = the
    /// ns-scale scan preset.
    #[pyo3(signature = (window = 60, check_every = 5, z_threshold = 3.0,
                        energy_floor_ps = 50.0, rg_floor_ps = 0.1, ss_floor_ps = 0.01,
                        skip_steps = 0, preset = None))]
    // Kwargs surface for Python (knobs + preset); wide by contract.
    #[allow(clippy::too_many_arguments)]
    fn set_trend_monitor(
        &mut self,
        window: usize,
        check_every: usize,
        z_threshold: f64,
        energy_floor_ps: f64,
        rg_floor_ps: f64,
        ss_floor_ps: f64,
        skip_steps: usize,
        preset: Option<&str>,
    ) -> PyResult<()> {
        let cfg = match preset {
            Some("rl_fail_fast") => crate::engine::TrendConfig::rl_fail_fast(),
            Some("default") => crate::engine::TrendConfig {
                enabled: true,
                ..Default::default()
            },
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "unknown trend preset '{other}' (expected rl_fail_fast | default, or pass knobs)"
                )));
            }
            None => crate::engine::TrendConfig {
                enabled: true,
                window,
                check_every,
                z_threshold,
                energy_floor_ps,
                rg_floor_ps,
                ss_floor_ps,
            },
        };
        self.engine.set_trend_monitor_skip(cfg, skip_steps);
        Ok(())
    }

    /// Empty the trend window without changing config (episode restart).
    fn reset_trend(&mut self) {
        self.engine.reset_trend();
    }

    /// Detach the trend monitor (no early termination).
    fn clear_trend_monitor(&mut self) {
        self.engine.clear_trend_monitor();
    }

    fn has_trend_monitor(&self) -> bool {
        self.engine.has_trend_monitor()
    }

    /// Switch the production integrator (diagnostics / thermostat tuning):
    ///   "langevin_middle" -> default LangevinMiddle gamma=0.5
    ///   "langevin_strong" -> LangevinMiddle gamma=10 (well-damped, settle-like)
    ///   "nve"             -> VerletVelocity with NO thermostat (energy-conservation probe)
    fn set_integrator(&mut self, mode: &str) -> PyResult<()> {
        let integrator = match mode {
            "langevin_middle" => crate::engine::md_core::Integrator::LangevinMiddle { gamma: 0.5 },
            "langevin_strong" => crate::engine::md_core::Integrator::LangevinMiddle { gamma: 10.0 },
            "nve" => crate::engine::md_core::Integrator::VerletVelocity { thermostat: None },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown integrator mode '{other}' (expected langevin_middle | langevin_strong | nve)"
                )));
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
    /// `add_hydrogens::resolve_h_clashes` uses to flag a clashing H (threshold
    /// 1.2 Å) — placement first tries to move/relax such H's on their parent
    /// bond sphere and only removes them as a last resort, so this shows which
    /// H's are hard clashes that slipped through. Returns
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
    ///
    /// The whole ramp+hold is NVT (barostat frozen): a cold build is tens of
    /// kbar overpressured and relieving that through NPT is a heat pump an
    /// RL rollout can't afford — see `EquilConfig` / `MdState::step`.
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
        d.set_item("last_md_us", self.last_md_us)?;
        d.set_item("last_metrics_us", self.last_metrics_us)?;
        d.set_item("steps", self.engine.state.step_count)?;
        Ok(d)
    }

    /// Fraction of residues currently receiving bias force.
    fn mask_fraction(&self) -> f32 {
        self.force.mask.fraction()
    }
}

fn metrics_due(step_count: usize, metrics_every: usize) -> bool {
    metrics_every != 0 && step_count % metrics_every == 0
}

#[cfg(test)]
mod fast_step_tests {
    use super::metrics_due;

    #[test]
    fn metrics_interval_is_disabled_at_zero() {
        assert!(!metrics_due(0, 0));
        assert!(!metrics_due(17, 0));
    }

    #[test]
    fn metrics_interval_uses_absolute_step_count() {
        assert!(metrics_due(20, 20));
        assert!(metrics_due(40, 20));
        assert!(!metrics_due(21, 20));
    }
}

impl PyEngine {
    /// Shared construction tail for build / mutate_with_solvent_reuse.
    fn from_engine(engine: SpiceEngine) -> Self {
        let n_res = engine.topology.sequence.len();
        let metrics = Metrics::new(&engine, MetricsConfig::default());
        let force = ForceAction::new(n_res, 16, 0.5, 20);
        Self {
            engine,
            force,
            metrics,
            last_md_us: 0,
            last_metrics_us: 0,
            coords_scratch: Vec::with_capacity(n_res),
            flat_coords_scratch: Vec::with_capacity(n_res * 3),
        }
    }

    /// Shared step driver. `want_metrics=false` skips the expensive
    /// `Metrics::compute` (O(N²) clash/surface + DSSP-lite) — the hot path for
    /// tight MD loops; metrics are then available via the `metrics()` method.
    fn step_impl<'py>(
        &mut self,
        py: Python<'py>,
        action: Option<PyReadonlyArray1<'_, f32>>,
        want_metrics: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let md_start = Instant::now();
        let result = match action {
            Some(a) => {
                let v = a
                    .as_slice()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                // v1.3.7: a wrong-length action used to be a debug_assert —
                // release builds panicked mid-loop (PanicException, or OOB
                // reads for longer arrays). Fail as a plain ValueError instead.
                if v.len() != self.force.m {
                    return Err(PyValueError::new_err(format!(
                        "action must have length M={} (engine's action dimension), got {}",
                        self.force.m,
                        v.len()
                    )));
                }
                self.force.step(&mut self.engine, v)
            }
            None => self.engine.step(None),
        };
        self.last_md_us = md_start.elapsed().as_micros() as u64;
        let metrics_start = Instant::now();
        let m = want_metrics.then(|| self.metrics.compute(&self.engine));
        self.last_metrics_us = if want_metrics {
            metrics_start.elapsed().as_micros() as u64
        } else {
            0
        };

        let d = PyDict::new(py);
        d.set_item("u_t_kcal", result.u_t_kcal)?;
        d.set_item("u_t_kj", result.u_t_kj)?;
        self.coords_scratch.clear();
        self.coords_scratch
            .extend(result.coords_ca.iter().map(|c| vec![c[0], c[1], c[2]]));
        d.set_item("coords_ca", PyArray2::from_vec2(py, &self.coords_scratch)?)?;
        d.set_item("step_count", result.step_count)?;
        d.set_item("time_ps", result.time_ps)?;
        d.set_item("crashed", result.crashed)?;
        d.set_item("crash_reason", result.crash_reason.clone())?;
        // Structured fail-fast verdict from an armed trend monitor (see
        // `set_trend_monitor`): "energy_rise" | "rg_expand" | "ss_loss" | None.
        // When present it implies crashed=True with a `trend_alarm:` reason.
        d.set_item("trend_alarm", result.trend_alarm)?;
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
        .map_err(PyValueError::new_err)
}

/// Validate that a sequence contains only standard amino acids.
#[pyfunction]
fn validate_sequence(seq: &str) -> PyResult<()> {
    crate::mutate::validate_sequence(seq).map_err(PyValueError::new_err)
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
        &crate::engine::md_core::ComputationDevice::Cpu,
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
        d.set_item("mg", p.env.mg_cl2_m)?;
        d.set_item("ca", p.env.ca_cl2_m)?;
        d.set_item("sr", p.env.sr_cl2_m)?;
        d.set_item("ba", p.env.ba_cl2_m)?;
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
    mg_molars = None,
    ca_molars = None,
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
    mg_molars: Option<Vec<f32>>,
    ca_molars: Option<Vec<f32>>,
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
        mg_molars: mg_molars.unwrap_or_else(|| vec![0.0]),
        ca_molars: ca_molars.unwrap_or_else(|| vec![0.0]),
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
    mg_range = None,
    ca_range = None,
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
    mg_range: Option<(f32, f32, f32)>,
    ca_range: Option<(f32, f32, f32)>,
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
    let grid = crate::domain::EnvGrid::from_ranges(
        temp_range,
        ph_range,
        pressure_range,
        ionic_range,
        mg_range,
        ca_range,
    );
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
    d.set_item("mg", e.mg_cl2_m).unwrap();
    d.set_item("ca", e.ca_cl2_m).unwrap();
    d.set_item("sr", e.sr_cl2_m).unwrap();
    d.set_item("ba", e.ba_cl2_m).unwrap();
    d.set_item("redox_reducing", e.redox_reducing).unwrap();
    d.set_item("efield", Vec::from(e.efield)).unwrap();
    d.set_item("efield_omega", e.efield_omega).unwrap();
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
    anchor_mg = 0.0,
    anchor_ca = 0.0,
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
    mg_step = None,
    mg_max = 10,
    ca_step = None,
    ca_max = 10,
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
    anchor_mg: f32,
    anchor_ca: f32,
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
    mg_step: Option<f32>,
    mg_max: usize,
    ca_step: Option<f32>,
    ca_max: usize,
    n_steps: usize,
    equil_steps: usize,
    repeats: usize,
    relax_iters: Option<usize>,
    tolerance: f32,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    use crate::domain::{Axis, AxisProbe, Direction};

    let anchor = EnvParams::new(anchor_ph, anchor_temp, anchor_pressure, anchor_ionic)
        .with_divalent(anchor_mg, anchor_ca, 0.0, 0.0);
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
    if let Some(s) = mg_step {
        probes.push(AxisProbe {
            axis: Axis::Mg,
            step: s,
            max_steps: mg_max,
            precision: Some(s),
        });
    }
    if let Some(s) = ca_step {
        probes.push(AxisProbe {
            axis: Axis::Ca,
            step: s,
            max_steps: ca_max,
            precision: Some(s),
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
        &crate::engine::md_core::ComputationDevice::Cpu,
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
            if r.direction == Direction::Positive {
                "+"
            } else {
                "-"
            },
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
                // v1.3.7: metric keys are ALWAYS present (NaN when no
                // metrics were computed) — the scan_stability convention.
                // Unguarded `p['m2']` readers (py_domain_grid.py) no longer
                // hit KeyError on pruned/pre-crash points.
                match &p.metrics {
                    Some(m) => {
                        pd.set_item("m1", m.m1).unwrap();
                        pd.set_item("m2", m.m2).unwrap();
                        pd.set_item("m3", m.m3).unwrap();
                        pd.set_item("m4", m.m4).unwrap();
                        pd.set_item("m5", m.m5).unwrap();
                        pd.set_item("rg", m.rg).unwrap();
                    }
                    None => {
                        let nan = f64::NAN;
                        for k in ["m1", "m2", "m3", "m4", "m5", "rg"] {
                            pd.set_item(k, nan).unwrap();
                        }
                    }
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
    m.add_class::<PyEnv>()?;
    m.add_class::<PyEngine>()?;
    m.add_function(wrap_pyfunction!(mutate_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(validate_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(scan_stability, m)?)?;
    m.add_function(wrap_pyfunction!(scan_stability_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(scan_radial, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
