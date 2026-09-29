//! Browser boundary for `spice_engine` (v1.3.9 web port; v1.3.10 FFI-parity
//! analysis/control surface — mirrors the diagnostics of `src/ffi.rs` minus the
//! RL-training loops, same engine calls, same key spellings).
//!
//! Exposes an `extern "C"` surface the demo loader drives. Deliberately NOT
//! wasm-bindgen: the offline build box has no `wasm-bindgen` CLI to post-process
//! its custom-section glue, so we hand-roll a raw ABI — integer handles into a
//! thread-local engine registry, `ptr,len` UTF-8 in, JSON bytes out through a
//! module scratch buffer, and exactly two JS imports (`env.now_ms`, `env.log`).
//! Everything else (build/step/observables) mirrors the pyo3 layer in `src/ffi.rs`
//! but calls the pure-Rust engine API directly (the core has always been pyo3-free;
//! `ffi.rs` is a single gated seam).
//!
//! Compiled only for `target_arch = "wasm32"` + feature `web`; native builds never
//! see this module. It adds no dependency on the crate graph.
//!
//! # Calling contract (JS side)
//! - Allocate JS→Rust buffers with `spice_alloc(n)`, fill them, pass `(ptr, n)`,
//!   then `spice_free(ptr, n)`.
//! - Every JSON-returning call leaves bytes in the OUT buffer; read them with
//!   `spice_out_ptr()`/`spice_out_len()` BEFORE the next call (the next call
//!   overwrites OUT). Copy into a JS string/typed array immediately.
//! - Positions live in a separate persistent scratch (`spice_positions_ptr/len`,
//!   `spice_roles_ptr/len`) refreshed by `spice_positions(handle)`; copy them
//!   each frame (wasm memory can grow and invalidate old views).
//! - All entry points are panic-guarded: a Rust panic returns a negative code and
//!   sets `{"error": "..."}` in OUT (read via `spice_last_error()`), never a trap.

use std::cell::RefCell;
use std::panic::{self, AssertUnwindSafe};

use lin_alg::f32::Vec3;
use serde_json::{Value, json};

use crate::builder::BuildOptions;
use crate::engine::SpiceEngine;
use crate::engine::md_core::{BarostatCfg, ComputationDevice, MdOverrides, params::FfParamSet};
use crate::env::EnvParams;
use crate::metrics::radius_of_gyration;
use crate::structure::{AtomInput, StructureInput, build_from_input};

// ---- JS imports (module "env") --------------------------------------------------

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    /// Monotonic millisecond clock (`performance.now()`); also feeds `clock.rs`.
    fn now_ms() -> f64;
    /// Write a UTF-8 log line to the JS console.
    fn log(ptr: *const u8, len: usize);
}

fn env_log(msg: &str) {
    // Safety: `log` is a pure JS binding over a borrowed slice we keep alive.
    unsafe { log(msg.as_ptr(), msg.len()) }
}

// ---- module state (thread-local: wasm is single-threaded, so no Sync needed) ----

thread_local! {
    static OUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static LAST_ERR: RefCell<String> = const { RefCell::new(String::new()) };
    static ENGINES: RefCell<Vec<Option<SpiceEngine>>> = const { RefCell::new(Vec::new()) };
    static POS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static ROLES: RefCell<Vec<i32>> = const { RefCell::new(Vec::new()) };
    /// Numeric results of the analysis probes (ESP/SASA/Ca/...): a second
    /// OUT channel so big float arrays avoid the JSON round-trip. Read with
    /// `spice_res_ptr/len` (f64 elements) before the next numeric call.
    static RES: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    /// Per-handle lazily-built `Metrics` (reference rg/ss snapshot at build
    /// time) — the same object `PyEngine` holds once in ffi.rs.
    static METS: RefCell<Vec<Option<crate::metrics::Metrics>>> =
        const { RefCell::new(Vec::new()) };
}

/// The Amber parameter set, lazily built from embedded data — same recipe as
/// `ffi.rs::param_set` but pyo3-free.
fn param_set() -> &'static FfParamSet {
    static PS: std::sync::OnceLock<FfParamSet> = std::sync::OnceLock::new();
    PS.get_or_init(|| FfParamSet::new_amber().expect("load amber params"))
}

fn set_out(v: &Value) {
    OUT.with(|o| {
        *o.borrow_mut() = v.to_string().into_bytes();
    });
}

fn set_out_str(s: &str) {
    OUT.with(|o| {
        *o.borrow_mut() = s.as_bytes().to_vec();
    });
}

fn set_err(msg: impl Into<String>) -> i32 {
    let m = msg.into();
    LAST_ERR.with(|e| *e.borrow_mut() = m.clone());
    let v = json!({ "error": m });
    set_out(&v);
    -1
}

/// Read a `(ptr,len)` UTF-8 buffer JS wrote into our memory.
fn read_str<'a>(ptr: *const u8, len: usize) -> Result<&'a str, String> {
    if ptr.is_null() {
        return Err("null pointer".into());
    }
    // Safety: JS allocated exactly `len` readable bytes via spice_alloc.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    std::str::from_utf8(bytes).map_err(|e| format!("invalid utf8: {e}"))
}

/// Run a body, converting any panic into an error code + OUT message.
fn guarded<F: FnOnce() -> i32>(f: F) -> i32 {
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(code) => code,
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic".to_string());
            set_err(format!("engine panic: {msg}"))
        }
    }
}

/// Borrow the `handle`-th live engine.
fn with_engine<T>(handle: i32, f: impl FnOnce(&mut SpiceEngine) -> T) -> Result<T, String> {
    if handle <= 0 {
        return Err("invalid handle".into());
    }
    let idx = (handle - 1) as usize;
    ENGINES.with(|reg| {
        let mut reg = reg.borrow_mut();
        let slot = reg
            .get_mut(idx)
            .and_then(|s| s.as_mut())
            .ok_or_else(|| "invalid handle".to_string())?;
        Ok(f(slot))
    })
}

/// Borrow the `handle`-th live engine immutably (the analysis probes never
/// mutate; ffi's `&self` methods port 1:1).
fn with_engine_ref<T>(handle: i32, f: impl FnOnce(&SpiceEngine) -> T) -> Result<T, String> {
    if handle <= 0 {
        return Err("invalid handle".into());
    }
    let idx = (handle - 1) as usize;
    ENGINES.with(|reg| {
        let reg = reg.borrow();
        let slot = reg
            .get(idx)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| "invalid handle".to_string())?;
        Ok(f(slot))
    })
}

fn set_res(v: Vec<f64>) -> i32 {
    RES.with(|r| *r.borrow_mut() = v);
    RES.with(|r| r.borrow().len() as i32)
}

/// Read a JS-written `f64` array (`count` doubles, not bytes).
fn read_f64<'a>(ptr: *const f64, count: usize) -> Result<&'a [f64], String> {
    if count == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err("null pointer".into());
    }
    // Safety: JS wrote `count` f64 into a spice_alloc buffer (wasm malloc
    // guarantees 8-byte alignment, same contract as the f32 step-action path).
    Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
}

/// Parse a JSON request body from a `(ptr,len)` UTF-8 buffer.
fn read_json(ptr: *const u8, len: usize) -> Result<Value, String> {
    let text = read_str(ptr, len)?;
    serde_json::from_str(text).map_err(|e| format!("bad json: {e}"))
}

// ---- lifecycle ----------------------------------------------------------------

/// Install the panic→console hook. Call once before anything else. Returns 0.
#[unsafe(no_mangle)]
pub extern "C" fn spice_init() -> i32 {
    panic::set_hook(Box::new(|info| {
        env_log(&format!("[spice panic] {info}"));
    }));
    // Safety: `now_ms` is a pure JS binding; touching it only asserts the
    // import resolves so the linker keeps it live. No aliasing involved.
    let _ = unsafe { now_ms() };
    0
}

/// Allocate a byte buffer of `len` with capacity == len, returning its address.
/// Free with `spice_free(ptr, len)`.
#[unsafe(no_mangle)]
pub extern "C" fn spice_alloc(len: usize) -> usize {
    let mut v: Vec<u8> = Vec::with_capacity(len);
    v.resize(len, 0);
    let ptr = v.as_mut_ptr() as usize;
    std::mem::forget(v);
    ptr
}

/// Reclaim a `spice_alloc` buffer. `ptr,len` must match the original.
///
/// # Safety
/// `ptr,len` must come from a live `spice_alloc(len)` and not be used after.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn spice_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(unsafe { Vec::from_raw_parts(ptr, len, len) });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn spice_out_ptr() -> usize {
    OUT.with(|o| o.borrow().as_ptr() as usize)
}

#[unsafe(no_mangle)]
pub extern "C" fn spice_out_len() -> usize {
    OUT.with(|o| o.borrow().len())
}

/// Copy the last error message into OUT as raw text (JSON-wrapped).
#[unsafe(no_mangle)]
pub extern "C" fn spice_last_error() -> i32 {
    let m = LAST_ERR.with(|e| e.borrow().clone());
    set_out(&json!({ "error": m }));
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn spice_version() -> i32 {
    set_out_str(env!("CARGO_PKG_VERSION"));
    0
}

// ---- build --------------------------------------------------------------------

/// Flat build params (all optional; serde defaults fill the rest). Mirrors the
/// core knobs of `Engine.build` without the pyo3 resolution layer.
#[derive(serde::Deserialize, Default)]
struct BuildParams {
    #[serde(default)]
    ph: Option<f32>,
    #[serde(default)]
    temp_k: Option<f32>,
    #[serde(default)]
    pressure_bar: Option<f32>,
    #[serde(default)]
    ionic_strength_m: Option<f32>,
    #[serde(default)]
    relax_iters: Option<usize>,
    #[serde(default)]
    tolerance: Option<f32>,
    #[serde(default)]
    strict_incomplete: Option<bool>,
    #[serde(default)]
    box_pad_a: Option<f32>,
    #[serde(default)]
    mg_cl2_m: Option<f32>,
    #[serde(default)]
    ca_cl2_m: Option<f32>,
    #[serde(default)]
    sr_cl2_m: Option<f32>,
    #[serde(default)]
    ba_cl2_m: Option<f32>,
    #[serde(default)]
    redox_reducing: Option<f32>,
    /// JSON array of cosolvent specs (see engine `cosolvent_presets`).
    #[serde(default)]
    cosolvents_json: Option<String>,
    /// JSON array of salt formula units (see engine `species`).
    #[serde(default)]
    salts_json: Option<String>,
    /// Static/oscillating external field [Ex,Ey,Ez] kcal/(mol·e·A) (clamped
    /// like the native `EnvParams::with_efield`).
    #[serde(default)]
    efield_json: Option<[f32; 3]>,
    #[serde(default)]
    efield_omega: Option<f32>,
}

fn build_engine_from_text(cif: &str, params: &str) -> Result<i32, String> {
    let p: BuildParams =
        serde_json::from_str(params).map_err(|e| format!("bad params json: {e}"))?;

    // Parse the mmCIF text and filter waters / non-amino-acid residues, the
    // same way `PyStructure::from_mmcif` does for the Python path.
    let mm = bio_files::MmCif::new(cif).map_err(|e| format!("mmcif parse: {e}"))?;
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
    if input.atoms.is_empty() {
        return Err("no amino-acid residues parsed from mmCIF".into());
    }

    let mut env = EnvParams::new_raw(
        p.ph.unwrap_or(7.0),
        p.temp_k.unwrap_or(310.0),
        p.pressure_bar.unwrap_or(0.0),
        p.ionic_strength_m.unwrap_or(0.0),
    );
    env.mg_cl2_m = p.mg_cl2_m.unwrap_or(0.0);
    env.ca_cl2_m = p.ca_cl2_m.unwrap_or(0.0);
    env.sr_cl2_m = p.sr_cl2_m.unwrap_or(0.0);
    env.ba_cl2_m = p.ba_cl2_m.unwrap_or(0.0);
    env.redox_reducing = p.redox_reducing.unwrap_or(0.0);
    if let Some(e) = p.efield_json {
        const MAX: f32 = crate::env::sane::EFIELD_MAX;
        env.efield = e.map(|v| v.clamp(-MAX, MAX));
    }
    if let Some(w) = p.efield_omega {
        env.efield_omega = w.max(0.0);
    }

    let mut opts = BuildOptions::default();
    opts.env = env;
    opts.relax_iters = Some(p.relax_iters.unwrap_or(2000));
    opts.energy_minimization_tolerance = p.tolerance.unwrap_or(2.0);
    opts.strict_incomplete_residues = p.strict_incomplete.unwrap_or(true);
    if let Some(pad) = p.box_pad_a {
        if pad > 0.0 {
            opts.box_padding_angstrom = pad;
        }
    }
    if let Some(j) = p.cosolvents_json.filter(|s| !s.is_empty()) {
        opts.cosolvents = crate::engine::md_core::cosolvent_presets::parse_cosolvents_json(&j)
            .map_err(|e| format!("cosolvents: {e}"))?;
    }
    if let Some(j) = p.salts_json.filter(|s| !s.is_empty()) {
        opts.salts = crate::engine::md_core::species::parse_salts_json(&j)
            .map_err(|e| format!("salts: {e}"))?;
    }

    let dev = ComputationDevice::Cpu;
    let engine = build_from_input(&dev, param_set(), &input, &opts)?;
    // Reference snapshot (rg/ss at build time) — same moment PyEngine::from_engine
    // builds it, so web `spice_metrics` matches Python `metrics()` semantics.
    let mets = crate::metrics::Metrics::new(&engine, crate::metrics::MetricsConfig::default());

    let n_atoms = engine.state.atoms.len();
    let n_water = engine.state.water.len();
    let n_residues = engine.topology.residues.len();
    let net_charge_e = engine.state.net_charge_e();
    let eff_ionic = engine.state.effective_ionic_strength_m();

    let handle = ENGINES.with(|reg| {
        let mut reg = reg.borrow_mut();
        reg.push(Some(engine));
        reg.len() as i32 // handle = index+1
    });
    METS.with(|m| m.borrow_mut().push(Some(mets)));

    set_out(&json!({
        "handle": handle,
        "n_atoms": n_atoms,
        "n_water": n_water,
        "n_sites": n_atoms + 3 * n_water,
        "n_residues": n_residues,
        "net_charge_e": net_charge_e,
        "effective_ionic_m": eff_ionic,
    }));
    Ok(handle)
}

/// Build from pasted mmCIF text + a params JSON. Returns the handle (>0) or a
/// negative code (message in OUT). OUT holds the build summary JSON on success.
#[unsafe(no_mangle)]
pub extern "C" fn spice_build_mmcif(
    cif_ptr: *const u8,
    cif_len: usize,
    params_ptr: *const u8,
    params_len: usize,
) -> i32 {
    guarded(|| {
        let cif = match read_str(cif_ptr, cif_len) {
            Ok(s) => s,
            Err(e) => return set_err(e),
        };
        let params = match read_str(params_ptr, params_len) {
            Ok(s) => s,
            Err(e) => return set_err(e),
        };
        match build_engine_from_text(cif, params) {
            Ok(h) => h,
            Err(e) => set_err(e),
        }
    })
}

// ---- stepping -----------------------------------------------------------------

/// Observables snapshot JSON for the current state (no step taken).
fn observables(e: &SpiceEngine) -> Value {
    let s = &e.state;
    let solute = species_temps(e);
    json!({
        "step_count": s.step_count,
        "time_ps": s.time,
        "dt_ps": e.dt_ps,
        "u_total_kcal": s.potential_energy,
        "u_nonbonded_kcal": s.potential_energy_nonbonded,
        "u_bonded_kcal": s.potential_energy_bonded,
        "ke_kcal": s.kinetic_energy,
        "t_kin": s.last_temperature_k,
        "pressure_bar": s.last_pressure_bar,
        "n_clamped": s.last_clamped_count,
        "max_accel_clamped": s.last_clamped_mag,
        "solute_t_k": solute[0],
        "water_t_k": solute[1],
        "rg": radius_of_gyration(e),
        "effective_ionic_m": s.effective_ionic_strength_m(),
        "net_charge_e": s.net_charge_e(),
    })
}

/// [solute_T, water_T] for the compact observables view — the full
/// six-number kernel is `species_stats` below (one DOF/KE implementation,
/// kept in lockstep with `ffi.rs::species_temperatures` deliberately; the
/// single-bath solute lag is the whole point of the dual-bath observability).
fn species_temps(e: &SpiceEngine) -> [f64; 2] {
    let v = species_stats(e);
    [v[4], v[5]]
}

/// Advance one step (no external force). OUT holds the step result + observables.
#[unsafe(no_mangle)]
pub extern "C" fn spice_step(handle: i32) -> i32 {
    guarded(|| {
        let res = with_engine(handle, |e| {
            let r = e.step(None);
            let mut o = observables(e);
            let obj = o.as_object_mut().unwrap();
            obj.insert("u_t_kcal".into(), json!(r.u_t_kcal));
            obj.insert("u_t_kj".into(), json!(r.u_t_kj));
            obj.insert("crashed".into(), json!(r.crashed));
            obj.insert("crash_reason".into(), json!(r.crash_reason));
            obj.insert("trend_alarm".into(), json!(r.trend_alarm));
            (if r.crashed { 2 } else { 0 }, o)
        });
        match res {
            Ok((code, o)) => {
                set_out(&o);
                code
            }
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// Step with a per-solute-atom bias force. `force` is `n_atoms*3` f32 (Å·amu/ps²)
/// laid out as the engine's atom array. Returns 0 (JSON as `spice_step`).
#[unsafe(no_mangle)]
pub extern "C" fn spice_step_action(handle: i32, force_ptr: *const f32, force_len: usize) -> i32 {
    guarded(|| {
        let want = with_engine(handle, |e| e.state.atoms.len());
        let want = match want {
            Ok(n) => n,
            Err(_) => return set_err("invalid handle"),
        };
        if force_len != want * 3 {
            return set_err(format!("force length {force_len} != atoms({want})*3"));
        }
        // Safety: JS wrote `force_len` f32 into an aligned spice_alloc buffer.
        let flat = unsafe { std::slice::from_raw_parts(force_ptr, force_len) };
        let forces: Vec<Vec3> = flat
            .chunks_exact(3)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .collect();
        let res = with_engine(handle, |e| {
            let r = e.step_borrowed(Some(&forces));
            let mut o = observables(e);
            let m = o.as_object_mut().unwrap();
            m.insert("crashed".into(), json!(r.crashed));
            m.insert("crash_reason".into(), json!(r.crash_reason));
            (if r.crashed { 2 } else { 0 }, o)
        });
        match res {
            Ok((code, o)) => {
                set_out(&o);
                code
            }
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// Current observables without stepping. OUT holds the JSON.
#[unsafe(no_mangle)]
pub extern "C" fn spice_observables(handle: i32) -> i32 {
    guarded(|| match with_engine(handle, |e| observables(e)) {
        Ok(o) => {
            set_out(&o);
            0
        }
        Err(_) => set_err("invalid handle"),
    })
}

// ---- positions ----------------------------------------------------------------

/// Refresh the persistent positions/roles scratch. Order: solute atoms (role
/// 0 heavy / 1 hydrogen), then per water O (2), H0 (3), H1 (3). Returns the
/// number of sites; read floats via `spice_positions_ptr/len` (n*3) and roles
/// via `spice_roles_ptr/len` (n).
#[unsafe(no_mangle)]
pub extern "C" fn spice_positions(handle: i32) -> i32 {
    let snapshot = with_engine(handle, |e| {
        let s = &e.state;
        let mut pos: Vec<f32> = Vec::with_capacity(s.atoms.len() * 3 + s.water.len() * 9);
        let mut roles: Vec<i32> = Vec::with_capacity(pos.capacity() / 3);
        for a in &s.atoms {
            pos.extend_from_slice(&[a.posit.x, a.posit.y, a.posit.z]);
            roles.push(if a.element == na_seq::Element::Hydrogen {
                1
            } else {
                0
            });
        }
        for w in &s.water {
            for (atom, role) in [(&w.o, 2i32), (&w.h0, 3), (&w.h1, 3)] {
                pos.extend_from_slice(&[atom.posit.x, atom.posit.y, atom.posit.z]);
                roles.push(role);
            }
        }
        (pos, roles)
    });
    match snapshot {
        Ok((pos, roles)) => {
            let n = roles.len();
            POS.with(|p| *p.borrow_mut() = pos);
            ROLES.with(|r| *r.borrow_mut() = roles);
            n as i32
        }
        Err(_) => set_err("invalid handle"),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn spice_positions_ptr() -> usize {
    POS.with(|p| p.borrow().as_ptr() as usize)
}
#[unsafe(no_mangle)]
pub extern "C" fn spice_positions_len() -> usize {
    POS.with(|p| p.borrow().len())
}
#[unsafe(no_mangle)]
pub extern "C" fn spice_roles_ptr() -> usize {
    ROLES.with(|r| r.borrow().as_ptr() as usize)
}
#[unsafe(no_mangle)]
pub extern "C" fn spice_roles_len() -> usize {
    ROLES.with(|r| r.borrow().len())
}

// ---- control ------------------------------------------------------------------

/// Set thermostat target. `solute_k < 0` means "single bath" (None); passing a
/// real value engages the dual bath (v1.3.8 folding protocol rule: always set
/// solute_k = T for folding/annealing probes).
#[unsafe(no_mangle)]
pub extern "C" fn spice_set_temperature(handle: i32, k: f32, solute_k: f32) -> i32 {
    guarded(|| {
        let r = with_engine(handle, move |e| {
            e.set_temperature(k);
            if solute_k >= 0.0 {
                e.set_solute_temperature(Some(solute_k));
            } else {
                e.set_solute_temperature(None);
            }
        });
        match r {
            Ok(()) => 0,
            Err(_) => set_err("invalid handle"),
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn spice_set_gamma(handle: i32, gamma: f32) -> i32 {
    guarded(
        || match with_engine(handle, move |e| e.set_langevin_gamma(gamma)) {
            Ok(()) => 0,
            Err(_) => set_err("invalid handle"),
        },
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn spice_set_timestep(handle: i32, dt_ps: f32) -> i32 {
    guarded(|| match with_engine(handle, move |e| e.dt_ps = dt_ps) {
        Ok(()) => 0,
        Err(_) => set_err("invalid handle"),
    })
}

/// Enable/resize the barostat at runtime. `p_bar <= 0` disables it (pure NVT).
#[unsafe(no_mangle)]
pub extern "C" fn spice_set_pressure(handle: i32, p_bar: f32) -> i32 {
    guarded(|| {
        let r = with_engine(handle, move |e| {
            if p_bar > 0.0 {
                let mut cfg = BarostatCfg::default();
                cfg.pressure_target = p_bar;
                e.state.cfg.barostat_cfg = Some(cfg);
            } else {
                e.state.cfg.barostat_cfg = None;
            }
        });
        match r {
            Ok(()) => 0,
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// Toy-physics toggle: individually disable force terms (the demo's "turn off
/// PME to see the lattice breathe" knob). Flags: bonded, coulomb, lj, long_range.
#[unsafe(no_mangle)]
pub extern "C" fn spice_set_force_overrides(
    handle: i32,
    bonded: i32,
    coulomb: i32,
    lj: i32,
    long_range: i32,
) -> i32 {
    guarded(|| {
        let r = with_engine(handle, move |e| {
            let o: &mut MdOverrides = &mut e.state.cfg.overrides;
            o.bonded_disabled = bonded != 0;
            o.coulomb_disabled = coulomb != 0;
            o.lj_disabled = lj != 0;
            o.long_range_recip_disabled = long_range != 0;
        });
        match r {
            Ok(()) => 0,
            Err(_) => set_err("invalid handle"),
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn spice_reset_velocities(handle: i32) -> i32 {
    guarded(|| match with_engine(handle, |e| e.reset_velocities()) {
        Ok(()) => 0,
        Err(_) => set_err("invalid handle"),
    })
}

/// Drop an engine, freeing its slot (the registry keeps the hole; handles never
/// renumber). Returns 0.
#[unsafe(no_mangle)]
pub extern "C" fn spice_free_engine(handle: i32) -> i32 {
    if handle <= 0 {
        return set_err("invalid handle");
    }
    let idx = (handle - 1) as usize;
    ENGINES.with(|reg| {
        let mut reg = reg.borrow_mut();
        if idx < reg.len() {
            reg[idx] = None;
        }
    });
    METS.with(|m| {
        let mut m = m.borrow_mut();
        if idx < m.len() {
            m[idx] = None;
        }
    });
    0
}

// ==== analysis probes + control parity (v1.3.10) ==============================
// Port of `src/ffi.rs`'s read-only diagnostics and runtime-control surface to
// the raw wasm ABI. Same engine calls, same JSON key spellings, so Python and
// JS scripts stay parallel. NOT ported: `mutate_with_solvent_reuse` (RL build
// optimization), the RL training glue (`mask_fraction`, the PyForce 16-dim
// action pre-mapping — JS passes per-atom bias arrays directly), and NumPy
// plumbing (typed arrays are the native form here).

#[unsafe(no_mangle)]
pub extern "C" fn spice_res_ptr() -> usize {
    RES.with(|r| r.borrow().as_ptr() as usize)
}
#[unsafe(no_mangle)]
pub extern "C" fn spice_res_len() -> usize {
    RES.with(|r| r.borrow().len())
}

/// Electrostatic potential (kcal/mol/e) at probe points through the engine's
/// own PME. `points` is a JS Float64Array of 3n doubles (absolute box coords,
/// Å). Result: 3n->n f64 into RES. Report DIFFERENCES (phi(p)-phi(ref)) — the
/// constant PME offset is gauge-arbitrary (see ffi docs).
#[unsafe(no_mangle)]
pub extern "C" fn spice_esp(handle: i32, points_ptr: *const f64, points_len: usize) -> i32 {
    guarded(|| {
        if points_len % 3 != 0 {
            return set_err("esp: points length must be a multiple of 3");
        }
        let flat = match read_f64(points_ptr, points_len) {
            Ok(v) => v,
            Err(e) => return set_err(e),
        };
        let points: Vec<[f64; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let out = with_engine_ref(handle, move |e| {
            e.state.electrostatic_potential(points.as_slice())
        });
        match out {
            Ok(v) => set_res(v),
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// Electrostatic FIELD E = -grad phi (kcal/mol/e/A), one analytic PME pass per
/// point. Optional `positions` (length == `spice_pme_positions` result/3)
/// evaluates a time-averaged structure; pass (0, 0) for the current frame.
/// RES holds 3n doubles (Ex,Ey,Ez per point).
#[unsafe(no_mangle)]
pub extern "C" fn spice_efield(
    handle: i32,
    points_ptr: *const f64,
    points_len: usize,
    positions_ptr: *const f64,
    positions_len: usize,
) -> i32 {
    guarded(|| {
        if points_len % 3 != 0 {
            return set_err("efield: points length must be a multiple of 3");
        }
        if positions_len != 0 && positions_len % 3 != 0 {
            return set_err("efield: positions length must be 0 or a multiple of 3");
        }
        let flat = match read_f64(points_ptr, points_len) {
            Ok(v) => v,
            Err(e) => return set_err(e),
        };
        let points: Vec<[f64; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let pos_flat = match read_f64(positions_ptr, positions_len) {
            Ok(v) => v.to_vec(),
            Err(e) => return set_err(e),
        };
        let out = with_engine_ref(handle, move |e| {
            let pos: Option<Vec<[f64; 3]>> = if pos_flat.is_empty() {
                None
            } else {
                Some(
                    pos_flat
                        .chunks_exact(3)
                        .map(|c| [c[0], c[1], c[2]])
                        .collect(),
                )
            };
            e.state
                .electrostatic_field(points.as_slice(), pos.as_deref())
        });
        match out {
            Ok(v) => set_res(v.into_iter().flatten().collect()),
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// PME charge-site positions — the exact engine packing: solute atoms (wrapped
/// to [0,L)), then per water M/H0/H1 (the uncharged O is excluded), ions and
/// cosolvent ride inside the atom list. This is the index space `spice_efield`'s
/// optional `positions` argument expects. RES: 3n (same count as positions).
#[unsafe(no_mangle)]
pub extern "C" fn spice_pme_positions(handle: i32) -> i32 {
    guarded(|| {
        let out = with_engine_ref(handle, |e| e.state.pme_positions());
        match out {
            Ok(v) => set_res(v.into_iter().flatten().collect()),
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// Per-atom SASA (Shrake-Rupley, A^2), one value per `state.atoms` entry
/// (solute + ions + cosolvent; explicit water is NOT an occluder). `probe`
/// must sit in [0.5, 5.0]; `n_sphere` is clamped to >= 50 by the engine.
#[unsafe(no_mangle)]
pub extern "C" fn spice_atom_sasa(handle: i32, probe: f64, n_sphere: i32) -> i32 {
    guarded(|| {
        if !(0.5..=5.0).contains(&probe) {
            return set_err(format!(
                "probe_radius {probe} A out of sane range [0.5, 5.0]"
            ));
        }
        let n = n_sphere.max(1) as usize;
        let out = with_engine_ref(handle, move |e| e.state.atom_sasa(probe, n));
        match out {
            Ok(v) => set_res(v),
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// Current Ca coordinates, RES: 3 * n_residues doubles (f32 storage widened).
#[unsafe(no_mangle)]
pub extern "C" fn spice_coords_ca(handle: i32) -> i32 {
    guarded(|| {
        let out = with_engine_ref(handle, |e| {
            e.topology
                .ca_indices
                .iter()
                .map(|&i| {
                    let p = e.state.atoms[i].posit;
                    [p.x, p.y, p.z]
                })
                .collect::<Vec<[f32; 3]>>()
        });
        match out {
            Ok(v) => set_res(v.into_iter().flatten().map(f64::from).collect()),
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// Time-averaged Ca (pseudo-labels). RES: 3 * n_residues. `engine.step()`
/// accumulates the window automatically (same as through the Python FFI).
#[unsafe(no_mangle)]
pub extern "C" fn spice_pseudo_labels(handle: i32) -> i32 {
    guarded(|| {
        let out = with_engine_ref(handle, |e| e.time_averaged_ca());
        match out {
            Ok(v) => set_res(v.into_iter().flatten().map(f64::from).collect()),
            Err(_) => set_err("invalid handle"),
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn spice_reset_pseudo_labels(handle: i32) -> i32 {
    guarded(|| match with_engine(handle, |e| e.reset_pseudo_labels()) {
        Ok(()) => 0,
        Err(_) => set_err("invalid handle"),
    })
}

/// Per-residue max |force| (kcal/mol/A) — strain map for targeted mutation.
/// RES: n_residues.
#[unsafe(no_mangle)]
pub extern "C" fn spice_per_residue_max_force(handle: i32) -> i32 {
    guarded(|| {
        let out = with_engine_ref(handle, |e| {
            let n = e.state.atoms.len();
            let mut max_f = vec![0.0f32; e.topology.residues.len()];
            for (ri, r) in e.topology.residues.iter().enumerate() {
                let mut m = 0.0f32;
                for &i in &r.atom_indices {
                    if i < n {
                        let fmag = e.state.atoms[i].force.magnitude();
                        if fmag > m {
                            m = fmag;
                        }
                    }
                }
                max_f[ri] = m;
            }
            max_f
        });
        match out {
            Ok(v) => set_res(v.into_iter().map(f64::from).collect()),
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// Flat `[solute_ke, water_ke, solute_dof, water_dof, solute_t, water_t]` —
/// one implementation behind both `species_temperatures` (full JSON, ffi-
/// identical keys) and the compact `solute_t_k`/`water_t_k` in observables.
fn species_stats(e: &SpiceEngine) -> [f64; 6] {
    const NATIVE_TO_KCAL: f64 = 1.0 / 418.4;
    const R_KCAL: f64 = 0.001_987_204_1;
    let s = &e.state;

    let mut solute_ke_native = 0.0f64;
    let mut n_solute = 0usize;
    let mut n_solute_h = 0usize;
    for a in &s.atoms {
        if a.static_ {
            continue;
        }
        n_solute += 1;
        if a.element == na_seq::Element::Hydrogen {
            n_solute_h += 1;
        }
        solute_ke_native += (a.mass as f64) * a.vel.magnitude_squared() as f64;
    }
    let solute_ke = 0.5 * solute_ke_native * NATIVE_TO_KCAL;

    let mut water_ke_native = 0.0f64;
    for w in &s.water {
        for atom in [&w.o, &w.h0, &w.h1] {
            water_ke_native += (atom.mass as f64) * atom.vel.magnitude_squared() as f64;
        }
    }
    let water_ke = 0.5 * water_ke_native * NATIVE_TO_KCAL;

    // Solute DOF: 3 per non-static atom minus 1 per constrained H — mirrors
    // dynamics' dof_for_thermo (ffi.rs comment kept in lockstep deliberately).
    let solute_dof = (3 * n_solute - n_solute_h) as f64;
    let water_dof = (6 * s.water.len()) as f64;
    let solute_t = if solute_dof > 0.0 {
        2.0 * solute_ke / (solute_dof * R_KCAL)
    } else {
        0.0
    };
    let water_t = if water_dof > 0.0 {
        2.0 * water_ke / (water_dof * R_KCAL)
    } else {
        0.0
    };
    [
        solute_ke, water_ke, solute_dof, water_dof, solute_t, water_t,
    ]
}

#[unsafe(no_mangle)]
pub extern "C" fn spice_species_temperatures(handle: i32) -> i32 {
    guarded(|| {
        let out = with_engine_ref(handle, |e| species_stats(e));
        match out {
            Ok(v) => {
                set_out(&json!({
                    "solute_ke_kcal": v[0], "water_ke_kcal": v[1],
                    "solute_dof": v[2], "water_dof": v[3],
                    "solute_t_k": v[4], "water_t_k": v[5],
                }));
                0
            }
            Err(_) => set_err("invalid handle"),
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn spice_energy_terms(handle: i32) -> i32 {
    guarded(|| {
        let out = with_engine_ref(handle, |e| {
            let s = &e.state;
            json!({
                "total": s.potential_energy,
                "nonbonded": s.potential_energy_nonbonded,
                "bonded": s.potential_energy_bonded,
            })
        });
        match out {
            Ok(v) => {
                set_out(&v);
                0
            }
            Err(_) => set_err("invalid handle"),
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn spice_thermo_info(handle: i32) -> i32 {
    guarded(|| {
        const R_KCAL: f64 = 0.001_987_204_1;
        let out = with_engine_ref(handle, |e| {
            let s = &e.state;
            let n_atoms = s.atoms.len();
            let n_static = s.atoms.iter().filter(|a| a.static_).count();
            let n_h = s
                .atoms
                .iter()
                .filter(|a| a.element == na_seq::Element::Hydrogen && !a.static_)
                .count();
            let n_water = s.water.len();
            let thermo_dof = s.thermo_dof();
            let ke = s.kinetic_energy;
            let t_implied = if thermo_dof > 0 {
                2.0 * ke / (thermo_dof as f64 * R_KCAL)
            } else {
                0.0
            };
            json!({
                "n_atoms": n_atoms, "n_static": n_static, "n_hydrogens": n_h,
                "n_water": n_water, "thermo_dof": thermo_dof,
                "dof_for_thermo_now": s.dof_for_thermo_now(),
                "dof_water_6n": 6 * n_water, "dof_solute_3n": 3 * (n_atoms - n_static),
                "kinetic_energy_kcal": ke, "t_implied_k": t_implied,
            })
        });
        match out {
            Ok(v) => {
                set_out(&v);
                0
            }
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// SETTLE sanity probe: water KE split into rigid-body vs internal. If
/// internal is significant, constraints leak DOF and water_t is inflated
/// (the v1.3.8 OU calibration used exactly this).
#[unsafe(no_mangle)]
pub extern "C" fn spice_water_rigid_split(handle: i32) -> i32 {
    guarded(|| {
        const NATIVE_TO_KCAL: f64 = 1.0 / 418.4;
        const R_KCAL: f64 = 0.001_987_204_1;
        let out = with_engine_ref(handle, |e| {
            use lin_alg::f32::Mat3 as Mat3F32;
            let s = &e.state;
            let mut total_accum = 0.0f64;
            let mut rigid_accum = 0.0f64;
            let mut n_water = 0usize;
            for w in &s.water {
                n_water += 1;
                let m_total = w.o.mass + w.h0.mass + w.h1.mass;
                let r_com =
                    (w.o.posit * w.o.mass + w.h0.posit * w.h0.mass + w.h1.posit * w.h1.mass)
                        / m_total;
                let v_com =
                    (w.o.vel * w.o.mass + w.h0.vel * w.h0.mass + w.h1.vel * w.h1.mass) / m_total;
                for atom in [&w.o, &w.h0, &w.h1] {
                    total_accum += (atom.mass as f64) * atom.vel.magnitude_squared() as f64;
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
                let omega = Mat3F32::from_arr(i_arr).solve_system(l);
                rigid_accum +=
                    (m_total as f64) * (v_com.magnitude_squared() as f64) + l.dot(omega) as f64;
            }
            let total_ke = 0.5 * total_accum * NATIVE_TO_KCAL;
            let rigid_ke = 0.5 * rigid_accum * NATIVE_TO_KCAL;
            let internal_ke = total_ke - rigid_ke;
            let tw = |ke: f64, dof: f64| {
                if n_water > 0 {
                    2.0 * ke / (dof * R_KCAL)
                } else {
                    0.0
                }
            };
            json!({
                "water_total_ke_kcal": total_ke, "water_rigid_ke_kcal": rigid_ke,
                "water_internal_ke_kcal": internal_ke, "n_water": n_water,
                "water_rigid_t_k": tw(rigid_ke, (6 * n_water) as f64),
                "water_9dof_t_k": tw(total_ke, (9 * n_water) as f64),
                "water_internal_t_k": tw(internal_ke, (3 * n_water) as f64),
            })
        });
        match out {
            Ok(v) => {
                set_out(&v);
                0
            }
            Err(_) => set_err("invalid handle"),
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn spice_env_info(handle: i32) -> i32 {
    guarded(|| {
        let out = with_engine_ref(handle, |e| {
            let env = &e.env;
            json!({
                "ph": env.ph, "temp_k": env.temp_k, "pressure_bar": env.pressure_bar,
                "ionic_strength_m": env.ionic_strength_m,
                "mg_cl2_m": env.mg_cl2_m, "ca_cl2_m": env.ca_cl2_m,
                "sr_cl2_m": env.sr_cl2_m, "ba_cl2_m": env.ba_cl2_m,
                "redox_reducing": env.redox_reducing,
                "efield": env.efield, "efield_omega": env.efield_omega,
            })
        });
        match out {
            Ok(v) => {
                set_out(&v);
                0
            }
            Err(_) => set_err("invalid handle"),
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn spice_exclusion_diagnostics(handle: i32) -> i32 {
    guarded(|| {
        let out = with_engine_ref(handle, |e| {
            let (e12, s14, bonds, angles, dihedrals, pairs) = e.state.exclusion_diagnostics();
            json!({
                "excluded_12_13": e12, "scaled_1_4": s14, "bonds_topology": bonds,
                "angles": angles, "dihedrals": dihedrals, "neighbor_pairs": pairs,
            })
        });
        match out {
            Ok(v) => {
                set_out(&v);
                0
            }
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// Pressure-audit dump (diagnostic-only, not a stable API): every PME-relevant
/// site with live charges/LJ, the cell + cutoffs + alpha, exclusion/scaled-14
/// pair lists, live per-site forces, and the virial buckets. External brute-
/// force recomputations (LAMMPS referee) match against this. JSON is large —
/// fine for the demo scale, avoid polling it every frame on 2LYZ.
#[unsafe(no_mangle)]
pub extern "C" fn spice_debug_state_dump(handle: i32) -> i32 {
    guarded(|| {
        let out = with_engine_ref(handle, |e| {
            let s = &e.state;
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
            let (ex, sc) = s.pairs_lists();
            let (bond, short, long, constr) = s.virial_components();
            json!({
                "cell": [s.cell.extent.x, s.cell.extent.y, s.cell.extent.z],
                "alpha": s.cfg.spme_alpha,
                "coulomb_cutoff": s.cfg.coulomb_cutoff,
                "lj_cutoff": s.cfg.lj_cutoff,
                "n_std": s.atoms.len(),
                "sites": sites,
                "forces": forces,
                "excluded_12_13": ex.iter().map(|&(i, j)| [i, j]).collect::<Vec<_>>(),
                "scaled_1_4": sc.iter().map(|&(i, j)| [i, j]).collect::<Vec<_>>(),
                "scale_coul_14": 1.0f64 / 1.2,
                "scale_lj_14": 0.5f64,
                "w_bonded": bond, "w_short": short, "w_long": long, "w_constr": constr,
                "ke_full": s.kinetic_energy,
                "pressure_bar": s.last_pressure_bar,
            })
        });
        match out {
            Ok(v) => {
                set_out(&v);
                0
            }
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// True PDB-style atom names in `state.atoms` order (ions/cosolvent report
/// their force_field_type label). OUT: JSON array of strings.
#[unsafe(no_mangle)]
pub extern "C" fn spice_atom_names(handle: i32) -> i32 {
    guarded(|| {
        let out = with_engine_ref(handle, |e| e.atom_names());
        match out {
            Ok(v) => {
                set_out(&Value::Array(v.into_iter().map(Value::from).collect()));
                0
            }
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// Per-atom (element, one-letter residue, seq_id, mmCIF serial) — identify
/// atoms that hit the accel clamp or a clash row.
#[unsafe(no_mangle)]
pub extern "C" fn spice_atom_labels(handle: i32) -> i32 {
    guarded(|| {
        let out = with_engine_ref(handle, |e| {
            let n = e.state.atoms.len();
            let mut res_of: Vec<(char, i32)> = vec![('?', 0); n];
            for r in &e.topology.residues {
                for &i in &r.atom_indices {
                    if i < n {
                        res_of[i] = (r.one_letter, r.seq_id);
                    }
                }
            }
            e.state
                .atoms
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    json!({
                        "element": format!("{:?}", a.element),
                        "residue": res_of[i].0.to_string(),
                        "seq_id": res_of[i].1,
                        "serial": a.serial_number,
                    })
                })
                .collect::<Vec<_>>()
        });
        match out {
            Ok(v) => {
                set_out(&Value::Array(v));
                0
            }
            Err(_) => set_err("invalid handle"),
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn spice_sequence(handle: i32) -> i32 {
    guarded(
        || match with_engine_ref(handle, |e| e.topology.sequence.clone()) {
            Ok(seq) => {
                set_out_str(&seq);
                0
            }
            Err(_) => set_err("invalid handle"),
        },
    )
}

/// Selection query. IN JSON: {"res_seq":[i32...], "names":[...]?,
/// "sidechain_heavy":bool?}. OUT: JSON array of `state.atoms` indices.
#[unsafe(no_mangle)]
pub extern "C" fn spice_select_atoms(handle: i32, spec_ptr: *const u8, spec_len: usize) -> i32 {
    guarded(|| {
        let spec = match read_json(spec_ptr, spec_len) {
            Ok(v) => v,
            Err(e) => return set_err(e),
        };
        let res_seq: Vec<i32> = match serde_json::from_value(
            spec.get("res_seq").cloned().unwrap_or(Value::Array(vec![])),
        ) {
            Ok(v) => v,
            Err(e) => return set_err(format!("res_seq: {e}")),
        };
        let names: Option<Vec<String>> = match spec.get("names") {
            Some(v) if !v.is_null() => match serde_json::from_value(v.clone()) {
                Ok(n) => Some(n),
                Err(e) => return set_err(format!("names: {e}")),
            },
            _ => None,
        };
        let sidechain_heavy = spec
            .get("sidechain_heavy")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let out = with_engine_ref(handle, move |e| {
            e.select_atoms(&res_seq, names.as_deref(), sidechain_heavy)
        });
        match out {
            Ok(v) => {
                set_out(&Value::Array(
                    v.into_iter().map(|i| json!(i)).collect::<Vec<_>>(),
                ));
                0
            }
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// Pair contact count under minimum image. IN JSON: {"a":[usize],"b":[usize],
/// "cutoff":f64}. Same set twice = unordered pairs once, self skipped. OUT
/// {"count":n}. Rejects out-of-range indices (engine would panic).
#[unsafe(no_mangle)]
pub extern "C" fn spice_contact_count(handle: i32, spec_ptr: *const u8, spec_len: usize) -> i32 {
    guarded(|| {
        let spec = match read_json(spec_ptr, spec_len) {
            Ok(v) => v,
            Err(e) => return set_err(e),
        };
        let read = |k: &str| -> Result<Vec<usize>, String> {
            serde_json::from_value(spec.get(k).cloned().unwrap_or(Value::Array(vec![])))
                .map_err(|e| format!("{k}: {e}"))
        };
        let a = match read("a") {
            Ok(v) => v,
            Err(e) => return set_err(e),
        };
        let b = match read("b") {
            Ok(v) => v,
            Err(e) => return set_err(e),
        };
        let cutoff = spec.get("cutoff").and_then(|v| v.as_f64()).unwrap_or(4.0);
        let out = with_engine_ref(handle, move |e| {
            let n = e.state.atoms.len();
            match a.iter().chain(b.iter()).find(|&&i| i >= n) {
                Some(&bad) => Err(format!("atom index {bad} out of range (n_atoms = {n})")),
                None => Ok(e.contact_count(&a, &b, cutoff)),
            }
        });
        match out {
            Ok(Ok(count)) => {
                set_out(&json!({ "count": count }));
                0
            }
            Ok(Err(msg)) => set_err(msg),
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// Channel/pore clearance profile. IN JSON: {"path":[[x,y,z],...], "spacing"?:0.5,
/// "exclude"?:[usize], "include_water"?:bool}. OUT {"profile":[...],"bottleneck":f64}.
#[unsafe(no_mangle)]
pub extern "C" fn spice_bottleneck(handle: i32, spec_ptr: *const u8, spec_len: usize) -> i32 {
    guarded(|| {
        let spec = match read_json(spec_ptr, spec_len) {
            Ok(v) => v,
            Err(e) => return set_err(e),
        };
        let path: Vec<[f64; 3]> =
            match serde_json::from_value(spec.get("path").cloned().unwrap_or(Value::Array(vec![])))
            {
                Ok(p) => p,
                Err(e) => return set_err(format!("path: {e}")),
            };
        if path.len() < 2 {
            return set_err("path needs at least 2 points ([start, end] is fine)");
        }
        let spacing = spec.get("spacing").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let exclude: Vec<usize> = match serde_json::from_value(
            spec.get("exclude").cloned().unwrap_or(Value::Array(vec![])),
        ) {
            Ok(x) => x,
            Err(e) => return set_err(format!("exclude: {e}")),
        };
        let include_water = spec
            .get("include_water")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let out = with_engine_ref(handle, move |e| {
            e.bottleneck_profile(&path, spacing, &exclude, include_water)
        });
        match out {
            Ok(profile) => {
                let bottleneck = profile.iter().cloned().fold(f64::INFINITY, f64::min);
                set_out(&json!({ "profile": profile, "bottleneck": bottleneck }));
                0
            }
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// The five physical metrics (ffi.rs `metrics()` keys). The reference rg/ss
/// snapshot is captured at build time (same as PyEngine), so call this after
/// the structure you want to protect is in place.
#[unsafe(no_mangle)]
pub extern "C" fn spice_metrics(handle: i32) -> i32 {
    guarded(|| {
        let idx = (handle - 1) as usize;
        let out = with_engine_ref(handle, move |e| {
            METS.with(|m| {
                let m = m.borrow();
                m.get(idx)
                    .and_then(|slot| slot.as_ref())
                    .map(|met| met.compute(e))
            })
        });
        match out {
            Ok(Some(r)) => {
                set_out(&json!({
                    "m1": r.m1, "m2": r.m2, "m3": r.m3, "m4": r.m4, "m5": r.m5,
                    "rg": r.rg, "u_t_kcal": r.u_t_kcal,
                    "n_ss_ref": r.n_ss_ref, "n_ss_kept": r.n_ss_kept,
                    "n_surface_charged": r.n_surface_charged,
                    "stability_margin": r.stability_margin, "rmsf": r.rmsf,
                }));
                0
            }
            Ok(None) => set_err("invalid handle"),
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// Engine-internal per-phase timing sums (µs) — where a wasm step actually
/// goes. (ffi's `last_md_us`/`last_metrics_us` are Python-side timers; time
/// `spice_step` yourself on the JS side if you need per-call numbers.)
#[unsafe(no_mangle)]
pub extern "C" fn spice_computation_time(handle: i32) -> i32 {
    guarded(|| {
        let out = with_engine_ref(handle, |e| {
            let ct = &e.state.computation_time;
            json!({
                "bonded_us": ct.bonded_sum,
                "nonbonded_short_us": ct.non_bonded_short_range_sum,
                "ewald_long_us": ct.ewald_long_range_sum,
                "neighbor_all_us": ct.neighbor_all_sum,
                "neighbor_rebuild_us": ct.neighbor_rebuild_sum,
                "neighbor_rebuild_count": ct.neighbor_rebuild_count,
                "integration_us": ct.integration_sum,
                "ambient_us": ct.ambient_sum,
                "snapshot_us": ct.snapshot_sum,
                "total_us": ct.total,
                "steps": e.state.step_count,
            })
        });
        match out {
            Ok(v) => {
                set_out(&v);
                0
            }
            Err(_) => set_err("invalid handle"),
        }
    })
}

fn force_rows(handle: i32, min_force: f32, hydrogens_only: bool) -> Result<Value, String> {
    with_engine_ref(handle, move |e| {
        let n = e.state.atoms.len();
        let mut res_of: Vec<(char, i32)> = vec![('?', 0); n];
        let mut atom_res: Vec<usize> = vec![0; n];
        for (ri, r) in e.topology.residues.iter().enumerate() {
            for &i in &r.atom_indices {
                if i < n {
                    res_of[i] = (r.one_letter, r.seq_id);
                    atom_res[i] = ri;
                }
            }
        }
        let atoms = &e.state.atoms;
        let mut out: Vec<Value> = Vec::new();
        for i in 0..n {
            let a = &atoms[i];
            if hydrogens_only && a.element != na_seq::Element::Hydrogen {
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
            out.push(json!({
                "element": format!("{:?}", a.element),
                "residue": res_of[i].0.to_string(),
                "seq_id": res_of[i].1,
                "serial": a.serial_number,
                "force": fmag,
                "min_d": min_d,
            }));
        }
        out.sort_by(|x, y| {
            y["force"]
                .as_f64()
                .partial_cmp(&x["force"].as_f64())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Value::Array(out)
    })
}

/// Hydrogens with |force| >= min_force + nearest NON-bonded distance (the
/// resolve_h_clashes 1.2 A flag). O(N^2) — demo scale only.
#[unsafe(no_mangle)]
pub extern "C" fn spice_clash_report(handle: i32, min_force: f32) -> i32 {
    guarded(|| match force_rows(handle, min_force, true) {
        Ok(v) => {
            set_out(&v);
            0
        }
        Err(_) => set_err("invalid handle"),
    })
}

/// Every atom with |force| >= min_force + nearest NON-bonded distance.
#[unsafe(no_mangle)]
pub extern "C" fn spice_force_report(handle: i32, min_force: f32) -> i32 {
    guarded(|| match force_rows(handle, min_force, false) {
        Ok(v) => {
            set_out(&v);
            0
        }
        Err(_) => set_err("invalid handle"),
    })
}

/// Pressure audit: rigidly dilate by `lam` and return (u, total virial, P) at
/// the scaled state WITHOUT stepping. MUTATES forces/PE — re-call with 1.0 or
/// re-step to restore. W excludes the constraint bucket by design.
#[unsafe(no_mangle)]
pub extern "C" fn spice_rigid_scale_probe(handle: i32, lam: f64) -> i32 {
    guarded(|| {
        let out = with_engine(handle, move |e| {
            let dev = ComputationDevice::Cpu;
            e.state.debug_rigid_scale_probe(lam, &dev)
        });
        match out {
            Ok((u, w, p)) => {
                set_out(&json!({ "u_kcal": u, "virial_kcal": w, "pressure_bar": p }));
                0
            }
            Err(_) => set_err("invalid handle"),
        }
    })
}

// ---- control parity ------------------------------------------------------------

/// Hot integrator switch: "langevin_middle" (default gamma 0.5) |
/// "langevin_strong" (gamma 10) | "nve" (VerletVelocity, no thermostat —
/// energy-conservation probe).
#[unsafe(no_mangle)]
pub extern "C" fn spice_set_integrator(handle: i32, mode_ptr: *const u8, mode_len: usize) -> i32 {
    guarded(|| {
        let mode = match read_str(mode_ptr, mode_len) {
            Ok(s) => s.to_string(),
            Err(e) => return set_err(e),
        };
        let r = with_engine(handle, move |e| {
            let integrator = match mode.as_str() {
                "langevin_middle" => {
                    crate::engine::md_core::Integrator::LangevinMiddle { gamma: 0.5 }
                }
                "langevin_strong" => {
                    crate::engine::md_core::Integrator::LangevinMiddle { gamma: 10.0 }
                }
                "nve" => crate::engine::md_core::Integrator::VerletVelocity { thermostat: None },
                other => {
                    return Err(format!(
                        "unknown integrator mode '{other}' (langevin_middle | langevin_strong | nve)"
                    ));
                }
            };
            e.state.cfg.integrator = integrator;
            Ok(())
        });
        match r {
            Ok(Ok(())) => 0,
            Ok(Err(msg)) => set_err(msg),
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// Arm the fail-fast trend monitor. IN JSON: {"preset":"rl_fail_fast"|"default"}
/// or knobs {"window","check_every","z_threshold","energy_floor_ps",
/// "rg_floor_ps","ss_floor_ps","skip_steps"}. Fires as `crashed=true` +
/// `trend_alarm` on the next step.
#[unsafe(no_mangle)]
pub extern "C" fn spice_set_trend(handle: i32, cfg_ptr: *const u8, cfg_len: usize) -> i32 {
    guarded(|| {
        let spec = match read_json(cfg_ptr, cfg_len) {
            Ok(v) => v,
            Err(e) => return set_err(e),
        };
        let g = |k: &str, d: f64| spec.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
        let gu = |k: &str, d: usize| {
            spec.get(k)
                .and_then(|v| v.as_u64())
                .map(|x| x as usize)
                .unwrap_or(d)
        };
        let cfg = match spec.get("preset").and_then(|v| v.as_str()) {
            Some("rl_fail_fast") => crate::engine::TrendConfig::rl_fail_fast(),
            Some("default") => crate::engine::TrendConfig {
                enabled: true,
                ..Default::default()
            },
            Some(other) => {
                return set_err(format!(
                    "unknown trend preset '{other}' (rl_fail_fast | default, or pass knobs)"
                ));
            }
            None => crate::engine::TrendConfig {
                enabled: true,
                window: gu("window", 60),
                check_every: gu("check_every", 5),
                z_threshold: g("z_threshold", 3.0),
                energy_floor_ps: g("energy_floor_ps", 50.0),
                rg_floor_ps: g("rg_floor_ps", 0.1),
                ss_floor_ps: g("ss_floor_ps", 0.01),
            },
        };
        let skip = gu("skip_steps", 0);
        let r = with_engine(handle, move |e| e.set_trend_monitor_skip(cfg, skip));
        match r {
            Ok(()) => 0,
            Err(_) => set_err("invalid handle"),
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn spice_reset_trend(handle: i32) -> i32 {
    guarded(|| match with_engine(handle, |e| e.reset_trend()) {
        Ok(()) => 0,
        Err(_) => set_err("invalid handle"),
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn spice_clear_trend(handle: i32) -> i32 {
    guarded(|| match with_engine(handle, |e| e.clear_trend_monitor()) {
        Ok(()) => 0,
        Err(_) => set_err("invalid handle"),
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn spice_has_trend(handle: i32) -> i32 {
    guarded(
        || match with_engine_ref(handle, |e| e.has_trend_monitor()) {
            Ok(has) => {
                set_out(&json!({ "has": has }));
                0
            }
            Err(_) => set_err("invalid handle"),
        },
    )
}

/// Bisect knob: skip the Langevin thermostat on rigid water.
#[unsafe(no_mangle)]
pub extern "C" fn spice_set_skip_water_thermostat(handle: i32, skip: i32) -> i32 {
    guarded(|| {
        let r = with_engine(handle, move |e| {
            e.state.cfg.overrides.skip_water_thermostat = skip != 0;
        });
        match r {
            Ok(()) => 0,
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// Harmonic distance restraint (AF3-style ligand coordination, SMD pulls).
#[unsafe(no_mangle)]
pub extern "C" fn spice_add_restraint(handle: i32, i0: u32, i1: u32, r0: f32, k: f32) -> i32 {
    guarded(|| {
        let r = with_engine(handle, move |e| {
            e.add_distance_restraint(i0 as usize, i1 as usize, r0, k)
        });
        match r {
            Ok(()) => 0,
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// Re-target restraint #idx (the ramp of a steered pull); rc 1 = applied,
/// rc 0 = index out of range.
#[unsafe(no_mangle)]
pub extern "C" fn spice_update_restraint(handle: i32, idx: u32, r0: f32, k: f32) -> i32 {
    guarded(|| {
        let r = with_engine(handle, move |e| {
            e.update_distance_restraint(idx as usize, r0, k)
        });
        match r {
            Ok(true) => 1,
            Ok(false) => 0,
            Err(_) => set_err("invalid handle"),
        }
    })
}

/// Release ALL restraints — the unbiased window of quench-and-refold.
#[unsafe(no_mangle)]
pub extern "C" fn spice_clear_restraints(handle: i32) -> i32 {
    guarded(
        || match with_engine(handle, |e| e.clear_distance_restraints()) {
            Ok(()) => 0,
            Err(_) => set_err("invalid handle"),
        },
    )
}

/// Post-build equilibration (ffi `equilibrate`): NVT strain-relief ramp +
/// hold, barostat frozen. IN JSON (all optional): {"ramp_steps":300,
/// "t_start_k":100, "k_restraint":0, "hold_steps":100,
/// "restrain_hydrogens":false, "friction_gamma":10}. Runs `total` steps
/// SYNCHRONOUSLY — at ~12 ms/step (mini, single-thread scalar wasm) a
/// 400-step default ramp blocks the tab for ~5 s. OUT: observables JSON on
/// success; {"error": ...} if the system blows up mid-ramp. Resets the
/// metrics/pseudo-label history on success (same contract as native).
#[unsafe(no_mangle)]
pub extern "C" fn spice_equilibrate(handle: i32, cfg_ptr: *const u8, cfg_len: usize) -> i32 {
    guarded(|| {
        let spec = if cfg_len == 0 {
            json!({})
        } else {
            match read_json(cfg_ptr, cfg_len) {
                Ok(v) => v,
                Err(e) => return set_err(e),
            }
        };
        let d = crate::equilibrate::EquilConfig::default();
        let g = |k: &str, dflt: f32| {
            spec.get(k)
                .and_then(|v| v.as_f64())
                .map(|x| x as f32)
                .unwrap_or(dflt)
        };
        let gu = |k: &str, dflt: usize| {
            spec.get(k)
                .and_then(|v| v.as_u64())
                .map(|x| x as usize)
                .unwrap_or(dflt)
        };
        let cfg = crate::equilibrate::EquilConfig {
            ramp_steps: gu("ramp_steps", d.ramp_steps),
            t_start_k: g("t_start_k", d.t_start_k),
            k_restraint: g("k_restraint", d.k_restraint),
            hold_steps: gu("hold_steps", d.hold_steps),
            restrain_hydrogens: spec
                .get("restrain_hydrogens")
                .and_then(|v| v.as_bool())
                .unwrap_or(d.restrain_hydrogens),
            friction_gamma: g("friction_gamma", d.friction_gamma),
        };
        let r = with_engine(handle, move |e| {
            match crate::equilibrate::equilibrate(e, &cfg) {
                Ok(()) => Ok(observables(e)),
                Err(msg) => Err(msg),
            }
        });
        match r {
            Ok(Ok(o)) => {
                set_out(&o);
                0
            }
            Ok(Err(msg)) => set_err(msg),
            Err(_) => set_err("invalid handle"),
        }
    })
}
