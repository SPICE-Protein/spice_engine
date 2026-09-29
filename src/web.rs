//! Browser boundary for `spice_engine` (v1.3.9 web port).
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

/// [solute_T, water_T] — the same DOF/KE arithmetic as `ffi.rs::species_temperatures`
/// (kept in lockstep deliberately; single-bath solute lag is the whole point of the
/// dual-bath observability).
fn species_temps(e: &SpiceEngine) -> [f64; 2] {
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

    let solute_dof = (3 * n_solute - n_solute_h) as f64;
    let water_dof = (6 * s.water.len()) as f64;
    let st = if solute_dof > 0.0 {
        2.0 * solute_ke / (solute_dof * R_KCAL)
    } else {
        0.0
    };
    let wt = if water_dof > 0.0 {
        2.0 * water_ke / (water_dof * R_KCAL)
    } else {
        0.0
    };
    [st, wt]
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
    0
}
