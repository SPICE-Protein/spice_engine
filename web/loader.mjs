// spice_engine browser (WebAssembly) loader — a thin, dependency-free wrapper around the raw
// extern-"C" ABI exported by the `spice_engine` wasm build (v1.3.9 web port; v1.3.10 adds
// the FFI-parity analysis/control probes — prefer api.mjs for application code).
//
// The wasm module is NOT wasm-bindgen-processed (there is no CLI offline), so
// its surface is integers and pointers: handles into a Rust-side engine
// registry, `ptr,len` UTF-8 buffers in, and a module OUT scratch that carries
// JSON out. This file hides all of that behind an ordinary JS class.
//
//   import { loadSpice } from "./loader.mjs";
//   const spice = await loadSpice({ url: "dist/spice_engine.wasm.gz" });
//   const eng   = spice.build(mmCifText, { temp_k: 300, pressure_bar: 0, box_pad_a: 6 });
//   eng.setTemperature(300, { soluteK: 300 });   // dual bath for any thermostatted protocol
//   for (let i = 0; i < 500; i++) eng.step();
//   const { coords, roles, nSites } = eng.positions();   // re-slice every frame
//   eng.free();
//
// Browser baseline: needs wasm + (optional) wasm-simd128. simd128 is
// Chrome 101+, Firefox 100+, Safari 16.0+; the scalar build runs everywhere.
// Serve over http(s) (fetch of the .wasm/.gz), not file://.

const enc = new TextEncoder();
const dec = new TextDecoder();

async function fetchWasmBytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: HTTP ${res.status}`);
  let bytes = new Uint8Array(await res.arrayBuffer());
  if (url.endsWith(".gz")) {
    if (typeof DecompressionStream !== "undefined") {
      const ds = new DecompressionStream("gzip");
      const buf = await new Response(new Blob([bytes]).stream().pipeThrough(ds)).arrayBuffer();
      bytes = new Uint8Array(buf);
    } else {
      throw new Error("gzip wasm needs DecompressionStream (modern browsers)");
    }
  }
  return bytes;
}

class Spice {
  /** Raw wasm exports + a live `memory` handle. */
  constructor(exports) {
    this.exports = exports;
  }

  get memory() {
    return this.exports.memory;
  }

  // -- memory helpers --------------------------------------------------------
  // Allocate a Rust-side buffer of `bytes.length`, fill it, return its address.
  _put(bytes) {
    const ptr = this.exports.spice_alloc(bytes.length);
    new Uint8Array(this.memory.buffer, ptr, bytes.length).set(bytes);
    return ptr;
  }
  _drop(ptr, len) {
    if (ptr) this.exports.spice_free(ptr, len);
  }

  // JSON-returning calls leave bytes in OUT; read them before the next call.
  _outJson() {
    const ptr = this.exports.spice_out_ptr();
    const len = this.exports.spice_out_len();
    const text = dec.decode(new Uint8Array(this.memory.buffer, ptr, len));
    if (len === 0) return null;
    return JSON.parse(text);
  }
  _outText() {
    const ptr = this.exports.spice_out_ptr();
    const len = this.exports.spice_out_len();
    return dec.decode(new Uint8Array(this.memory.buffer, ptr, len));
  }

  version() {
    this.exports.spice_version();
    return this._outText(); // raw semver string, not JSON
  }

  lastError() {
    this.exports.spice_last_error();
    return this._outJson();
  }

  /** Build a system from mmCIF text + a params object. Returns a handle. */
  _buildFrom(cifText, params) {
    const cifB = enc.encode(cifText);
    const parB = enc.encode(JSON.stringify(params ?? {}));
    const cifP = this._put(cifB);
    const parP = this._put(parB);
    let handle, info;
    try {
      handle = this.exports.spice_build_mmcif(cifP, cifB.length, parP, parB.length);
      info = this._outJson();
    } finally {
      this._drop(cifP, cifB.length);
      this._drop(parP, parB.length);
    }
    if (handle <= 0) {
      throw new Error(`build failed: ${info && info.error ? info.error : `code ${handle}`}`);
    }
    return new SpiceEngine(this, handle, info);
  }

  build(mmCifText, params) {
    return this._buildFrom(mmCifText, params);
  }
}

// Build params the native `Engine.build` mirrors. All optional; documented keys:
//   ph, temp_k, pressure_bar, ionic_strength_m, relax_iters, tolerance,
//   strict_incomplete, box_pad_a, mg_cl2_m, ca_cl2_m, sr_cl2_m, ba_cl2_m,
//   redox_reducing, cosolvents_json, salts_json.

class SpiceEngine {
  constructor(spice, handle, info) {
    this.spice = spice;
    this.handle = handle;
    this.info = info; // { n_atoms, n_water, n_sites, n_residues, net_charge_e, ... }
  }

  _json(rc) {
    const obj = this.spice._outJson();
    if (rc < 0) {
      throw new Error(`engine call failed (rc ${rc}): ${obj && obj.error ? obj.error : "?"}`);
    }
    return obj;
  }

  /** Advance one step. Returns the metrics JSON (crashed ⇒ throw). */
  step() {
    const rc = this.spice.exports.spice_step(this.handle);
    const m = this._json(rc);
    if (m.crashed) {
      const e = new Error(`MD crashed: ${m.crash_reason ?? "unknown"}`);
      e.metrics = m;
      throw e;
    }
    return m;
  }

  /** Step with a per-solute-atom bias force (Float32Array of n_atoms*3). */
  stepAction(forces) {
    const need = this.info.n_atoms * 3;
    if (forces.length !== need) throw new Error(`forces length ${forces.length} != ${need}`);
    const bytes = new Uint8Array(forces.buffer, forces.byteOffset, forces.byteLength);
    const ptr = this.spice._put(bytes);
    let rc, m;
    try {
      rc = this.spice.exports.spice_step_action(this.handle, ptr, forces.length);
      m = this._json(rc);
    } finally {
      this.spice._drop(ptr, bytes.length);
    }
    if (m && m.crashed) {
      const e = new Error(`MD crashed: ${m.crash_reason ?? "unknown"}`);
      e.metrics = m;
      throw e;
    }
    return m;
  }

  observables() {
    const rc = this.spice.exports.spice_observables(this.handle);
    return this._json(rc);
  }

  /**
   * Snapshot of every site: solute atoms (role 0 heavy / 1 H) then per water
   * O (2), H0 (3), H1 (3). Returns fresh typed-array COPIES each call — wasm
   * memory can grow and detach an old view, so never cache across a step.
   */
  positions() {
    const nSites = this.spice.exports.spice_positions(this.handle);
    if (nSites <= 0) throw new Error(`spice_positions failed (${nSites})`);
    const pPtr = this.spice.exports.spice_positions_ptr();
    const pLen = this.spice.exports.spice_positions_len();
    const rPtr = this.spice.exports.spice_roles_ptr();
    const rLen = this.spice.exports.spice_roles_len();
    // slice() copies out of the live buffer (survives later memory.grow).
    const coords = new Float32Array(this.spice.memory.buffer.slice(pPtr, pPtr + pLen * 4));
    const roles = new Int32Array(this.spice.memory.buffer.slice(rPtr, rPtr + rLen * 4));
    return { coords, roles, nSites: nSites | 0 };
  }

  /**
   * Thermostat. `k` is the solvent/bath setpoint; `soluteK` (if given) enables
   * the dual bath — the solute (protein + ions) then gets its own Langevin
   * setpoint. Always pass soluteK = T for thermostatted protocols (annealing,
   * relaxation, pulled runs): water owns ~96% of the heat capacity, so a single
   * bath leaves the solute at ~0.72-0.78× setpoint on short windows. Omit
   * soluteK for a single bath.
   */
  setTemperature(k, { soluteK = -1 } = {}) {
    const rc = this.spice.exports.spice_set_temperature(this.handle, k, soluteK);
    if (rc < 0) throw new Error("set_temperature failed");
  }
  setGamma(gamma) {
    if (this.spice.exports.spice_set_gamma(this.handle, gamma) < 0) throw new Error("set_gamma failed");
  }
  setTimestep(dtPs) {
    if (this.spice.exports.spice_set_timestep(this.handle, dtPs) < 0) throw new Error("set_timestep failed");
  }
  /** pBar <= 0 ⇒ barostat off (pure NVT). */
  setPressure(pBar) {
    if (this.spice.exports.spice_set_pressure(this.handle, pBar) < 0) throw new Error("set_pressure failed");
  }
  /** Toy mode: individually disable force terms (1 = off). */
  setForceOverrides({ bonded = 0, coulomb = 0, lj = 0, longRange = 0 } = {}) {
    const rc = this.spice.exports.spice_set_force_overrides(
      this.handle, bonded, coulomb, lj, longRange,
    );
    if (rc < 0) throw new Error("set_force_overrides failed");
  }
  resetVelocities() {
    if (this.spice.exports.spice_reset_velocities(this.handle) < 0) throw new Error("reset_velocities failed");
  }

  // ---- v1.3.10 analysis + control parity (mirrors the Python FFI) ----------
  // JSON-in / JSON-out calls share `this._json(rc)`; numeric probes return a
  // fresh Float64Array COPIED out of the RES scratch (same view-detach rule as
  // positions()). Points/forces are passed as flat Float64Array [x,y,z,...].

  _call(name) {
    const rc = this.spice.exports[name](this.handle);
    return this._json(rc);
  }
  _res() {
    const ptr = this.spice.exports.spice_res_ptr();
    const len = this.spice.exports.spice_res_len();
    return new Float64Array(this.spice.memory.buffer.slice(ptr, ptr + len * 8));
  }
  _flat(points) {
    // Accept flat typed array | plain array | array of triples -> flat Float64Array.
    if (points instanceof Float64Array) return points;
    if (Array.isArray(points) && points.length && Array.isArray(points[0])) {
      const out = new Float64Array(points.length * 3);
      points.forEach((p, i) => { out[3 * i] = p[0]; out[3 * i + 1] = p[1]; out[3 * i + 2] = p[2]; });
      return out;
    }
    return Float64Array.from(points);
  }
  _probe(name, points) {
    const f = this._flat(points);
    const bytes = new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
    const ptr = this.spice._put(bytes);
    let rc;
    try {
      rc = this.spice.exports[name](this.handle, ptr, f.length);
    } finally {
      this.spice._drop(ptr, bytes.length);
    }
    if (rc < 0) {
      const e = this.spice._outJson();
      throw new Error(`${name} failed: ${e && e.error ? e.error : rc}`);
    }
    return this._res();
  }
  _jsonArg(name, spec) {
    const bytes = enc.encode(JSON.stringify(spec ?? {}));
    const ptr = this.spice._put(bytes);
    let rc, out;
    try {
      rc = this.spice.exports[name](this.handle, ptr, bytes.length);
      out = this._json(rc);
    } finally {
      this.spice._drop(ptr, bytes.length);
    }
    return out;
  }

  /** Electrostatic potential (kcal/mol/e) at probe points via the engine PME.
   *  Absolute values are gauge-arbitrary — use DIFFERENCES phi(p)-phi(ref). */
  esp(points) { return this._probe("spice_esp", points); }

  /** Electrostatic field E = -grad phi (3 doubles per point), one analytic PME
   *  pass. `positions`: optional PME-order snapshot (see pmePositions()). */
  efield(points, positions = null) {
    const f = this._flat(points);
    const pf = positions ? this._flat(positions) : new Float64Array(0);
    const b1 = new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
    const b2 = pf.length ? new Uint8Array(pf.buffer, pf.byteOffset, pf.byteLength) : null;
    const p1 = this.spice._put(b1);
    const p2 = b2 ? this.spice._put(b2) : 0;
    let rc;
    try {
      rc = this.spice.exports.spice_efield(this.handle, p1, f.length, p2, pf.length);
    } finally {
      this.spice._drop(p1, b1.length);
      if (b2) this.spice._drop(p2, b2.length);
    }
    if (rc < 0) {
      const e = this.spice._outJson();
      throw new Error(`spice_efield failed: ${e && e.error ? e.error : rc}`);
    }
    return this._res();
  }
  pmePositions() { return this._probeRes("spice_pme_positions"); }
  _probeRes(name) {
    const rc = this.spice.exports[name](this.handle);
    if (rc < 0) {
      const e = this.spice._outJson();
      throw new Error(`${name} failed: ${e && e.error ? e.error : rc}`);
    }
    return this._res();
  }

  /** Per-atom SASA (A^2, Shrake-Rupley) for solute/ion atoms; water is not an
   *  occluder. One value per state.atoms entry (== info.n_atoms). */
  atomSasa({ probeRadius = 1.4, nSphere = 200 } = {}) {
    const rc = this.spice.exports.spice_atom_sasa(this.handle, probeRadius, nSphere);
    if (rc < 0) {
      const e = this.spice._outJson();
      throw new Error(`spice_atom_sasa failed: ${e && e.error ? e.error : rc}`);
    }
    return this._res();
  }
  coordsCa() { return this._probeRes("spice_coords_ca"); }
  pseudoLabels() { return this._probeRes("spice_pseudo_labels"); }
  perResidueMaxForce() { return this._probeRes("spice_per_residue_max_force"); }
  resetPseudoLabels() {
    if (this.spice.exports.spice_reset_pseudo_labels(this.handle) < 0) throw new Error("reset_pseudo_labels failed");
  }

  metrics() { return this._call("spice_metrics"); }          // five metrics + rg/rmsf/margin
  energyTerms() { return this._call("spice_energy_terms"); } // {total,nonbonded,bonded}
  speciesTemperatures() { return this._call("spice_species_temperatures"); }
  thermoInfo() { return this._call("spice_thermo_info"); }
  waterRigidSplit() { return this._call("spice_water_rigid_split"); }
  envInfo() { return this._call("spice_env_info"); }
  exclusionDiagnostics() { return this._call("spice_exclusion_diagnostics"); }
  debugStateDump() { return this._call("spice_debug_state_dump"); } // LARGE json — don't poll per frame
  computationTime() { return this._call("spice_computation_time"); }

  atomNames() { return this._call("spice_atom_names"); }   // true PDB names, state.atoms order
  atomLabels() { return this._call("spice_atom_labels"); } // {element,residue,seq_id,serial}[]
  sequence() {
    const rc = this.spice.exports.spice_sequence(this.handle);
    if (rc < 0) throw new Error("spice_sequence failed");
    return this.spice._outText();
  }

  /** {resSeq:[...], names:[...]?, sidechainHeavy?:bool} -> state.atoms indices. */
  selectAtoms(spec) {
    const wire = {
      res_seq: spec.resSeq ?? spec.res_seq ?? [],
      names: spec.names ?? null,
      sidechain_heavy: spec.sidechainHeavy ?? spec.sidechain_heavy ?? false,
    };
    return this._jsonArg("spice_select_atoms", wire);
  }
  contacts(a, b, cutoff = 4.0) {
    const r = this._jsonArg("spice_contact_count", { a, b, cutoff });
    return r.count;
  }
  /** {path:[[x,y,z],...], spacing?, exclude?, includeWater?} -> {profile, bottleneck}. */
  bottleneck(spec) {
    const wire = {
      path: spec.path,
      spacing: spec.spacing ?? 0.5,
      exclude: spec.exclude ?? [],
      include_water: spec.includeWater ?? spec.include_water ?? false,
    };
    return this._jsonArg("spice_bottleneck", wire);
  }
  clashReport(minForce = 50) {
    const rc = this.spice.exports.spice_clash_report(this.handle, minForce);
    return this._json(rc);
  }
  forceReport(minForce = 50) {
    const rc = this.spice.exports.spice_force_report(this.handle, minForce);
    return this._json(rc);
  }
  /** Audit probe: dilate by lam, return {u_kcal, virial_kcal, pressure_bar}.
   *  MUTATES forces/PE — re-step (or probe with 1.0) before continuing a run. */
  rigidScaleProbe(lam) {
    const rc = this.spice.exports.spice_rigid_scale_probe(this.handle, lam);
    return this._json(rc);
  }

  /** mode: "langevin_middle" | "langevin_strong" | "nve". */
  setIntegrator(mode) {
    const bytes = enc.encode(mode);
    const ptr = this.spice._put(bytes);
    let rc;
    try {
      rc = this.spice.exports.spice_set_integrator(this.handle, ptr, bytes.length);
    } finally {
      this.spice._drop(ptr, bytes.length);
    }
    if (rc < 0) {
      const e = this.spice._outJson();
      throw new Error(`spice_set_integrator failed: ${e && e.error ? e.error : rc}`);
    }
  }
  /** Arm fail-fast trend monitor: {preset:"rl_fail_fast"|"default"} or knobs. */
  setTrend(cfg) {
    const bytes = enc.encode(JSON.stringify(cfg ?? {}));
    const ptr = this.spice._put(bytes);
    let rc;
    try {
      rc = this.spice.exports.spice_set_trend(this.handle, ptr, bytes.length);
    } finally {
      this.spice._drop(ptr, bytes.length);
    }
    if (rc < 0) return this._json(rc); // throws with the error message
  }
  resetTrend() {
    if (this.spice.exports.spice_reset_trend(this.handle) < 0) throw new Error("reset_trend failed");
  }
  clearTrend() {
    if (this.spice.exports.spice_clear_trend(this.handle) < 0) throw new Error("clear_trend failed");
  }
  hasTrend() { return this._call("spice_has_trend").has; }
  setSkipWaterThermostat(skip) {
    if (this.spice.exports.spice_set_skip_water_thermostat(this.handle, skip ? 1 : 0) < 0) {
      throw new Error("set_skip_water_thermostat failed");
    }
  }

  addRestraint(i0, i1, r0, k) {
    if (this.spice.exports.spice_add_restraint(this.handle, i0, i1, r0, k) < 0) {
      throw new Error("add_restraint failed");
    }
  }
  /** true if restraint idx existed (SMD ramp retarget). */
  updateRestraint(idx, r0, k) {
    return this.spice.exports.spice_update_restraint(this.handle, idx, r0, k) === 1;
  }
  clearRestraints() {
    if (this.spice.exports.spice_clear_restraints(this.handle) < 0) throw new Error("clear_restraints failed");
  }

  /** Equilibration ramp+hold, synchronous: {rampSteps, tStartK, kRestraint,
   *  holdSteps, restrainHydrogens, frictionGamma} (defaults match Python).
   *  Returns observables JSON; throws on mid-ramp blow-up. */
  equilibrate(cfg = {}) {
    const wire = {
      ramp_steps: cfg.rampSteps, t_start_k: cfg.tStartK, k_restraint: cfg.kRestraint,
      hold_steps: cfg.holdSteps, restrain_hydrogens: cfg.restrainHydrogens,
      friction_gamma: cfg.frictionGamma,
    };
    for (const k of Object.keys(wire)) if (wire[k] === undefined) delete wire[k];
    const out = this._jsonArg("spice_equilibrate", wire);
    if (out && out.error) throw new Error(`equilibrate failed: ${out.error}`);
    return out;
  }

  free() {
    this.spice.exports.spice_free_engine(this.handle);
    this.handle = 0;
  }
}

/**
 * Instantiate the wasm and return a `Spice` runtime.
 * @param {{url: string}} opts - location of the .wasm or .wasm.gz artifact.
 * @param {{log?: (s:string)=>void}} hooks - optional console sink for env.log.
 */
export async function loadSpice({ url }, hooks = {}) {
  const bytes = await fetchWasmBytes(url);
  let instance;
  const env = {
    now_ms: () => (typeof performance !== "undefined" ? performance.now() : Date.now()),
    log: (ptr, len) => {
      const s = dec.decode(new Uint8Array(instance.exports.memory.buffer, ptr, len));
      (hooks.log ?? ((m) => console.log("[spice]", m)))(s);
    },
  };
  const module = await WebAssembly.compile(bytes);
  instance = await WebAssembly.instantiate(module, { env });
  const spice = new Spice(instance.exports);
  instance.exports.spice_init();
  return spice;
}
