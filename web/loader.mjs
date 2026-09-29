// spice_engine browser (WebAssembly) loader — a thin, dependency-free wrapper around the raw
// extern-"C" ABI exported by the `spice_engine` wasm build (v1.3.9 web port).
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
