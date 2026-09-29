// spice_engine Web API — the ergonomic facade front-ends should code against.
//
// loader.mjs wraps the raw extern-"C" ABI (handles, pointers, OUT buffers).
// This file wraps *that* into a normal-looking JS object: camelCase build
// params, chainable control, a render-loop helper, and plain snapshots for a
// canvas. Zero dependencies; runs in browsers and node ≥ 18 (fetch,
// DecompressionStream). TypeScript users: see api.d.ts.
//
//   import { Sim } from "./api.mjs";
//
//   const sim = await Sim.load(await (await fetch("mini.cif")).text(), {
//     env: { tempK: 300, soluteK: 300, ionicStrengthM: 0.15, boxPadA: 6 },
//     wasm: "dist/spice_engine.wasm.gz",   // or the nightly.link .zip staged locally
//   });
//   console.log(sim.info);                  // { nAtoms, nSites, nWater, netChargeE, ... }
//
//   const stop = sim.animate((s, m) => {    // requestAnimationFrame (fallback: setInterval)
//     const { coords, roles } = s.snapshot();
//     drawPoints(coords, roles);            // your renderer
//     label.textContent = `${m.u_total_kcal.toFixed(1)} kcal/mol · ${m.time_ps.toFixed(2)} ps`;
//   }, { stepsPerFrame: 2 });
//   // stop(); sim.dispose();
//
// Two bath rule from the engine's protocol docs applies here: pass soluteK with
// tempK for any thermostatted protocol — a single bath leaves the solute at
// ~0.72-0.78x setpoint because water owns the heat capacity.

import { loadSpice } from "./loader.mjs";

// camelCase env keys -> wasm build params (mirrors native Engine.build).
const ENV_KEYS = {
  ph: "ph",
  tempK: "temp_k",
  pressureBar: "pressure_bar",
  ionicStrengthM: "ionic_strength_m",
  relaxIters: "relax_iters",
  tolerance: "tolerance",
  strictIncomplete: "strict_incomplete",
  boxPadA: "box_pad_a",
  mgCl2M: "mg_cl2_m",
  caCl2M: "ca_cl2_m",
  srCl2M: "sr_cl2_m",
  baCl2M: "ba_cl2_m",
  redoxReducing: "redox_reducing",
  cosolventsJson: "cosolvents_json",
  saltsJson: "salts_json",
  efieldJson: "efield_json",   // [Ex, Ey, Ez] kcal/(mol·e·A)
  efieldOmega: "efield_omega", // rad/ps, 0 = static
};

function toBuildParams(env = {}) {
  const out = {};
  for (const [camel, snake] of Object.entries(ENV_KEYS)) {
    if (env[camel] !== undefined) out[snake] = env[camel];
  }
  // soluteK is runtime-only (dual bath), never a build param; ignore it here.
  return out;
}

export class Sim {
  /** Prefer Sim.load() / createSim(); the constructor is for inner use. */
  constructor(spice) {
    this.spice = spice;
    this.engine = null;
    this.info = null;
    this.last = null;
    this._timer = null;
  }

  /** Load only the runtime (no system yet). */
  static async create({ wasm = "dist/spice_engine.wasm.gz", log } = {}) {
    const spice = await loadSpice({ url: wasm }, { log: log ?? ((m) => console.log(m)) });
    return new Sim(spice);
  }

  /** One-shot: load runtime + build a system. cifText is mmCIF content. */
  static async load(cifText, { env = {}, wasm, log } = {}) {
    const sim = await Sim.create({ wasm, log });
    sim.build(cifText, env);
    return sim;
  }

  /** Build (or rebuild) the system from mmCIF text + camelCase env. */
  build(cifText, env = {}) {
    this.dispose();
    this.engine = this.spice.build(cifText, toBuildParams(env));
    const i = this.engine.info;
    this.info = {
      handle: this.engine.handle,
      nAtoms: i.n_atoms,
      nResidues: i.n_residues,
      nWater: i.n_water,
      nSites: i.n_sites,
      netChargeE: i.net_charge_e,
      effectiveIonicM: i.effective_ionic_m,
    };
    // Dual bath at build time if both setpoints given.
    if (env.tempK !== undefined && env.soluteK !== undefined) {
      this.engine.setTemperature(env.tempK, { soluteK: env.soluteK });
    }
    return this;
  }

  /** n steps (default 1); returns the last metrics dict. Throws on crash. */
  step(n = 1) {
    this._need();
    for (let i = 0; i < n; i++) this.last = this.engine.step();
    return this.last;
  }

  /** One step with a Float32Array bias field of length nAtoms*3. */
  stepAction(forces) {
    this._need();
    this.last = this.engine.stepAction(forces);
    return this.last;
  }

  /** Hot control surface; any subset. soluteK re-arms the dual bath,
   *  soluteK: null drops back to a single bath. */
  set({ tempK, soluteK, gamma, timestepPs, pressureBar } = {}) {
    this._need();
    if (tempK !== undefined) {
      this.engine.setTemperature(tempK, { soluteK: soluteK === undefined ? -1 : (soluteK ?? -1) });
    } else if (soluteK !== undefined) {
      // keep the current water setpoint, change only the solute: re-read from observables.
      const obs = this.engine.observables();
      const waterK = obs.water_t_k ?? 300;
      this.engine.setTemperature(waterK, { soluteK: soluteK ?? -1 });
    }
    if (gamma !== undefined) this.engine.setGamma(gamma);
    if (timestepPs !== undefined) this.engine.setTimestep(timestepPs);
    if (pressureBar !== undefined) this.engine.setPressure(pressureBar); // <=0 => NVT
    return this;
  }

  /** Toy-force switches: individually disable terms (true = off). */
  setForcesOff({ bonded = false, coulomb = false, lj = false, longRange = false } = {}) {
    this._need();
    this.engine.setForceOverrides({
      bonded: bonded ? 1 : 0,
      coulomb: coulomb ? 1 : 0,
      lj: lj ? 1 : 0,
      longRange: longRange ? 1 : 0,
    });
    return this;
  }

  resetVelocities() {
    this._need();
    this.engine.resetVelocities();
    return this;
  }

  /** Fresh copy for rendering: Float32Array coords (nSites*3, Å) + Int32Array
   *  roles (0 solute heavy, 1 solute H, 2 water O, 3 water H). Views are
   *  copies on purpose — wasm memory.grow detaches live views. */
  snapshot({ observables = false } = {}) {
    this._need();
    const { coords, roles, nSites } = this.engine.positions();
    const snap = { coords, roles, nSites, metrics: this.last };
    if (observables) snap.observables = this.engine.observables();
    return snap;
  }

  /** Full engine observables (energy terms, temperatures, pressure, Rg, ...). */
  observables() {
    this._need();
    return this.engine.observables();
  }

  // ---- v1.3.10 analysis + control surface (Python-FFI parity) --------------
  // Everything below mirrors src/ffi.rs read-only diagnostics and runtime
  // controls; numeric probes return Float64Array copies, JSON probes return
  // parsed objects with the SAME key spellings as the Python engine.

  /** True PDB atom names in solute-atom order (ions report their species label). */
  atomNames() { this._need(); return this.engine.atomNames(); }
  atomLabels() { this._need(); return this.engine.atomLabels(); }
  sequence() { this._need(); return this.engine.sequence(); }

  /** Residue selection -> solute-atom indices. `spec`: {resSeq, names?, sidechainHeavy?}. */
  select(spec) { this._need(); return this.engine.selectAtoms(spec); }
  /** Pairs of index sets closer than `cutoff` Å (min image). */
  contacts(a, b, cutoff = 4.0) { this._need(); return this.engine.contacts(a, b, cutoff); }
  /** Channel clearance: {path:[[x,y,z],...], spacing?, exclude?, includeWater?} -> {profile, bottleneck}. */
  bottleneck(spec) { this._need(); return this.engine.bottleneck(spec); }

  /** Per-atom SASA Å² (Shrake-Rupley; solute+ions only, water is the probe model). */
  sasa({ probeRadius = 1.4, nSphere = 200 } = {}) {
    this._need();
    return this.engine.atomSasa({ probeRadius, nSphere });
  }
  /** Electrostatic potential (kcal/mol/e) at points ([x,y,z] triples or flat
   *  Float64Array). Gauge-arbitrary absolute values — report DIFFERENCES. */
  esp(points) { this._need(); return this.engine.esp(points); }
  /** Electrostatic field E = -grad phi (3 per point), one analytic PME pass.
   *  {positions} (in `pmePositions()` order) evaluates a time-averaged frame. */
  field(points, { positions = null } = {}) {
    this._need();
    return this.engine.efield(points, positions);
  }
  pmePositions() { this._need(); return this.engine.pmePositions(); }
  coordsCa() { this._need(); return this.engine.coordsCa(); }
  pseudoLabels() { this._need(); return this.engine.pseudoLabels(); }
  resetPseudoLabels() { this._need(); this.engine.resetPseudoLabels(); return this; }
  perResidueMaxForce() { this._need(); return this.engine.perResidueMaxForce(); }

  /** The five physical metrics + rg/rmsf/stability_margin (reference snapshot
   *  taken at build time, exactly like Python `Engine.metrics()`). */
  metrics() { this._need(); return this.engine.metrics(); }
  energyTerms() { this._need(); return this.engine.energyTerms(); }
  speciesTemperatures() { this._need(); return this.engine.speciesTemperatures(); }
  thermoInfo() { this._need(); return this.engine.thermoInfo(); }
  waterRigidSplit() { this._need(); return this.engine.waterRigidSplit(); }
  envInfo() { this._need(); return this.engine.envInfo(); }
  exclusionDiagnostics() { this._need(); return this.engine.exclusionDiagnostics(); }
  /** LARGE audit dump (sites/forces/exclusions/virial buckets) — analysis, not per-frame. */
  debugStateDump() { this._need(); return this.engine.debugStateDump(); }
  computationTime() { this._need(); return this.engine.computationTime(); }
  clashReport(minForce = 50) { this._need(); return this.engine.clashReport(minForce); }
  forceReport(minForce = 50) { this._need(); return this.engine.forceReport(minForce); }
  /** Audit probe (MUTATES forces/PE): {u_kcal, virial_kcal, pressure_bar} at lam-dilated state. */
  rigidScaleProbe(lam) { this._need(); return this.engine.rigidScaleProbe(lam); }

  /** "langevin_middle" | "langevin_strong" | "nve" (energy-conservation probe). */
  setIntegrator(mode) { this._need(); this.engine.setIntegrator(mode); return this; }
  /** Arm the fail-fast trend monitor: "rl_fail_fast" | "default" | knob object. */
  setTrend(cfg) {
    this._need();
    this.engine.setTrend(typeof cfg === "string" ? { preset: cfg } : cfg);
    return this;
  }
  resetTrend() { this._need(); this.engine.resetTrend(); return this; }
  clearTrend() { this._need(); this.engine.clearTrend(); return this; }
  hasTrend() { this._need(); return this.engine.hasTrend(); }
  setSkipWaterThermostat(on) { this._need(); this.engine.setSkipWaterThermostat(on); return this; }

  /** Harmonic distance restraint (ligand coordination / SMD pulls). */
  addRestraint(i0, i1, r0, k) { this._need(); this.engine.addRestraint(i0, i1, r0, k); return this; }
  updateRestraint(idx, r0, k) { this._need(); return this.engine.updateRestraint(idx, r0, k); }
  clearRestraints() { this._need(); this.engine.clearRestraints(); return this; }

  /** NVT strain-relief ramp + hold (Synchronous — blocks the tab ~steps*ms).
   *  Defaults match Python; resets the metrics/pseudo-label history on success. */
  equilibrate(opts = {}) { this._need(); return this.engine.equilibrate(opts); }

  /**
   * Drive the loop for a renderer: steps then calls onFrame(this, metrics).
   * requestAnimationFrame when present, setInterval(fps) otherwise.
   * Returns { stop() }. onFrame errors stop the loop (rethrow included).
   */
  animate(onFrame, { stepsPerFrame = 1, fps = 30 } = {}) {
    this._need();
    let stopped = false;
    const tick = () => {
      if (stopped) return;
      try {
        const m = this.step(stepsPerFrame);
        onFrame(this, m);
      } catch (err) {
        stop();
        throw err;
      }
    };
    let handle;
    if (typeof requestAnimationFrame === "function") {
      const loop = () => { if (stopped) return; tick(); handle = requestAnimationFrame(loop); };
      handle = requestAnimationFrame(loop);
    } else {
      handle = setInterval(tick, Math.max(1, Math.round(1000 / fps)));
    }
    const stop = () => {
      stopped = true;
      if (typeof cancelAnimationFrame === "function") cancelAnimationFrame(handle);
      else clearInterval(handle);
    };
    return { stop };
  }

  dispose() {
    if (this.engine) {
      try { this.engine.free(); } catch { /* already gone */ }
      this.engine = null;
      this.info = null;
      this.last = null;
    }
  }

  _need() {
    if (!this.engine) throw new Error("no system built yet — Sim.load(cif, {env}) or sim.build()");
  }
}

export const createSim = (opts) => Sim.create(opts);
