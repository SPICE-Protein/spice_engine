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
