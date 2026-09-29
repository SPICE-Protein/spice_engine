// End-to-end test of the Web API (api.mjs) — self-contained: spins a local
// http server over web/ (the wasm build must exist in web/dist/, via
// `make web-dist` or `make web-nightly`), builds the mini fixture, steps,
// hot-changes the bath, snapshots through animate(), then disposes.
//
// Run:  node web/api.test.mjs        (from the repo root)

import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { Sim } from "./api.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const PORT = 8123;
const BASE = `http://127.0.0.1:${PORT}`;

const server = spawn("python3", ["-m", "http.server", String(PORT), "--directory", here], {
  stdio: "ignore",
});
const killServer = () => { try { server.kill(); } catch {} };
process.on("exit", killServer);

async function waitUp() {
  for (let i = 0; i < 50; i++) {
    try { const r = await fetch(`${BASE}/mini.cif`); if (r.ok) return; } catch {}
    await new Promise((res) => setTimeout(res, 100));
  }
  throw new Error("http server did not come up");
}

const assert = (cond, msg) => { if (!cond) { killServer(); throw new Error(`FAIL: ${msg}`); } };

await waitUp();

const cif = await (await fetch(`${BASE}/mini.cif`)).text();

const sim = await Sim.load(cif, {
  wasm: `${BASE}/dist/spice_engine.wasm.gz`,
  env: { tempK: 300, soluteK: 300, ionicStrengthM: 0.15, boxPadA: 6, relaxIters: 200, strictIncomplete: false },
});
assert(sim.info && sim.info.nAtoms > 20, `info.nAtoms: ${JSON.stringify(sim.info)}`);
assert(sim.info.nSites > sim.info.nAtoms, "waters present");

const m = sim.step(10);
assert(Number.isFinite(m.u_total_kcal) && !m.crashed, `step metrics: ${JSON.stringify(m)}`);

sim.set({ tempK: 320, soluteK: 320, timestepPs: 0.002 });
const snap = sim.snapshot({ observables: true });
assert(snap.coords.length === snap.nSites * 3, "coords length");
assert(snap.roles.length === snap.nSites, "roles length");
assert(snap.observables && typeof snap.observables.water_t_k === "number", "observables");
assert(new Set(snap.roles).size >= 2, "role mix (solute+water)");

let frames = 0;
let lastCoords = null;
const anim = sim.animate((s) => {
  frames++;
  const sn = s.snapshot();
  if (lastCoords) {
    let moved = 0;
    for (let i = 0; i < sn.coords.length; i += 9) if (sn.coords[i] !== lastCoords[i]) moved++;
    if (moved > 0 && frames >= 5) anim.stop();
  }
  lastCoords = sn.coords;
}, { stepsPerFrame: 3, fps: 60 });
await new Promise((res) => setTimeout(res, 3000));
anim.stop();
assert(frames >= 5, `animate frames: ${frames}`);

const forces = new Float32Array(sim.info.nAtoms * 3).fill(0.05);
const ma = sim.stepAction(forces);
assert(Number.isFinite(ma.u_total_kcal), "stepAction finite");

sim.dispose();
assert(sim.info === null, "disposed");
console.log(`frames=${frames} sites=${snap.nSites} build=${sim.spice.version?.() ?? "ok"}`);
console.log("API_TEST_PASSED");
killServer();
process.exit(0);
