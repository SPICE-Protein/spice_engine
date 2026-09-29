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

// ---- v1.3.10 FFI-parity probes ----------------------------------------------
const names = sim.atomNames();
assert(names.length === sim.info.nAtoms, `names ${names.length} vs ${sim.info.nAtoms}`);
assert(names.includes("CA"), "CA present in names");
assert(sim.atomLabels().length === sim.info.nAtoms, "labels length");
assert(typeof sim.sequence() === "string" && sim.sequence().length > 3, "sequence");

const sel = sim.select({ resSeq: [2, 3], sidechainHeavy: true });
assert(Array.isArray(sel) && sel.length > 0, `select: ${JSON.stringify(sel)}`);
assert(sel.every((i) => i < sim.info.nAtoms), "select in range");
const c = sim.contacts(sel, sel, 5.0);
assert(Number.isInteger(c) && c > 0, `contacts ${c}`);

const sasa = sim.sasa();
assert(sasa.length === sim.info.nAtoms, "sasa length");
let sasaSum = 0;
for (const v of sasa) { assert(Number.isFinite(v) && v >= 0, "sasa values"); sasaSum += v; }
assert(sasaSum > 0, "sasa positive");

const c0 = [snap.coords[0], snap.coords[1], snap.coords[2]];
const p1 = [c0[0] + 3, c0[1], c0[2]];
const phi = sim.esp([c0, p1]);
assert(phi.length === 2 && Number.isFinite(phi[0]) && Number.isFinite(phi[1]), `esp ${phi}`);
const efield = sim.field([c0]);
assert(efield.length === 3 && efield.every(Number.isFinite), "field");
const pme = sim.pmePositions();
assert(pme.length === 3 * sim.info.nSites, `pme sites ${pme.length / 3}`); // solute + M/H0/H1 per water (O is uncharged, excluded)
const efieldAvg = sim.field([c0], { positions: pme });
assert(efieldAvg.length === 3 && efieldAvg.every(Number.isFinite), "field on pme positions");

const bn = sim.bottleneck({ path: [c0, p1], spacing: 0.5 });
assert(bn.profile.length >= 2 && Number.isFinite(bn.bottleneck), "bottleneck");

const ca = sim.coordsCa();
const pl = sim.pseudoLabels();
assert(ca.length === 3 * sim.info.nResidues, `ca ${ca.length}`);
assert(pl.length === ca.length, "pseudo-labels align");
assert(sim.perResidueMaxForce().length === sim.info.nResidues, "per-residue forces");
sim.resetPseudoLabels();

const mt = sim.metrics();
for (const k of ["m1", "m2", "m3", "m4", "m5", "rg", "stability_margin", "rmsf"]) {
  assert(Number.isFinite(mt[k]), `metrics.${k}`);
}
assert(mt.n_ss_ref >= 0 && mt.n_ss_kept <= mt.n_ss_ref, "ss counters");

const et = sim.energyTerms();
assert(Math.abs(et.total - et.nonbonded - et.bonded) < 1e-6, "energy split");
const st = sim.speciesTemperatures();
assert(st.water_t_k > 0 && st.solute_dof > 0, "species temps");
assert(sim.thermoInfo().thermo_dof > 0, "thermo info");
const wr = sim.waterRigidSplit();
assert(Math.abs(wr.water_internal_ke_kcal) < 0.05 * Math.abs(wr.water_total_ke_kcal) + 1e-9,
  `settle leak: internal ${wr.water_internal_ke_kcal} of ${wr.water_total_ke_kcal}`);
assert(sim.envInfo().efield.length === 3, "env efield");
assert(sim.exclusionDiagnostics().excluded_12_13 > 0, "exclusions");
const dump = sim.debugStateDump();
assert(dump.sites.length === sim.info.nAtoms + 4 * sim.info.nWater, "dump sites");
assert(Number.isFinite(dump.w_constr), "dump virial buckets");
assert(sim.computationTime().steps >= 0, "computation time");
assert(sim.forceReport(0).length === sim.info.nAtoms, "force report rows");
assert(sim.clashReport(0).length <= sim.info.nAtoms, "clash rows");

sim.setIntegrator("nve");
assert(Number.isFinite(sim.step(1).u_total_kcal), "nve step");
sim.setIntegrator("langevin_middle");

sim.setTrend("rl_fail_fast");
assert(sim.hasTrend() === true, "trend armed");
sim.step(1);
sim.resetTrend();
sim.clearTrend();
assert(sim.hasTrend() === false, "trend cleared");

sim.setSkipWaterThermostat(true);
sim.step(1);
sim.setSkipWaterThermostat(false);

sim.addRestraint(sel[0], sel[1], 4.0, 5.0);
sim.step(1);
assert(sim.updateRestraint(0, 5.0, 5.0) === true, "restraint retarget");
assert(sim.updateRestraint(999, 5.0, 5.0) === false, "restraint idx oob");
sim.clearRestraints();

sim.set({ tempK: 300, soluteK: 300 });
const eqObs = sim.equilibrate({ rampSteps: 10, holdSteps: 5, tStartK: 260 });
assert(Number.isFinite(eqObs.u_total_kcal), "equilibrate observables");

const probe = sim.rigidScaleProbe(1.001);
assert(Number.isFinite(probe.u_kcal) && Number.isFinite(probe.pressure_bar), "rigid scale probe");
sim.step(1); // forces recompute; run stays healthy after the mutating probe

// rebuild with an external field to prove the new build knob reaches env_info
sim.build(cif, { tempK: 300, soluteK: 300, boxPadA: 6, relaxIters: 200, strictIncomplete: false, efieldJson: [0.1, 0, 0], efieldOmega: 1.5 });
sim.set({ tempK: 300, soluteK: 300 });
const env2 = sim.envInfo();
assert(Math.abs(env2.efield[0] - 0.1) < 1e-6 && Math.abs(env2.efield_omega - 1.5) < 1e-6,
  `efield build: ${JSON.stringify(env2.efield)}/${env2.efield_omega}`);
assert(Number.isFinite(sim.step(1).u_total_kcal), "oscillating-field step");

sim.dispose();
assert(sim.info === null, "disposed");
console.log(`frames=${frames} sites=${snap.nSites} build=${sim.spice.version?.() ?? "ok"}`);
console.log("API_TEST_PASSED");
killServer();
process.exit(0);
