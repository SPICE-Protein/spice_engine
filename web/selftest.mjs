// Node harness for loader.mjs — exercises the JS API end-to-end over http
// (fetch + DecompressionStream of the .wasm.gz), the same path a browser takes.
// Run from the repo root with a static server on :8099:
//   python3 -m http.server 8099 &  node selftest.mjs
import { loadSpice } from "./loader.mjs";

const BASE = "http://127.0.0.1:8099";
const cif = await (await fetch(`${BASE}/mini.cif`)).text();

const spice = await loadSpice({ url: `${BASE}/dist/spice_engine.wasm.gz` }, { log: (m) => console.log("[spice]", m) });
console.log("version:", spice.version());

const eng = spice.build(cif, { ph: 7, temp_k: 300, pressure_bar: 0, relax_iters: 150, tolerance: 2.0, strict_incomplete: false, box_pad_a: 6 });
console.log("build info:", eng.info);

eng.setTemperature(300, { soluteK: 300 });
eng.setTimestep(0.002);
eng.setGamma(0.5);

const p0 = eng.positions();
let m;
for (let i = 0; i < 30; i++) m = eng.step();
const p1 = eng.positions();

const moved = (() => { let k = 0; for (let i = 0; i < p1.coords.length; i++) if (p0.coords[i] !== p1.coords[i]) k++; return k; })();
const obs = eng.observables();
console.log("after 30 steps:", {
  step: m.step_count, t_ps: m.time_ps, U: m.u_total_kcal,
  solute_t_k: m.solute_t_k, water_t_k: m.water_t_k, pressure_bar: m.pressure_bar, rg: m.rg,
  sites: p1.nSites, moved_floats: moved,
});
console.log("observables fields:", Object.keys(obs).length);

// toy toggle + bias force path
eng.setForceOverrides({ bonded: 1 });
eng.setForceOverrides({ bonded: 0 });
const bf = new Float32Array(eng.info.n_atoms * 3); bf[0] = 1.0;
const am = eng.stepAction(bf);
console.log("stepAction ok, finite:", Number.isFinite(am.u_total_kcal));

eng.resetVelocities();
eng.free();

const ok = m.step_count >= 30 && p1.nSites > 0 && moved > p1.coords.length / 2
  && Number.isFinite(m.u_total_kcal) && "solute_t_k" in obs;
console.log(ok ? "LOADER_SELFTEST_PASSED" : "LOADER_SELFTEST_FAILED");
process.exit(ok ? 0 : 1);
