// Node smoke test for the spice_engine wasm build (v1.3.9 web port).
// Validates the raw extern-"C" ABI end-to-end without a browser: instantiate
// with only the two `env` imports the module declares, build a solvated system
// from a real mmCIF, step it, and read positions/observables back.
//
// Run:  node web/smoke.mjs [path/to/spice_engine.wasm] [path/to/some.cif]
import fs from "node:fs";

const wasmPath = process.argv[2] ?? "target/wasm32-unknown-unknown/release/spice_engine.wasm";
const cifPath = process.argv[3] ?? "data/test/2LYZ.cif";

let exports = null;
let mem = null;

const imports = {
  env: {
    // Monotonic ms clock; the loader would bind performance.now() in a browser.
    now_ms: () => Number(process.hrtime.bigint() / 1000n) / 1000.0,
    // Console sink: read a UTF-8 (ptr,len) string out of linear memory.
    log: (ptr, len) => {
      const s = new TextDecoder().decode(new Uint8Array(mem.buffer, ptr, len));
      console.log("[wasm]", s);
    },
  },
};

const buf = fs.readFileSync(wasmPath);
const module = new WebAssembly.Module(buf);
console.log("== imports ==");
for (const im of WebAssembly.Module.imports(module)) {
  console.log(`  ${im.module}.${im.name}  (${im.kind})`);
}
console.log("== spice_* exports ==");
const spiceExports = WebAssembly.Module.exports(module).filter((e) => e.name.startsWith("spice_"));
console.log(spiceExports.map((e) => e.name).sort().join(", "));

const instance = new WebAssembly.Instance(module, imports);
exports = instance.exports;
mem = exports.memory;
console.log(`memory pages: ${mem.buffer.byteLength / 65536} (${(mem.buffer.byteLength / 1e6).toFixed(1)} MB)`);

// ---- tiny helpers over the raw ABI ----------------------------------------
function allocAndWrite(bytes) {
  const ptr = exports.spice_alloc(bytes.length);
  new Uint8Array(mem.buffer, ptr, bytes.length).set(bytes);
  return ptr;
}
function callStr(fn, str) {
  const enc = new TextEncoder().encode(str);
  const ptr = allocAndWrite(enc);
  const rc = fn(ptr, enc.length);
  exports.spice_free(ptr, enc.length);
  return rc;
}
function readOut() {
  const ptr = exports.spice_out_ptr();
  const len = exports.spice_out_len();
  return new TextDecoder().decode(new Uint8Array(mem.buffer, ptr, len));
}

function assert(cond, msg) {
  if (!cond) {
    console.error("FAIL:", msg);
    process.exit(1);
  }
}

// ---- drive the API ---------------------------------------------------------
assert(exports.spice_init() === 0, "spice_init returned non-zero");

exports.spice_version();
const version = readOut();
console.log("engine version:", version);
assert(/^\d+\.\d+\.\d+/.test(version), "version string not semver-ish");

const cif = fs.readFileSync(cifPath, "utf8");
// NVT demo config: no barostat (pressure_bar <= 0 => None), modest pad, small
// relaxation so the scalar single-thread wasm finishes quickly.
const params = JSON.stringify({
  ph: 7.0,
  temp_k: 300.0,
  pressure_bar: 0.0,
  ionic_strength_m: 0.0,
  relax_iters: 200,
  tolerance: 2.0,
  strict_incomplete: false,
  box_pad_a: 6.0,
});

const t0 = process.hrtime.bigint();
// spice_build_mmcif(cif_ptr, cif_len, params_ptr, params_len)
const cifEnc = new TextEncoder().encode(cif);
const parEnc = new TextEncoder().encode(params);
const cifPtr = allocAndWrite(cifEnc);
const parPtr = allocAndWrite(parEnc);
const handle = exports.spice_build_mmcif(cifPtr, cifEnc.length, parPtr, parEnc.length);
exports.spice_free(cifPtr, cifEnc.length);
exports.spice_free(parPtr, parEnc.length);
const buildSec = Number(process.hrtime.bigint() - t0) / 1e9;

const buildInfo = JSON.parse(readOut());
console.log(`build handle=${handle} in ${buildSec.toFixed(1)}s ->`, buildInfo);
assert(handle > 0, `build failed (rc=${handle}): ${buildInfo.error}`);
assert(buildInfo.n_atoms > 20, "unexpectedly few solute atoms");
assert(buildInfo.n_water > 20, "unexpectedly few waters");

// positions snapshot
const nSites = exports.spice_positions(handle);
assert(nSites > 0, "no positions");
const posPtr = exports.spice_positions_ptr();
const posLen = exports.spice_positions_len();
assert(posLen === nSites * 3, `positions len ${posLen} != sites*3 ${nSites * 3}`);
const posA = new Float32Array(mem.buffer.slice(posPtr, posPtr + posLen * 4));
assert(posA.every(Number.isFinite), "initial positions contain non-finite values");

// step a handful of times, timed
let metrics = null;
const N_STEPS = 50;
const tStep0 = process.hrtime.bigint();
for (let i = 0; i < N_STEPS; i++) {
  const rc = exports.spice_step(handle);
  metrics = JSON.parse(readOut());
  assert(rc >= 0, `step ${i} errored: ${metrics.error ?? rc}`);
  assert(!metrics.crashed, `step ${i} crashed: ${metrics.crash_reason}`);
  assert(Number.isFinite(metrics.u_total_kcal), `step ${i} non-finite energy`);
}
const stepMs = Number(process.hrtime.bigint() - tStep0) / 1e6 / N_STEPS;
const nsPerDay = (1000 / stepMs) * (2e-6) * 86400; // 2 fs/step
console.log(`timing: ${stepMs.toFixed(2)} ms/step  (~${nsPerDay.toFixed(3)} ns/day @2fs, scalar single-thread wasm)`);
console.log(`after ${N_STEPS} steps:`, {
  step_count: metrics.step_count,
  time_ps: metrics.time_ps,
  u_total_kcal: metrics.u_total_kcal,
  pressure_bar: metrics.pressure_bar,
  solute_t_k: metrics.solute_t_k,
  water_t_k: metrics.water_t_k,
  rg: metrics.rg,
});
assert(metrics.step_count >= N_STEPS, "step_count did not advance");

// positions changed
const nSites2 = exports.spice_positions(handle);
const posB = new Float32Array(mem.buffer.slice(exports.spice_positions_ptr(), exports.spice_positions_ptr() + exports.spice_positions_len() * 4));
assert(nSites2 === nSites, "site count changed between snapshots");
let moved = 0;
for (let i = 0; i < posA.length; i++) if (posA[i] !== posB[i]) moved++;
assert(moved > posA.length * 0.5, `fewer than half the coordinates moved (${moved}/${posA.length})`);
assert(posB.every(Number.isFinite), "post-step positions non-finite");

// control surface smoke: temperature / timestep / observables
assert(exports.spice_set_temperature(handle, 310.0, 310.0) === 0, "set_temperature failed");
assert(exports.spice_set_timestep(handle, 0.002) === 0, "set_timestep failed");
assert(exports.spice_set_gamma(handle, 1.0) === 0, "set_gamma failed");
exports.spice_observables(handle);
const obs = JSON.parse(readOut());
for (const key of ["u_total_kcal", "pressure_bar", "solute_t_k", "water_t_k", "rg", "dt_ps"]) {
  assert(key in obs, `observables missing ${key}`);
}
console.log("observables keys ok:", Object.keys(obs).length, "fields");

assert(exports.spice_free_engine(handle) === 0, "free_engine failed");

console.log("\nALL_SMOKE_CHECKS_PASSED  build_s=" + buildSec.toFixed(1));
