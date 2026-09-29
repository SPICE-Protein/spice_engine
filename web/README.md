# `web/` — spice_engine in the browser (WebAssembly)

The demo lives **in this repo** (no separate repository). Full API reference,
feature-graph, ABI design notes and measured performance budget:
[`docs/web_demo.md`](../docs/web_demo.md); capability summary:
[`docs/capabilities.md` §11](../docs/capabilities.md).

| file | role |
|---|---|
| `api.mjs` | **the Web API front-ends should use** — ergonomic facade over the loader: `Sim.load(cif, {env})`, `step/stepAction`, `set({tempK,soluteK,...})`, `setForcesOff`, `snapshot()`, `animate(onFrame)`, `dispose()`, plus the v1.3.10 FFI-parity probes (`esp/field/sasa/select/contacts/bottleneck/metrics/equilibrate/restraints/trend/integrator/debugStateDump/…` — same keys as the Python engine). Zero deps, browser + node ≥ 18 |
| `api.d.ts` | TypeScript declarations for `api.mjs` (exact wasm ABI metric keys) |
| `api.test.mjs` | self-contained end-to-end test of the Web API including every parity probe (`make web-apitest`) |
| `loader.mjs` | lower-level JS wrapper over the raw `extern "C"` ABI (`loadSpice({url})` → `SpiceEngine` handle class; gzip via client-side `DecompressionStream`) |
| `index.html` | minimal reference page — build → `requestAnimationFrame` stepping → canvas dots coloured by atom role, toy force-term switches, dual-bath temperature inputs |
| `mini.cif` | 5-residue fast fixture (~1.5 k sites, builds in ~3 s in-tab) |
| `dist/` | **gitignored** — wasm blobs are never committed. Populate with `make web-dist` (local build) or `make web-nightly` (curls the newest main-branch CI artifact through **nightly.link** — no GitHub token; artifact URLs 404 for anonymous users natively, nightly.link re-serves them for public repos) |
| `smoke.mjs` | node correctness gate — instantiate → build → step → read back (`make web-smoke` / `make web-verify`) |
| `selftest.mjs` | node end-to-end test of `loader.mjs` + `index.html` assets over http |

Try it:

```bash
make web-serve     # auto-builds web/dist/ if missing → http://localhost:8080
# or skip the toolchain entirely:
make web-nightly   # pull the latest CI artifact (scalar+simd) via nightly.link
```

JS API at a glance:

```js
import { Sim } from "./api.mjs";

const sim = await Sim.load(await (await fetch("mini.cif")).text(), {
  env: { tempK: 300, soluteK: 300, ionicStrengthM: 0.15, boxPadA: 6 },
  wasm: "dist/spice_engine.wasm.gz",
});
const stop = sim.animate((s, m) => {
  const { coords, roles } = s.snapshot();      // Float32Array / Int32Array copies
  drawPoints(coords, roles);                    // your renderer
  label.textContent = `${m.u_total_kcal.toFixed(1)} kcal/mol · ${m.time_ps.toFixed(2)} ps`;
}, { stepsPerFrame: 2 });
// sim.set({ tempK: 320, soluteK: 320 });  ·  sim.setForcesOff({ longRange: true })  ·  stop(); sim.dispose();
```

Protocol rule: always pass `soluteK` with `tempK` for thermostatted runs (single
bath leaves the solute at ~0.72-0.78× setpoint — water owns the heat capacity).

Bring your own renderer — `api.mjs` is the whole interface; `index.html` is
only a proof that the API round-trips.
