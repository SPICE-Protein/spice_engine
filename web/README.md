# `web/` — spice_engine in the browser (WebAssembly)

The demo lives **in this repo** (no separate repository). Full API reference,
feature-graph, ABI design notes and measured performance budget:
[`docs/web_demo.md`](../docs/web_demo.md); capability summary:
[`docs/capabilities.md` §11](../docs/capabilities.md).

| file | role |
|---|---|
| `loader.mjs` | dependency-free JS wrapper over the raw `extern "C"` ABI (`loadSpice({url})` → `SpiceEngine` handle class; gzip via client-side `DecompressionStream`) |
| `index.html` | minimal reference page — build → `requestAnimationFrame` stepping → canvas dots coloured by atom role, toy force-term switches, dual-bath temperature inputs |
| `mini.cif` | 5-residue fast fixture (~1.5 k sites, builds in ~3 s in-tab) |
| `dist/` | **gitignored** — wasm blobs are never committed. Populate with `make web-dist` (local build) or `make web-nightly` (curls the rolling **`nightly` GitHub Release** — no token; Actions artifacts themselves require auth even on public repos, which is why the workflow mirrors the blobs there) |
| `smoke.mjs` | node correctness gate — instantiate → build → step → read back (`make web-smoke` / `make web-verify`) |
| `selftest.mjs` | node end-to-end test of `loader.mjs` + `index.html` assets over http |

Try it:

```bash
make web-serve     # auto-builds web/dist/ if missing → http://localhost:8080
# or skip the toolchain entirely:
make web-nightly   # curl the nightly Release's scalar+simd .wasm.gz into web/dist/
```

Bring your own renderer — the loader is the whole interface; `index.html` is
only a proof that the API round-trips.
