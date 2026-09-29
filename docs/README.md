# spice_engine documentation

`spice_engine` (v1.3.9) is a **general-purpose all-atom molecular-dynamics engine** — a Rust core that compiles to a PyO3 Python extension, an rlib for Rust consumers, and a 1.8 MB WebAssembly module that runs PME MD in a browser tab. It takes atom coordinates in, builds a fully solvated, charge- and protonation-consistent all-atom system, runs NVT/NPT dynamics, and streams structure/energy/health signals out. **Protein adaptive evolution (SPICE's RL loop and stability scans) is the application that drives development — a consumer of the engine, not its definition**; nothing ML-related runs inside this crate.

## Reading paths

| Who you are | Read in this order |
|---|---|
| **Just using the engine (Python user)** | [capabilities.md](capabilities.md): the full API tables, the environment-knob matrix, the limitations list. That's enough. |
| **New engine contributor (MD background optional)** | capabilities.md first, then the code: the hot path is `src/engine/` (the `#[path]` chain under `engine/core/mod.rs` maps every subsystem to its own file). Each subsystem's "why" lives in its own doc comments, and the measured decision history lives in the named integration tests under `tests/` — those are the engineering ledger. |
| **Reviewing code / debugging one specific thing** | capabilities.md cites the owning method for every knob; follow it into the code, then the test with the matching name. |

## Pages

| Document | Contents |
|---|---|
| [capabilities.md](capabilities.md) | **Capability reference**: every Python entry point (`Structure`/`Env`/`Engine`, scans, sequence tools), the step-dict key contract, the observability getters, the knob-reachability matrix, limitations & "explicitly not done" decisions, the compatibility contract. §11 covers the browser build. |
| [web_demo.md](web_demo.md) | **WebAssembly port (v1.3.9)**: why the core compiled unchanged, the dependency-graph surgery (`web` feature graph), the hand-rolled extern-"C" ABI, API list, measured in-tab budget. Runnable demo + JS loader live in [`web/`](../web/README.md) (`make web-serve`). |

## The one-minute version

A protein CIF → (drop alternate conformations, rebuild hydrogens from the charge library as sole authority, set protonation by pH, gate out incomplete residues, assign types & charges) → placed in a periodic box, tiled with OPC water, NaCl/divalent ions inserted by displacing waters at a target concentration, L-BFGS-stress-relaxed → `SpiceEngine` advances at 2 fs steps: every step computes bonded + electrostatic (LJ/real-space-Coulomb/SPME) forces, integrates (velocity-Verlet + fused Langevin), constrains rigid water and X-H bonds, optionally scales the box (C-rescale) → every step returns `{energy, Cα coords, kinetic temperature, crashed?, trend_alarm}` (instantaneous pressure is a method call: `pressure_bar()`), plus five structural-health metrics (m1..m5 + Rg). The RL loop scores them; the stability scans turn them into pH×T×I phase maps.

All numbers, code anchors, and decision histories live in code comments and in the named tests. That is where every claim ultimately points.
