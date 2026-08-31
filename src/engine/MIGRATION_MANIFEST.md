# CPU migration manifest

Copied from the local `dynamics` source as a migration baseline:

- `integrator/`: integration and thermostat
- `geometry/`: box and configuration
- `neighbors/`: neighbor-list implementation remains under `forcefield/neighbors/`
- `constraints/`: bonded constraint/minimization support
- `solvent/`: CPU solvent initialization, OPC SETTLE, packing and templates
- `chemistry/`: alchemical, hydrogen-bond and surface helpers
- `utility/`: timing and shared utilities
- `forcefield/`: bonded, non-bonded, parameters and CPU SIMD sources

Intentionally not copied:

- GPU/CUDA implementation and CUDA kernels
- ML charge-inference and training sources
- the monolithic `dynamics/src/lib.rs` state/device/GPU module
- trajectory-only and test-only files

The copied Rust sources are migration baselines, not yet compiled SE modules;
imports referencing private `dynamics` types must be replaced before activation.
