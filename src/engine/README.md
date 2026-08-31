# Engine migration baseline

This directory contains CPU-only source snapshots copied from the current
`dynamics` implementation while SPICE Engine is being detached from that crate.
The snapshots are organized by responsibility and are not compiled as modules
until their `dynamics`-internal imports are replaced by SE-owned types.

Intentionally excluded: GPU/CUDA sources, ML charge-inference sources, and
training binaries.
