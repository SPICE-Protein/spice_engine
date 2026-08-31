# CPU engine core migration boundary

The upstream `dynamics/src/lib.rs` is intentionally not copied here because it
couples CPU state, device selection, and GPU/CUDA resources in one monolithic
module. The SE migration keeps CPU state and integration split by responsibility
and will add SE-owned state types here incrementally.
