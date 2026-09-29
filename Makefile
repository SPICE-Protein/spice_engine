# Deployment helpers for spice_engine.
# Usage: make build | check | test | install | wheel | clean
# Override the interpreter when deploying elsewhere: make install PY=/path/to/python

PY ?= /opt/homebrew/Caskroom/miniconda/base/envs/spice/bin/python
# Resolve the conda env root (…/envs/spice) from the interpreter path.
VENV := $(abspath $(dir $(PY))/..)

# --- macOS 26+ (Tahoe) build workaround -------------------------------------------
# The release profile strips symbols (`strip = "symbols"`), but macOS 26+ dyld then
# rejects the large LTO'd extension at import with:
#   "mis-aligned LINKEDIT string pool"  (the strip step leaves __LINKEDIT's string
#   table misaligned for the new dyld). Keeping symbols makes the module importable.
# Scoped to Darwin only, so Linux/Windows CI wheels still ship stripped.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  DEV_RUSTFLAGS := -C strip=none
else
  DEV_RUSTFLAGS :=
endif

.PHONY: build check test install wheel clean web-check wasm wasm-simd web-smoke web-verify web-serve web-dist web-nightly

## Compile the native lib only (fast feedback, no Python bindings).
build:
	cargo build --release

## Type-check everything including the PyO3 FFI.
check:
	cargo check --features python

## Full Rust test suite (MD integration tests; needs --release).
test:
	cargo test --release

## Build + install the Python extension into the active env (dev loop).
install:
	CONDA_PREFIX=$(VENV) VIRTUAL_ENV=$(VENV) RUSTFLAGS="$(DEV_RUSTFLAGS) $(RUSTFLAGS)" $(PY) -m maturin develop --release

## Build a distributable wheel into target/wheels.
wheel:
	RUSTFLAGS="$(DEV_RUSTFLAGS) $(RUSTFLAGS)" $(PY) -m maturin build --release

## Clean all build artifacts.
clean:
	cargo clean
	rm -rf target/wheels

# --- Browser (wasm32) targets (v1.3.9 web port) ---------------------------------
# RUSTFLAGS is scoped PER RECIPE so the `getrandom_backend` cfg can never leak
# into a native build and the wheel / .cargo/config stay untouched.
WASM_TARGET := wasm32-unknown-unknown
WEB_FEATURES := --no-default-features --features web
# `unsupported` makes getrandom compile (runtime Err, never called — entropy.rs
# seeds via clock+counter); wasm_js is FORBIDDEN (pulls wasm-bindgen/js-sys, and
# there is no wasm-bindgen CLI offline to post-process its glue).
WEB_RUSTFLAGS := --cfg getrandom_backend=\"unsupported\"
WEB_SIMD_RUSTFLAGS := $(WEB_RUSTFLAGS) -C target-feature=+simd128

## Type-check the browser build (web feature, wasm target, scalar).
web-check:
	RUSTFLAGS="$(WEB_RUSTFLAGS)" cargo check --offline $(WEB_FEATURES) --target $(WASM_TARGET)

## Release-build the baseline wasm (scalar fallback — runs everywhere).
wasm:
	RUSTFLAGS="$(WEB_RUSTFLAGS)" cargo build --offline --release $(WEB_FEATURES) --target $(WASM_TARGET)
	@ls -lh target/$(WASM_TARGET)/release/spice_engine.wasm

## Release-build the simd128 wasm (Chrome 101+/Firefox 100+/Safari 16+).
wasm-simd:
	RUSTFLAGS="$(WEB_SIMD_RUSTFLAGS)" cargo build --offline --release $(WEB_FEATURES) --target $(WASM_TARGET)
	@cp target/$(WASM_TARGET)/release/spice_engine.wasm target/$(WASM_TARGET)/release/spice_engine_simd.wasm
	@ls -lh target/$(WASM_TARGET)/release/spice_engine_simd.wasm

## Node end-to-end smoke of the baseline wasm (build 2LYZ + step + read back).
web-smoke: wasm
	node --max-old-space-size=4096 web/smoke.mjs target/$(WASM_TARGET)/release/spice_engine.wasm

## Build both variants and smoke each (correctness gate; simd is non-bit-exact).
web-verify: wasm wasm-simd
	node --max-old-space-size=4096 web/smoke.mjs target/$(WASM_TARGET)/release/spice_engine.wasm
	node --max-old-space-size=4096 web/smoke.mjs target/$(WASM_TARGET)/release/spice_engine_simd.wasm

## Stage the demo binaries under web/dist/ (gzipped; NOT committed — the repo
## keeps no wasm blobs, see web-nightly for the CI-built alternative).
web-dist: wasm wasm-simd
	@mkdir -p web/dist
	@cp target/$(WASM_TARGET)/release/spice_engine.wasm target/$(WASM_TARGET)/release/spice_engine_simd.wasm web/dist/
	@gzip -kf web/dist/spice_engine.wasm web/dist/spice_engine_simd.wasm
	@ls -lh web/dist/

## Fetch the nightly wasm from the public GitHub Release into web/dist/ —
## NO token needed (Actions artifacts themselves require auth even on public
## repos, which is why the workflow mirrors the blobs to a rolling `nightly`
## prerelease). Plain curl; also works as a direct <script>/fetch URL in a page.
WEB_NIGHTLY_BASE := https://github.com/SPICE-Protein/spice_engine/releases/download/nightly
web-nightly:
	@mkdir -p web/dist
	curl -fSL $(WEB_NIGHTLY_BASE)/spice_engine.wasm.gz -o web/dist/spice_engine.wasm.gz
	curl -fSL $(WEB_NIGHTLY_BASE)/spice_engine_simd.wasm.gz -o web/dist/spice_engine_simd.wasm.gz
	@ls -lh web/dist/

## Serve the demo dir locally (builds the binaries if missing).
web-serve:
	@test -f web/dist/spice_engine.wasm.gz || $(MAKE) web-dist
	python3 -m http.server 8080 --directory web
