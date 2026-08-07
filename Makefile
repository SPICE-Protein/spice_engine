# Deployment helpers for spice_engine.
# Usage: make build | check | test | install | wheel | clean
# Override the interpreter when deploying elsewhere: make install PY=/path/to/python

PY ?= /opt/homebrew/Caskroom/miniconda/base/envs/spice/bin/python
# Resolve the conda env root (…/envs/spice) from the interpreter path.
VENV := $(abspath $(dir $(PY))/..)

.PHONY: build check test install wheel clean

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
	CONDA_PREFIX=$(VENV) VIRTUAL_ENV=$(VENV) $(PY) -m maturin develop --release

## Build a distributable wheel into target/wheels.
wheel:
	$(PY) -m maturin build --release

## Clean all build artifacts.
clean:
	cargo clean
	rm -rf target/wheels
