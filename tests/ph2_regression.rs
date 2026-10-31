//! Regression: a low-pH build must type protonated Asp/Glu as ASH/GLH so the
//! carboxyl O–H uses the OH–HO bond terms. Before the fix, pH 2 left an
//! O2–HB2 pair with no parameters and the build aborted with
//! "Missing bond params for O2-HB2" (4B89 and four other X-ray entries).

use bio_files::MmCif;
use spice_engine::engine::md_core::ComputationDevice;
use spice_engine::engine::md_core::params::FfParamSet;
use spice_engine::env::EnvParams;
use spice_engine::{BuildOptions, build_system};
use std::path::Path;

#[test]
#[ignore = "expensive: full 4B89 build at pH 2"]
fn low_ph_build_survives_protonated_carboxyl() {
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("load Amber parameters");
    let protein = MmCif::load(Path::new("data/test/4B89_clean.cif")).expect("load 4B89");
    let opts = BuildOptions {
        env: EnvParams::new(2.0, 298.0, 1.0, 0.15),
        strict_incomplete_residues: false,
        ..Default::default()
    };
    let engine = build_system(&dev, &params, protein, &opts).expect("pH 2 build");
    assert!(!engine.state.atoms.is_empty());
}

/// C-terminal Asp/Glu takes the same path: the C-terminal charge lib has no
/// ASH/GLH unit, so the H map must keep the standard (deprotonated) carboxyl
/// even at pH 2. 1QQE ends in Asp.
#[test]
#[ignore = "expensive: full 1QQE build at pH 2"]
fn low_ph_build_survives_c_terminal_acid() {
    let dev = ComputationDevice::Cpu;
    let params = FfParamSet::new_amber().expect("load Amber parameters");
    let protein = MmCif::load(Path::new("data/test/1QQE_clean.cif")).expect("load 1QQE");
    let opts = BuildOptions {
        env: EnvParams::new(2.0, 298.0, 1.0, 0.15),
        strict_incomplete_residues: false,
        ..Default::default()
    };
    let engine = build_system(&dev, &params, protein, &opts).expect("pH 2 build");
    assert!(!engine.state.atoms.is_empty());
}
