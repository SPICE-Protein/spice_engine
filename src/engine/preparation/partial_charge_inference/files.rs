//! For managing inference-related files

use std::{
    fs,
    fs::File,
    io,
    io::Write,
    path::{Path, PathBuf},
};

use bincode::{Decode, Encode};

pub const MODEL_PATH: &str = "geostd_model.safetensors";
pub const VOCAB_PATH: &str = "geostd_model.vocab";

/// Amber parameter distribution directory (GAFF frcmod/mol2 sources for
/// partial-charge inference). Developer-machine default; override with the
/// `AMBER_GEOSTD_PATH` environment variable.
pub const GEOSTD_PATH: &str = "C:/users/the_a/Desktop/bio_misc/Amber/amber_geostd";

/// Resolved GEOSTD directory: `AMBER_GEOSTD_PATH` if set, else the
/// developer default in [`GEOSTD_PATH`].
pub fn geostd_path() -> PathBuf {
    match std::env::var_os("AMBER_GEOSTD_PATH") {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(GEOSTD_PATH),
    }
}

/// Find Mol2  paths. Assumes there are per-letter subfolders one-layer deep.
/// todo: FRCmod as well
pub fn find_mol2_paths(geostd_dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut result = Vec::new();

    for entry in fs::read_dir(geostd_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            for subentry in fs::read_dir(&path)? {
                let subentry = subentry?;
                let subpath = subentry.path();

                if subpath
                    .extension()
                    .map(|e| e.to_string_lossy().to_lowercase())
                    == Some("mol2".to_string())
                {
                    result.push(subpath);
                }
            }
        }
    }

    Ok(result)
}

/// Load from file, using Bincode. We currently use this for preference files.
pub(crate) fn load_from_bytes_bincode<T: Decode<()>>(buffer: &[u8]) -> io::Result<T> {
    let config = bincode::config::standard();

    let (decoded, _len) = match bincode::decode_from_slice(buffer, config) {
        Ok(v) => v,
        Err(_) => {
            eprintln!("Error loading from file. Did the format change?");
            return Err(io::Error::other("error loading"));
        }
    };
    Ok(decoded)
}

/// Save to file, using Bincode. We currently use this for preference files.
///
/// v1.3.9: moved here from the candle module's `mod.rs` — this file is the
/// candle-free half of partial-charge inference (plain bincode I/O), and is
/// compiled as `md_core::pci_files` even in slim builds (the embedded water
/// template loads through `load_from_bytes_bincode` at solvent init).
pub(crate) fn save<T: Encode>(path: &Path, data: &T) -> io::Result<()> {
    let config = bincode::config::standard();

    let encoded: Vec<u8> = bincode::encode_to_vec(data, config).unwrap();

    let mut file = File::create(path)?;
    file.write_all(&encoded)?;
    Ok(())
}
