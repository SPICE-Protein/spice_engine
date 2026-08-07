//! Mutation utilities (a lightweight Rust-side layer).
//!
//! Architecture boundary: mutations with real side-chain modelling are generated
//! on the Python side (Rosetta / backbone tools), ingested through `structure.rs`
//! and rebuilt. This module only provides Rust-side conveniences:
//! - sequence validation and single-point mutation rewriting (for naming, logs,
//!   and cross-checking against `StructureInput::sequence()`).
//! - `apply_mutations` returns a new sequence that can be handed to the Python
//!   side to generate structure, or used directly in sequence-only tests.

use crate::structure::one_letter_from_resname;

/// Single-point mutation: `position` is a 0-based residue index, `to` is the
/// one-letter target residue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mutation {
    pub position: usize,
    pub to: char,
}

impl Mutation {
    pub fn new(position: usize, to: char) -> Self {
        Self { position, to }
    }
}

/// Validate that a sequence contains only standard one-letter amino acids (U
/// selenocysteine allowed).
pub fn validate_sequence(seq: &str) -> Result<(), String> {
    let valid = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V', 'U',
    ];
    for (i, c) in seq.chars().enumerate() {
        if !valid.contains(&c) {
            return Err(format!("invalid residue '{c}' at position {i}"));
        }
    }
    if seq.is_empty() {
        return Err("empty sequence".to_string());
    }
    Ok(())
}

/// Apply a set of mutations to a sequence in order (each position may be changed
/// once; duplicates are an error).
pub fn apply_mutations(seq: &str, ms: &[Mutation]) -> Result<String, String> {
    validate_sequence(seq)?;
    let mut chars: Vec<char> = seq.chars().collect();
    let mut touched = vec![false; chars.len()];
    for m in ms {
        if m.position >= chars.len() {
            return Err(format!(
                "mutation position {} out of range (len {})",
                m.position,
                chars.len()
            ));
        }
        if touched[m.position] {
            return Err(format!("duplicate mutation at position {}", m.position));
        }
        // target residue validity (checked by mapping back through three-letter)
        let mut found = false;
        for ch in [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
            'W', 'Y', 'V',
        ] {
            if ch == m.to {
                found = true;
                break;
            }
        }
        if !found {
            return Err(format!("invalid target residue '{}' at position {}", m.to, m.position));
        }
        chars[m.position] = m.to;
        touched[m.position] = true;
    }
    Ok(chars.into_iter().collect())
}

/// Three-letter residue name → one-letter (forwards to `structure`, for use by
/// mutation-related code).
pub fn one_letter(res_name: &str) -> Option<char> {
    one_letter_from_resname(res_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_tools() {
        let s = "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYF";
        validate_sequence(s).unwrap();
        assert!(validate_sequence("M K").is_err());

        let m_seq = apply_mutations(s, &[Mutation::new(0, 'A'), Mutation::new(5, 'W')]).unwrap();
        assert_eq!(m_seq.chars().next(), Some('A'));
        assert_eq!(m_seq.as_bytes()[5] as char, 'W');
        // unmutated positions preserved
        assert_eq!(m_seq.as_bytes()[1] as char, s.as_bytes()[1] as char);
        // out-of-range / duplicate
        assert!(apply_mutations(s, &[Mutation::new(999, 'A')]).is_err());
        assert!(apply_mutations(s, &[Mutation::new(0, 'A'), Mutation::new(0, 'G')]).is_err());
        assert!(apply_mutations(s, &[Mutation::new(0, 'X')]).is_err());
    }

    #[test]
    fn resname_map() {
        assert_eq!(one_letter("ALA"), Some('A'));
        assert_eq!(one_letter("Trp"), Some('W'));
        assert_eq!(one_letter("HOH"), None);
    }
}
