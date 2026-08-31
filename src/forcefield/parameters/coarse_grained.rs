//! Explicit atom-to-bead construction for Martini-style coarse graining.
//! A bead is a new particle with provenance, never an alias for an atom.

use lin_alg::f32::Vec3;

use super::mapping::{MartiniMapping, MartiniParseError, MartiniResidueTopology};

#[derive(Clone, Debug, PartialEq)]
pub struct CoarseGrainedBead {
    pub name: String,
    pub position: Vec3,
    pub source_atoms: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CoarseGrainedTopology {
    pub beads: Vec<CoarseGrainedBead>,
    pub bonds: Vec<(usize, usize)>,
    pub atom_to_bead: Vec<Option<usize>>,
}

/// Construct bead coordinates by averaging the mapped atom coordinates.
/// Missing atoms are rejected; atoms explicitly mapped to `None` are retained
/// in `atom_to_bead` as unrepresented all-atom input, not silently promoted to
/// beads.
pub fn build_coarse_grained_topology(
    mapping: &MartiniMapping,
    atom_names: &[&str],
    positions: &[Vec3],
    residue_topology: Option<&MartiniResidueTopology>,
) -> Result<CoarseGrainedTopology, MartiniParseError> {
    if atom_names.len() != positions.len() {
        return Err(MartiniParseError(
            "atom name/position length mismatch".into(),
        ));
    }
    let mut beads = mapping
        .beads
        .iter()
        .map(|name| CoarseGrainedBead {
            name: name.clone(),
            position: Vec3::new_zero(),
            source_atoms: Vec::new(),
        })
        .collect::<Vec<_>>();
    let mut atom_to_bead = vec![None; atom_names.len()];
    for (atom_idx, atom_name) in atom_names.iter().enumerate() {
        let Some(assignment) = mapping.assignments.iter().find(|a| a.atom == *atom_name) else {
            continue;
        };
        let Some(bead_name) = &assignment.bead else {
            continue;
        };
        let bead_idx = mapping
            .beads
            .iter()
            .position(|b| b == bead_name)
            .ok_or_else(|| MartiniParseError(format!("unknown bead '{}'", bead_name)))?;
        beads[bead_idx].source_atoms.push(atom_idx);
        beads[bead_idx].position += positions[atom_idx];
        atom_to_bead[atom_idx] = Some(bead_idx);
    }
    for bead in &mut beads {
        if bead.source_atoms.is_empty() {
            return Err(MartiniParseError(format!(
                "bead '{}' has no source atoms",
                bead.name
            )));
        }
        bead.position *= 1.0 / bead.source_atoms.len() as f32;
    }
    let bonds = residue_topology
        .map(|top| {
            top.bonds
                .iter()
                .filter_map(|bond| {
                    let a = beads.iter().position(|b| b.name == bond.first)?;
                    let b = beads.iter().position(|b| b.name == bond.second)?;
                    Some((a, b))
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(CoarseGrainedTopology {
        beads,
        bonds,
        atom_to_bead,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forcefield::parameters::mapping::parse_mapping;

    #[test]
    fn creates_new_beads_with_provenance_and_centers() {
        let mapping = parse_mapping(
            "[ molecule ]\nALA\n[ from ]\nu\n[ to ]\nm\n[ martini ]\nBB SC1\n[ atoms ]\n1 N BB\n2 CA BB\n3 CB SC1\n",
        ).unwrap();
        let names = ["N", "CA", "CB"];
        let pos = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
        ];
        let top = build_coarse_grained_topology(&mapping, &names, &pos, None).unwrap();
        assert_eq!(top.beads.len(), 2);
        assert_eq!(top.beads[0].source_atoms, vec![0, 1]);
        assert_eq!(top.beads[0].position.x, 1.0);
        assert_eq!(top.atom_to_bead, vec![Some(0), Some(0), Some(1)]);
    }
}
