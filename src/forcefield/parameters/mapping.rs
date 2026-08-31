//! SE-owned Martini 3 mapping and coarse-grained residue topology descriptors.
//!
//! These descriptors deliberately stop at atom-to-bead construction metadata;
//! they do not reinterpret beads as all-atom particles.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtomBeadAssignment {
    pub atom: String,
    pub bead: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MartiniMapping {
    pub residue: String,
    pub from: String,
    pub to: String,
    pub beads: Vec<String>,
    pub assignments: Vec<AtomBeadAssignment>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MartiniBead {
    pub name: String,
    pub bead_type: String,
    pub charge: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MartiniBond {
    pub first: String,
    pub second: String,
    pub function: i32,
    pub length: f32,
    pub force_constant: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MartiniResidueTopology {
    pub residue: String,
    pub nrexcl: usize,
    pub beads: Vec<MartiniBead>,
    pub bonds: Vec<MartiniBond>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MartiniParseError(pub String);

impl fmt::Display for MartiniParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for MartiniParseError {}

fn section(line: &str) -> Option<String> {
    let line = line.trim();
    line.strip_prefix('[')?
        .strip_suffix(']')
        .map(|s| s.trim().to_ascii_lowercase())
}

fn data_line(line: &str) -> Option<String> {
    let line = line.split(';').next()?.trim();
    if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
        None
    } else {
        Some(line.into())
    }
}

pub fn parse_mapping(text: &str) -> Result<MartiniMapping, MartiniParseError> {
    let mut current = String::new();
    let mut residue = None;
    let mut from = None;
    let mut to = None;
    let mut beads = Vec::new();
    let mut assignments = Vec::new();
    for raw in text.lines() {
        if let Some(s) = section(raw) {
            current = s;
            continue;
        }
        let Some(line) = data_line(raw) else { continue };
        match current.as_str() {
            "molecule" if residue.is_none() => {
                residue = Some(
                    line.split_whitespace()
                        .next()
                        .unwrap_or_default()
                        .to_string(),
                )
            }
            "from" if from.is_none() => {
                from = Some(
                    line.split_whitespace()
                        .next()
                        .unwrap_or_default()
                        .to_string(),
                )
            }
            "to" if to.is_none() => {
                to = Some(
                    line.split_whitespace()
                        .next()
                        .unwrap_or_default()
                        .to_string(),
                )
            }
            "martini" => beads.extend(line.split_whitespace().map(str::to_owned)),
            "atoms" => {
                let fields: Vec<_> = line.split_whitespace().collect();
                if fields.len() >= 3 {
                    let bead = if fields[2].starts_with('!') {
                        None
                    } else {
                        Some(fields[2].to_owned())
                    };
                    assignments.push(AtomBeadAssignment {
                        atom: fields[1].to_owned(),
                        bead,
                    });
                }
            }
            _ => {}
        }
    }
    let mapping = MartiniMapping {
        residue: residue.ok_or_else(|| MartiniParseError("mapping has no molecule".into()))?,
        from: from.ok_or_else(|| MartiniParseError("mapping has no from section".into()))?,
        to: to.ok_or_else(|| MartiniParseError("mapping has no to section".into()))?,
        beads,
        assignments,
    };
    validate_mapping(&mapping)?;
    Ok(mapping)
}

pub fn validate_mapping(mapping: &MartiniMapping) -> Result<(), MartiniParseError> {
    if mapping.beads.is_empty() {
        return Err(MartiniParseError("mapping has no beads".into()));
    }
    let bead_set: std::collections::HashSet<_> = mapping.beads.iter().collect();
    let mut atoms = std::collections::HashSet::new();
    for a in &mapping.assignments {
        if !atoms.insert(&a.atom) {
            return Err(MartiniParseError(format!("duplicate atom '{}'", a.atom)));
        }
        if let Some(bead) = &a.bead
            && !bead_set.contains(bead)
        {
            return Err(MartiniParseError(format!(
                "atom '{}' refers to unknown bead '{}'",
                a.atom, bead
            )));
        }
    }
    if mapping.assignments.is_empty() {
        return Err(MartiniParseError("mapping has no atom assignments".into()));
    }
    Ok(())
}

pub fn parse_residue_topology(
    text: &str,
    wanted: &str,
) -> Result<MartiniResidueTopology, MartiniParseError> {
    let mut current = String::new();
    let mut residue = String::new();
    let mut nrexcl = 1usize;
    let mut beads = Vec::new();
    let mut bonds = Vec::new();
    let mut in_wanted = false;
    for raw in text.lines() {
        if let Some(s) = section(raw) {
            current = s;
            continue;
        }
        let Some(line) = data_line(raw) else { continue };
        let fields: Vec<_> = line.split_whitespace().collect();
        if current == "moleculetype" && fields.len() >= 2 {
            if in_wanted {
                break;
            }
            residue = fields[0].to_owned();
            nrexcl = fields[1]
                .parse()
                .map_err(|_| MartiniParseError(format!("invalid nrexcl for {}", residue)))?;
            in_wanted = residue.eq_ignore_ascii_case(wanted);
            if in_wanted {
                beads.clear();
                bonds.clear();
            }
            continue;
        }
        if !in_wanted {
            continue;
        }
        if current == "atoms" && fields.len() >= 7 {
            let charge = fields[6]
                .parse()
                .map_err(|_| MartiniParseError(format!("invalid charge for {}", residue)))?;
            beads.push(MartiniBead {
                name: fields[4].to_owned(),
                bead_type: fields[1].to_owned(),
                charge,
            });
        } else if (current == "bonds" || current == "constraints") && fields.len() >= 4 {
            let function = fields[2]
                .parse()
                .map_err(|_| MartiniParseError("invalid bond function".into()))?;
            let length = fields[3]
                .parse()
                .map_err(|_| MartiniParseError("invalid bond length".into()))?;
            let force_constant = fields.get(4).and_then(|v| v.parse().ok());
            bonds.push(MartiniBond {
                first: fields[0].to_owned(),
                second: fields[1].to_owned(),
                function,
                length,
                force_constant,
            });
        }
    }
    if !in_wanted || beads.is_empty() {
        return Err(MartiniParseError(format!(
            "residue '{}' not found or has no beads",
            wanted
        )));
    }
    Ok(MartiniResidueTopology {
        residue,
        nrexcl,
        beads,
        bonds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_validates_ala_mapping() {
        let text = "[ molecule ]\nALA\n[ from ]\nuniversal\n[ to ]\nmartini3001\n[ martini ]\nBB SC1\n[ atoms ]\n1 N BB\n2 HN BB\n3 CA BB\n4 HA !BB\n5 CB SC1\n";
        let m = parse_mapping(text).unwrap();
        assert_eq!(m.residue, "ALA");
        assert_eq!(m.beads, vec!["BB", "SC1"]);
        assert_eq!(m.assignments.iter().filter(|a| a.bead.is_some()).count(), 4);
    }

    #[test]
    fn rejects_unknown_bead() {
        let text = "[ molecule ]\nX\n[ from ]\nu\n[ to ]\nm\n[ martini ]\nBB\n[ atoms ]\n1 N SC1\n";
        assert!(parse_mapping(text).is_err());
    }

    #[test]
    fn parses_residue_topology() {
        let text = "[ moleculetype ]\nALA 1\n[ atoms ]\n1 SP2 1 ALA BB 1 0\n2 TC3 1 ALA SC1 2 0\n[ constraints ]\nBB SC1 1 0.270\n\n[ moleculetype ]\nGLY 1\n[ atoms ]\n1 SP1 1 GLY BB 1 0\n";
        let t = parse_residue_topology(text, "ALA").unwrap();
        assert_eq!(t.beads.len(), 2);
        assert_eq!(t.bonds[0].first, "BB");
    }
}
