//! Shared SE-owned half-neighbor LJ+Coulomb evaluator.

use lin_alg::f32::Vec3;

use super::super::types::{EnergyVirial, ForceBuffer, ForceFieldError, ForceFieldSystem};

/// Evaluate every unexcluded `i < j` pair exactly once. This mirrors the
/// half-neighbor/Newton-pair accounting used by LAMMPS: both forces are
/// updated, while energy and virial are accumulated once.
pub fn evaluate_half_pairs(
    system: &ForceFieldSystem<'_>,
    forces: &mut ForceBuffer,
    coulomb_k: f32,
) -> Result<EnergyVirial, ForceFieldError> {
    system.validate()?;
    if forces.forces.len() != system.atoms.len() {
        return Err(ForceFieldError("force buffer length mismatch".into()));
    }
    forces.clear();
    let mut energy = 0.0_f64;
    let mut virial = 0.0_f64;
    for i in 0..system.atoms.len() {
        for j in (i + 1)..system.atoms.len() {
            if system
                .bonds
                .iter()
                .any(|&(a, b)| (a == i && b == j) || (a == j && b == i))
            {
                continue;
            }
            let a = system.atoms[i];
            let b = system.atoms[j];
            let d = Vec3::new(
                a.position.x - b.position.x,
                a.position.y - b.position.y,
                a.position.z - b.position.z,
            );
            let r2 = d.x * d.x + d.y * d.y + d.z * d.z;
            if r2 <= f32::EPSILON {
                continue;
            }
            let inv_r = r2.sqrt().recip();
            let sigma = 0.5 * (a.sigma + b.sigma);
            let epsilon = (a.epsilon * b.epsilon).max(0.0).sqrt();
            let sr = sigma * inv_r;
            let sr2 = sr * sr;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            let q = a.charge * b.charge;
            let lj_energy = 4.0 * epsilon * (sr12 - sr6);
            let coul_energy = coulomb_k * q * inv_r;
            let magnitude =
                24.0 * epsilon * (2.0 * sr12 - sr6) * inv_r + coulomb_k * q * inv_r * inv_r;
            let f = d * (inv_r * magnitude);
            forces.forces[i] += f;
            forces.forces[j] -= f;
            energy += (lj_energy + coul_energy) as f64;
            virial += (d.x * f.x + d.y * f.y + d.z * f.z) as f64;
        }
    }
    Ok(EnergyVirial { energy, virial })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forcefield::types::ForceAtom;

    #[test]
    fn half_pair_updates_newton_forces_and_counts_energy_once() {
        let atoms = [
            ForceAtom {
                position: Vec3::new(0.0, 0.0, 0.0),
                charge: 0.0,
                sigma: 1.0,
                epsilon: 1.0,
            },
            ForceAtom {
                position: Vec3::new(2.0, 0.0, 0.0),
                charge: 0.0,
                sigma: 1.0,
                epsilon: 1.0,
            },
        ];
        let system = ForceFieldSystem {
            atoms: &atoms,
            bonds: &[],
        };
        let mut forces = ForceBuffer::zeros(2);
        let ev = evaluate_half_pairs(&system, &mut forces, 332.0522).unwrap();
        assert!(ev.energy.is_finite() && ev.virial.is_finite());
        assert!((forces.forces[0].x + forces.forces[1].x).abs() < 1e-6);
        assert!(forces.forces[0].x.abs() > 0.0);

        let bonded = ForceFieldSystem {
            atoms: &atoms,
            bonds: &[(0, 1)],
        };
        let ev_bonded = evaluate_half_pairs(&bonded, &mut forces, 332.0522).unwrap();
        assert_eq!(ev_bonded.energy, 0.0);
        assert_eq!(forces.forces[0], Vec3::new_zero());
        assert_eq!(forces.forces[1], Vec3::new_zero());
    }
}
