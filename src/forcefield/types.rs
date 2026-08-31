//! Stable SE-owned data types shared by force-field implementations.

use lin_alg::f32::Vec3;

/// Spatial resolution in `[0.0, 1.0]`; `1.0` is all-atom.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Resolution(f32);

impl Resolution {
    pub const ALL_ATOM: Self = Self(1.0);

    pub fn new(level: f32) -> Option<Self> {
        level
            .is_finite()
            .then_some(level)
            .filter(|&v| (0.0..=1.0).contains(&v))
            .map(Self)
    }

    pub const fn level(self) -> f32 {
        self.0
    }

    pub fn is_all_atom(self) -> bool {
        self.0 == 1.0
    }
}

impl Default for Resolution {
    fn default() -> Self {
        Self::ALL_ATOM
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForceFieldRegion {
    pub start: usize,
    pub end: usize,
    pub resolution: Resolution,
}

impl ForceFieldRegion {
    pub fn new(start: usize, end: usize, resolution: Resolution) -> Result<Self, ForceFieldError> {
        if start >= end {
            return Err(ForceFieldError(format!(
                "invalid force-field region [{start}, {end})"
            )));
        }
        Ok(Self {
            start,
            end,
            resolution,
        })
    }
}

/// SE-owned atom state exposed to force-field evaluation; no dynamics type leaks
/// through this boundary.
#[derive(Clone, Copy, Debug, Default)]
pub struct ForceAtom {
    pub position: Vec3,
    pub charge: f32,
    pub sigma: f32,
    pub epsilon: f32,
}

pub struct ForceFieldSystem<'a> {
    pub atoms: &'a [ForceAtom],
    pub bonds: &'a [(usize, usize)],
}

impl<'a> ForceFieldSystem<'a> {
    pub fn validate(&self) -> Result<(), ForceFieldError> {
        if self.atoms.iter().any(|a| {
            !a.position.x.is_finite()
                || !a.position.y.is_finite()
                || !a.position.z.is_finite()
                || !a.charge.is_finite()
                || !a.sigma.is_finite()
                || !a.epsilon.is_finite()
        }) {
            return Err(ForceFieldError(
                "system contains non-finite atom data".into(),
            ));
        }
        if self
            .bonds
            .iter()
            .any(|&(i, j)| i >= self.atoms.len() || j >= self.atoms.len() || i == j)
        {
            return Err(ForceFieldError(
                "system contains an invalid bond index".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ForceBuffer {
    pub forces: Vec<Vec3>,
}

impl ForceBuffer {
    pub fn zeros(n_atoms: usize) -> Self {
        Self {
            forces: vec![Vec3::new_zero(); n_atoms],
        }
    }
    pub fn clear(&mut self) {
        self.forces.fill(Vec3::new_zero());
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EnergyVirial {
    pub energy: f64,
    pub virial: f64,
}

#[derive(Debug, Clone)]
pub struct ForceFieldError(pub String);

impl std::fmt::Display for ForceFieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ForceFieldError {}
