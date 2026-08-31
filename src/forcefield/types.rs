//! Stable data types shared by SE force-field implementations.

use dynamics::AtomDynamics;
use lin_alg::f32::Vec3;

/// Spatial resolution level of a force-field representation.
///
/// Normalized spatial resolution in `[0.0, 1.0]`.
///
/// `1.0` is the all-atom representation. Coarser representations are closer
/// to `0.0`; `0.0` is allowed as the limiting coarse-grained value. Resolution
/// is metadata only and is not interpreted by the current evaluator yet.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Resolution(f32);

impl Resolution {
    pub const ALL_ATOM: Self = Self(1.0);

    pub fn new(level: f32) -> Option<Self> {
        if level.is_finite() && (0.0..=1.0).contains(&level) {
            Some(Self(level))
        } else {
            None
        }
    }

    /// Return the normalized resolution in `[0.0, 1.0]`.
    pub const fn level(self) -> f32 {
        self.0
    }

    pub fn is_all_atom(self) -> bool {
        self.0 == Self::ALL_ATOM.0
    }
}

impl Default for Resolution {
    fn default() -> Self {
        Self::ALL_ATOM
    }
}

/// Region assignment used to describe a future mixed-resolution system.
/// Ranges are half-open atom-index intervals: `[start, end)`.
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

/// Borrowed system view passed to force-field preparation/evaluation.
/// This is intentionally temporary; `AtomDynamics` will be replaced by an SE-owned type later.
pub struct ForceFieldSystem<'a> {
    pub atoms: &'a [AtomDynamics],
    pub bonds: &'a [(usize, usize)],
}

/// Per-atom force accumulator, separate from `dynamics::AtomDynamics`.
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
