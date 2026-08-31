//! Public SE force-field interfaces.

use super::types::{
    EnergyVirial, ForceBuffer, ForceFieldError, ForceFieldRegion, ForceFieldSystem, Resolution,
};

/// A force-field definition. Preparation happens outside the MD hot loop.
pub trait ForceField: Send + Sync {
    type Prepared: PreparedForceField;

    fn name(&self) -> &'static str;

    /// Resolution level used by this force field. All-atom fields return `1`.
    fn resolution(&self) -> Resolution {
        Resolution::ALL_ATOM
    }

    fn prepare(&self) -> Result<Self::Prepared, ForceFieldError>;
}

/// A compiled/indexed force field used during force evaluation.
pub trait PreparedForceField: Send + Sync {
    fn name(&self) -> &'static str;

    fn resolution(&self) -> Resolution {
        Resolution::ALL_ATOM
    }

    /// Evaluate this field over its assigned region(s). Region composition is
    /// metadata-only in v1.1; overlap/cross-resolution coupling is deferred.
    fn regions(&self) -> &[ForceFieldRegion] {
        &[]
    }

    fn evaluate(
        &self,
        system: &ForceFieldSystem<'_>,
        forces: &mut ForceBuffer,
    ) -> Result<EnergyVirial, ForceFieldError>;
}
