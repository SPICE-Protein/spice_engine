//! Environment parameters (the SPICE vector) that condition a simulation.

/// Biologically sensible ranges for the environment parameters. Values outside
/// these are clamped on construction, so a scan / RL cannot drive a system into
/// clearly non-physical territory.
pub mod sane {
    /// Water exists as a liquid roughly 273–373 K; proteins are studied from
    /// cold-adapted (~250 K) to hyperthermophile (~400 K) conditions.
    pub const TEMP_K_MIN: f32 = 250.0;
    pub const TEMP_K_MAX: f32 = 400.0;
    /// pH scale bounds (extremophiles ~0–13; scale is 0–14).
    pub const PH_MIN: f32 = 0.0;
    pub const PH_MAX: f32 = 14.0;
    /// 0 disables the barostat; deep-sea pressures reach ~1000 bar.
    pub const PRESSURE_BAR_MIN: f32 = 0.0;
    pub const PRESSURE_BAR_MAX: f32 = 2000.0;
    /// Physiological ionic strength ~0.15 M; extremes ~1 M.
    pub const IONIC_M_MIN: f32 = 0.0;
    pub const IONIC_M_MAX: f32 = 2.0;
}

/// Environmental conditions for an MD run / episode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvParams {
    /// pH — sets protonation states at system build time.
    pub ph: f32,
    /// Target temperature, Kelvin.
    pub temp_k: f32,
    /// Target pressure, bar. `0.0` disables the barostat.
    pub pressure_bar: f32,
    /// Ionic strength, mol/L (NaCl). `0.0` adds no salt.
    pub ionic_strength_m: f32,
}

impl Default for EnvParams {
    fn default() -> Self {
        Self {
            ph: 7.0,
            temp_k: 310.0,
            pressure_bar: 1.0,
            ionic_strength_m: 0.0,
        }
    }
}

impl EnvParams {
    /// Construct, clamping each field into the biologically sensible ranges in
    /// [`sane`]. Use [`EnvParams::new_raw`] to skip clamping, or call
    /// [`EnvParams::validate`] to check whether any value was out of range.
    pub fn new(ph: f32, temp_k: f32, pressure_bar: f32, ionic_strength_m: f32) -> Self {
        Self {
            ph,
            temp_k,
            pressure_bar,
            ionic_strength_m,
        }
        .clamped()
    }

    /// Construct without range clamping (for callers that pre-validate).
    pub fn new_raw(ph: f32, temp_k: f32, pressure_bar: f32, ionic_strength_m: f32) -> Self {
        Self {
            ph,
            temp_k,
            pressure_bar,
            ionic_strength_m,
        }
    }

    /// Clamp every field into the [`sane`] biological ranges.
    pub fn clamped(mut self) -> Self {
        self.ph = self.ph.clamp(sane::PH_MIN, sane::PH_MAX);
        self.temp_k = self.temp_k.clamp(sane::TEMP_K_MIN, sane::TEMP_K_MAX);
        self.pressure_bar = self.pressure_bar.clamp(sane::PRESSURE_BAR_MIN, sane::PRESSURE_BAR_MAX);
        self.ionic_strength_m =
            self.ionic_strength_m.clamp(sane::IONIC_M_MIN, sane::IONIC_M_MAX);
        self
    }

    /// `true` if all fields are within the [`sane`] biological ranges.
    pub fn is_sane(&self) -> bool {
        *self == self.clamped()
    }

    /// Error listing any field that is outside the [`sane`] biological ranges.
    pub fn validate(&self) -> Result<(), String> {
        let mut bad = Vec::new();
        if !(sane::PH_MIN..=sane::PH_MAX).contains(&self.ph) {
            bad.push(format!("ph={} (range {}-{})", self.ph, sane::PH_MIN, sane::PH_MAX));
        }
        if !(sane::TEMP_K_MIN..=sane::TEMP_K_MAX).contains(&self.temp_k) {
            bad.push(format!(
                "temp_k={} (range {}-{})",
                self.temp_k, sane::TEMP_K_MIN, sane::TEMP_K_MAX
            ));
        }
        if !(sane::PRESSURE_BAR_MIN..=sane::PRESSURE_BAR_MAX).contains(&self.pressure_bar) {
            bad.push(format!(
                "pressure_bar={} (range {}-{})",
                self.pressure_bar, sane::PRESSURE_BAR_MIN, sane::PRESSURE_BAR_MAX
            ));
        }
        if !(sane::IONIC_M_MIN..=sane::IONIC_M_MAX).contains(&self.ionic_strength_m) {
            bad.push(format!(
                "ionic_strength_m={} (range {}-{})",
                self.ionic_strength_m, sane::IONIC_M_MIN, sane::IONIC_M_MAX
            ));
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(format!("EnvParams out of biological range: {}", bad.join(", ")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps_to_sane_ranges() {
        let e = EnvParams::new(-3.0, 500.0, 5000.0, 9.0);
        assert_eq!(e.ph, sane::PH_MIN);
        assert_eq!(e.temp_k, sane::TEMP_K_MAX);
        assert_eq!(e.pressure_bar, sane::PRESSURE_BAR_MAX);
        assert_eq!(e.ionic_strength_m, sane::IONIC_M_MAX);
        assert!(e.is_sane()); // after clamping, the values are all in range

        // validate() reports the *raw* input, so test it via new_raw.
        let raw = EnvParams::new_raw(-3.0, 500.0, 5000.0, 9.0);
        assert!(raw.validate().is_err());
        assert!(!raw.is_sane());
    }

    #[test]
    fn sane_values_pass_through() {
        let e = EnvParams::new(7.0, 310.0, 1.0, 0.15);
        assert!(e.is_sane());
        assert!(e.validate().is_ok());
    }

    #[test]
    fn raw_skips_clamp() {
        let e = EnvParams::new_raw(15.0, 100.0, 1.0, 0.0);
        assert_eq!(e.ph, 15.0);
        assert!(!e.is_sane());
    }
}

