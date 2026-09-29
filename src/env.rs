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
    /// Divalent salts (MgCl2/CaCl2 formula units) are used at much lower
    /// loading: physiological Mg²⁺ is tens of mM, RNA-titration experiments
    /// reach a few hundred mM. 1 M would be unphysical crowding.
    pub const DIVALENT_M_MIN: f32 = 0.0;
    pub const DIVALENT_M_MAX: f32 = 1.0;
    /// Max field amplitude component, kcal·mol⁻¹·e⁻¹·Å⁻¹ (≈ 21.6 V/nm —
    /// beyond any molecular effect we scan; electroporation ~0.5 V/nm).
    pub const EFIELD_MAX: f32 = 5.0;
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
    /// **NaCl background molarity**, mol/L (neutralizing counterions are separate).
    /// `0.0` adds no salt. This sets how many Na⁺/Cl⁻ PAIRS to insert; it is NOT
    /// the total ionic strength once divalent salts or `salts_json` electrolytes
    /// are added — for the actual I = ½Σcᵢzᵢ² of the built box use
    /// `MdState::effective_ionic_strength_m()`. (For pure NaCl the two coincide.)
    pub ionic_strength_m: f32,
    /// MgCl2 formula-unit molarity, mol/L. Inserts Mg²⁺ (Li-Merz 12-6-4
    /// OPC) with two Cl⁻ companions per unit, displacing waters genion-wise.
    pub mg_cl2_m: f32,
    /// CaCl2 formula-unit molarity, mol/L. See [`EnvParams::mg_cl2_m`].
    pub ca_cl2_m: f32,
    /// SrCl2 formula-unit molarity, mol/L (Li-Merz 12-6-4 OPC; the ion
    /// carries `Element::Other` upstream — identity is its ff_type "Sr2+").
    pub sr_cl2_m: f32,
    /// BaCl2 formula-unit molarity, mol/L. See [`EnvParams::mg_cl2_m`].
    pub ba_cl2_m: f32,
    /// Fraction ∈ [0,1] of detected disulfide bridges to REDUCE at build
    /// time (0 = fully oxidized = historical). Deterministic bridge selection;
    /// see `find_disulfide_sgs`.
    pub redox_reducing: f32,
    /// Static/oscillating external electric field amplitude per component,
    /// kcal·mol⁻¹·e⁻¹·Å⁻¹ (≈ 4.33×(V/nm): 1 V/nm ≈ 0.231). Applied to all
    /// charges incl. rigid-water sites; zero disables the term entirely.
    pub efield: [f32; 3],
    /// Oscillation frequency ω in rad/ps: E(t) = efield·cos(ωt). 0 = static.
    pub efield_omega: f32,
}

impl Default for EnvParams {
    fn default() -> Self {
        Self {
            ph: 7.0,
            temp_k: 310.0,
            pressure_bar: 1.0,
            ionic_strength_m: 0.0,
            mg_cl2_m: 0.0,
            ca_cl2_m: 0.0,
            sr_cl2_m: 0.0,
            ba_cl2_m: 0.0,
            redox_reducing: 0.0,
            efield: [0.0; 3],
            efield_omega: 0.0,
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
            mg_cl2_m: 0.0,
            ca_cl2_m: 0.0,
            sr_cl2_m: 0.0,
            ba_cl2_m: 0.0,
            redox_reducing: 0.0,
            efield: [0.0; 3],
            efield_omega: 0.0,
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
            mg_cl2_m: 0.0,
            ca_cl2_m: 0.0,
            sr_cl2_m: 0.0,
            ba_cl2_m: 0.0,
            redox_reducing: 0.0,
            efield: [0.0; 3],
            efield_omega: 0.0,
        }
    }

    /// Set all four divalent salt loadings (builder style; clamped like `new`).
    #[must_use]
    pub fn with_divalent(mut self, mg: f32, ca: f32, sr: f32, ba: f32) -> Self {
        let c = |v: f32| v.clamp(sane::DIVALENT_M_MIN, sane::DIVALENT_M_MAX);
        self.mg_cl2_m = c(mg);
        self.ca_cl2_m = c(ca);
        self.sr_cl2_m = c(sr);
        self.ba_cl2_m = c(ba);
        self
    }

    /// Set the disulfide-reducing fraction (builder style; clamped to [0,1]).
    #[must_use]
    pub fn with_redox(mut self, reducing_fraction: f32) -> Self {
        self.redox_reducing = reducing_fraction.clamp(0.0, 1.0);
        self
    }

    /// Set the electric field (builder style; each component clamped to a
    /// physically sane ±5 kcal·mol⁻¹·e⁻¹·Å⁻¹ ≈ 21.6 V/nm).
    #[must_use]
    pub fn with_efield(mut self, efield: [f32; 3], omega_rad_ps: f32) -> Self {
        self.efield = efield.map(|v| v.clamp(-sane::EFIELD_MAX, sane::EFIELD_MAX));
        self.efield_omega = omega_rad_ps.max(0.0);
        self
    }

    /// Clamp every field into the [`sane`] biological ranges.
    pub fn clamped(mut self) -> Self {
        self.ph = self.ph.clamp(sane::PH_MIN, sane::PH_MAX);
        self.temp_k = self.temp_k.clamp(sane::TEMP_K_MIN, sane::TEMP_K_MAX);
        self.pressure_bar = self
            .pressure_bar
            .clamp(sane::PRESSURE_BAR_MIN, sane::PRESSURE_BAR_MAX);
        self.ionic_strength_m = self
            .ionic_strength_m
            .clamp(sane::IONIC_M_MIN, sane::IONIC_M_MAX);
        self.mg_cl2_m = self
            .mg_cl2_m
            .clamp(sane::DIVALENT_M_MIN, sane::DIVALENT_M_MAX);
        self.ca_cl2_m = self
            .ca_cl2_m
            .clamp(sane::DIVALENT_M_MIN, sane::DIVALENT_M_MAX);
        self.sr_cl2_m = self
            .sr_cl2_m
            .clamp(sane::DIVALENT_M_MIN, sane::DIVALENT_M_MAX);
        self.ba_cl2_m = self
            .ba_cl2_m
            .clamp(sane::DIVALENT_M_MIN, sane::DIVALENT_M_MAX);
        self.redox_reducing = self.redox_reducing.clamp(0.0, 1.0);
        self.efield = self
            .efield
            .map(|v| v.clamp(-sane::EFIELD_MAX, sane::EFIELD_MAX));
        self.efield_omega = self.efield_omega.max(0.0);
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
            bad.push(format!(
                "ph={} (range {}-{})",
                self.ph,
                sane::PH_MIN,
                sane::PH_MAX
            ));
        }
        if !(sane::TEMP_K_MIN..=sane::TEMP_K_MAX).contains(&self.temp_k) {
            bad.push(format!(
                "temp_k={} (range {}-{})",
                self.temp_k,
                sane::TEMP_K_MIN,
                sane::TEMP_K_MAX
            ));
        }
        if !(sane::PRESSURE_BAR_MIN..=sane::PRESSURE_BAR_MAX).contains(&self.pressure_bar) {
            bad.push(format!(
                "pressure_bar={} (range {}-{})",
                self.pressure_bar,
                sane::PRESSURE_BAR_MIN,
                sane::PRESSURE_BAR_MAX
            ));
        }
        if !(sane::IONIC_M_MIN..=sane::IONIC_M_MAX).contains(&self.ionic_strength_m) {
            bad.push(format!(
                "ionic_strength_m={} (range {}-{})",
                self.ionic_strength_m,
                sane::IONIC_M_MIN,
                sane::IONIC_M_MAX
            ));
        }
        for (name, v) in [
            ("mg_cl2_m", self.mg_cl2_m),
            ("ca_cl2_m", self.ca_cl2_m),
            ("sr_cl2_m", self.sr_cl2_m),
            ("ba_cl2_m", self.ba_cl2_m),
            ("redox_reducing", self.redox_reducing),
        ] {
            // efield components validated separately below (array).
            if !(sane::DIVALENT_M_MIN..=sane::DIVALENT_M_MAX).contains(&v) {
                bad.push(format!(
                    "{name}={v} (range {}-{})",
                    sane::DIVALENT_M_MIN,
                    sane::DIVALENT_M_MAX
                ));
            }
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "EnvParams out of biological range: {}",
                bad.join(", ")
            ))
        }
    }
}

/// Full user-editable build recipe — the shape behind the `se.Env` Python
/// object (v1.3.7). Distinct from [`EnvParams`] (the physical vector): it also
/// carries the build-tail knobs (`relax_iters`, `tolerance`,
/// `strict_incomplete`) and the salts/cosolvents JSON channels, so ONE object
/// can be passed to `Engine.build(structure, env=...)` and
/// `Engine.mutate_with_solvent_reuse(..., env=...)` — the two calls can no
/// longer drift, because there is nothing to re-type.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvSpec {
    pub ph: f32,
    pub temp_k: f32,
    pub pressure_bar: f32,
    pub ionic_strength_m: f32,
    /// Iteration cap of the build-tail L-BFGS minimization (BuildOptions
    /// default 2000).
    pub relax_iters: usize,
    /// Minimization convergence tolerance (kcal/mol gradient scale).
    pub tolerance: f32,
    /// Reject residues missing charge-lib sidechain heavy atoms (strict) vs
    /// build with the atoms present (lenient).
    pub strict_incomplete: bool,
    pub mg_cl2_m: f32,
    pub ca_cl2_m: f32,
    pub sr_cl2_m: f32,
    pub ba_cl2_m: f32,
    pub redox_reducing: f32,
    pub cosolvents_json: String,
    pub salts_json: String,
    /// Solvent box padding (Å) around the solute. `0.0` keeps the builder
    /// default (10 Å, sized for a FOLDED state). PFDE-style denaturing runs
    /// need a much larger box: an unfolded chain whose contour exceeds the
    /// native-sized cell threads through the periodic images and can neither
    /// extend honestly nor collapse back (measured on 1L2Y: Rg pinned at
    /// ~8 A by a 43 A box at 505 K, then re-inflated to "30 A" by image
    /// wrapping at 550-600 K). Pass e.g. 30 for a trp-cage denature/refold.
    pub box_pad_a: f32,
}

impl Default for EnvSpec {
    fn default() -> Self {
        Self {
            ph: 7.0,
            temp_k: 310.0,
            pressure_bar: 1.0,
            ionic_strength_m: 0.0,
            // 2000: BuildOptions default (builder.rs); tolerance 2.0 likewise.
            relax_iters: 2000,
            tolerance: 2.0,
            strict_incomplete: true,
            mg_cl2_m: 0.0,
            ca_cl2_m: 0.0,
            sr_cl2_m: 0.0,
            ba_cl2_m: 0.0,
            redox_reducing: 0.0,
            cosolvents_json: String::new(),
            salts_json: String::new(),
            box_pad_a: 0.0,
        }
    }
}

impl EnvSpec {
    /// Project the physical environment vector (clamped like the FFI's
    /// historical scalar path — same `EnvParams::new` + `with_*` chain).
    pub fn env_params(&self) -> EnvParams {
        EnvParams::new(
            self.ph,
            self.temp_k,
            self.pressure_bar,
            self.ionic_strength_m,
        )
        .with_divalent(self.mg_cl2_m, self.ca_cl2_m, self.sr_cl2_m, self.ba_cl2_m)
        .with_redox(self.redox_reducing)
    }
}

/// The FFI's flattened optional scalars: every one of `Engine.build` /
/// `mutate_with_solvent_reuse`'s post-`structure` parameters, `None` meaning
/// "not passed". `None` defaults keep the 7 positional slots of the legacy
/// signature intact for callers that DO pass them (spice_rl, ~25 diagnostic
/// scripts) while allowing `env=` to carry the whole block.
#[derive(Clone, Debug, Default)]
pub struct BuildScalars {
    pub ph: Option<f32>,
    pub temp: Option<f32>,
    pub pressure: Option<f32>,
    pub ionic_strength_m: Option<f32>,
    pub relax_iters: Option<usize>,
    pub tolerance: Option<f32>,
    pub strict_incomplete: Option<bool>,
    pub mg_molar: Option<f32>,
    pub ca_molar: Option<f32>,
    pub sr_molar: Option<f32>,
    pub ba_molar: Option<f32>,
    pub redox_reducing: Option<f32>,
    pub cosolvents_json: Option<String>,
    pub salts_json: Option<String>,
}

/// A fully resolved build recipe (the FFI hands this to `BuildOptions`).
#[derive(Clone, Debug)]
pub struct ResolvedBuild {
    pub env: EnvParams,
    pub relax_iters: usize,
    pub tolerance: f32,
    pub strict_incomplete: bool,
    pub cosolvents_json: String,
    pub salts_json: String,
    /// Env-only geometry knob (0 = BuildOptions default). Never a trailing
    /// scalar, so it needs no collision/deprecation plumbing.
    pub box_pad_a: f32,
    /// Trailing scalars the caller passed EXPLICITLY (v1.3.7: these earn a
    /// DeprecationWarning at the FFI — move into `se.Env`). `strict_incomplete`
    /// is NOT on this list: it is actively used (spice_rl) and stays
    /// first-class. The six core scalars likewise stay un-warned.
    pub deprecated_used: Vec<&'static str>,
}

/// Resolve the "scalars OR Env" duality into one complete recipe. Rules
/// (FFI doc mirrors this):
/// - `env = Some`: the Env is the whole story; any additionally-passed scalar
///   is a collision error (prevents silent override ambiguity).
/// - `env = None`: the six core values (ph..tolerance) must ALL be present —
///   identical to the pre-1.3.7 required-args behavior. Trailing values fall
///   back to `EnvSpec` defaults. Explicit trailing scalars are recorded in
///   `deprecated_used`.
pub fn resolve_build_args(
    s: &BuildScalars,
    env: Option<&EnvSpec>,
) -> Result<ResolvedBuild, String> {
    let given = [
        ("ph", s.ph.is_some()),
        ("temp", s.temp.is_some()),
        ("pressure", s.pressure.is_some()),
        ("ionic_strength_m", s.ionic_strength_m.is_some()),
        ("relax_iters", s.relax_iters.is_some()),
        ("tolerance", s.tolerance.is_some()),
        ("strict_incomplete", s.strict_incomplete.is_some()),
        ("mg_molar", s.mg_molar.is_some()),
        ("ca_molar", s.ca_molar.is_some()),
        ("sr_molar", s.sr_molar.is_some()),
        ("ba_molar", s.ba_molar.is_some()),
        ("redox_reducing", s.redox_reducing.is_some()),
        ("cosolvents_json", s.cosolvents_json.is_some()),
        ("salts_json", s.salts_json.is_some()),
    ];
    if let Some(sp) = env {
        let extra: Vec<&str> = given.iter().filter(|(_, v)| *v).map(|(n, _)| *n).collect();
        if !extra.is_empty() {
            return Err(format!(
                "pass either env=se.Env(...) or the individual scalars, not both — \
                 also given: {}; build your Env once and mutate its fields (env.temp_k = ...) \
                 to vary one knob",
                extra.join(", ")
            ));
        }
        return Ok(ResolvedBuild {
            env: sp.env_params(),
            relax_iters: sp.relax_iters,
            tolerance: sp.tolerance,
            strict_incomplete: sp.strict_incomplete,
            cosolvents_json: sp.cosolvents_json.clone(),
            salts_json: sp.salts_json.clone(),
            box_pad_a: sp.box_pad_a,
            deprecated_used: Vec::new(),
        });
    }
    let core_missing: Vec<&str> = given[..6]
        .iter()
        .filter(|(_, v)| !*v)
        .map(|(n, _)| *n)
        .collect();
    if !core_missing.is_empty() {
        return Err(format!(
            "missing required environment values: {} — pass them positionally/keyword \
             or provide env=se.Env(...)",
            core_missing.join(", ")
        ));
    }
    let mut base = EnvSpec {
        ph: s.ph.unwrap(),
        temp_k: s.temp.unwrap(),
        pressure_bar: s.pressure.unwrap(),
        ionic_strength_m: s.ionic_strength_m.unwrap(),
        relax_iters: s.relax_iters.unwrap(),
        tolerance: s.tolerance.unwrap(),
        strict_incomplete: s.strict_incomplete.unwrap_or(true),
        ..Default::default()
    };
    let mut deprecated_used = Vec::new();
    let mut take_deprecated = |slot: &mut f32, given: Option<f32>, name: &'static str| {
        if let Some(v) = given {
            *slot = v;
            deprecated_used.push(name);
        }
    };
    take_deprecated(&mut base.mg_cl2_m, s.mg_molar, "mg_molar");
    take_deprecated(&mut base.ca_cl2_m, s.ca_molar, "ca_molar");
    take_deprecated(&mut base.sr_cl2_m, s.sr_molar, "sr_molar");
    take_deprecated(&mut base.ba_cl2_m, s.ba_molar, "ba_molar");
    take_deprecated(&mut base.redox_reducing, s.redox_reducing, "redox_reducing");
    if let Some(j) = &s.cosolvents_json {
        if !j.is_empty() {
            deprecated_used.push("cosolvents_json");
        }
        base.cosolvents_json = j.clone();
    }
    if let Some(j) = &s.salts_json {
        if !j.is_empty() {
            deprecated_used.push("salts_json");
        }
        base.salts_json = j.clone();
    }
    Ok(ResolvedBuild {
        env: base.env_params(),
        relax_iters: base.relax_iters,
        tolerance: base.tolerance,
        strict_incomplete: base.strict_incomplete,
        cosolvents_json: base.cosolvents_json,
        salts_json: base.salts_json,
        box_pad_a: base.box_pad_a,
        deprecated_used,
    })
}

#[cfg(test)]
mod resolve_tests {
    use super::*;

    #[test]
    fn legacy_scalars_resolve_exactly_as_before() {
        // What spice_rl passes positionally: the six core.
        let s = BuildScalars {
            ph: Some(7.0),
            temp: Some(310.0),
            pressure: Some(1.0),
            ionic_strength_m: Some(0.15),
            relax_iters: Some(2000),
            tolerance: Some(2.0),
            ..Default::default()
        };
        let r = resolve_build_args(&s, None).unwrap();
        assert_eq!(r.env, EnvParams::new(7.0, 310.0, 1.0, 0.15));
        assert_eq!(r.relax_iters, 2000);
        assert_eq!(r.tolerance, 2.0);
        assert!(r.strict_incomplete); // default unchanged
        assert!(r.cosolvents_json.is_empty() && r.salts_json.is_empty());
        assert!(r.deprecated_used.is_empty());
        // Env-only geometry knob stays at the sentinel (builder default) for
        // every legacy positional caller (spice_rl golden path).
        assert_eq!(r.box_pad_a, 0.0);
    }

    #[test]
    fn box_pad_a_flows_only_through_env() {
        let e = EnvSpec {
            ph: 7.0,
            temp_k: 300.0,
            box_pad_a: 30.0,
            ..Default::default()
        };
        let r = resolve_build_args(&BuildScalars::default(), Some(&e)).unwrap();
        assert_eq!(r.box_pad_a, 30.0);
        // The DEFAULT Env (box_pad_a = 0) must also resolve to the sentinel,
        // so se.Env(...) callers without an explicit box keep 10 A padding.
        let r0 = resolve_build_args(&BuildScalars::default(), Some(&EnvSpec::default())).unwrap();
        assert_eq!(r0.box_pad_a, 0.0);
    }

    #[test]
    fn trailing_scalars_resolve_but_earn_deprecation() {
        let s = BuildScalars {
            ph: Some(7.0),
            temp: Some(310.0),
            pressure: Some(0.0),
            ionic_strength_m: Some(0.0),
            relax_iters: Some(2000),
            tolerance: Some(2.0),
            mg_molar: Some(0.05),
            salts_json: Some("[]".to_string()),
            // strict_incomplete explicitly given: first-class, NOT deprecated.
            strict_incomplete: Some(false),
            ..Default::default()
        };
        let r = resolve_build_args(&s, None).unwrap();
        assert_eq!(r.env.mg_cl2_m, 0.05);
        assert!(!r.strict_incomplete);
        assert_eq!(r.deprecated_used, vec!["mg_molar", "salts_json"]);
        // An explicitly-passed EMPTY json is still legacy use — resolve keeps
        // it (no-op downstream) but does not warn (empty == default content).
    }

    #[test]
    fn env_only_and_conflict_and_missing() {
        let e = EnvSpec {
            ph: 8.0,
            temp_k: 320.0,
            cosolvents_json: "[{\"name\":\"UREA\",\"molarity\":0.5}]".to_string(),
            ..Default::default()
        };
        let r = resolve_build_args(&BuildScalars::default(), Some(&e)).unwrap();
        assert_eq!(r.env.ph, 8.0);
        assert_eq!(r.relax_iters, 2000);
        assert!(r.cosolvents_json.contains("UREA"));
        // env + any scalar → conflict listing BOTH core and trailing names.
        let mixed = BuildScalars {
            ph: Some(7.0),
            ca_molar: Some(0.01),
            ..Default::default()
        };
        let err = resolve_build_args(&mixed, Some(&e)).unwrap_err();
        assert!(err.contains("not both") && err.contains("ph") && err.contains("ca_molar"));
        // env=None with missing core → error points at env=.
        let missing = resolve_build_args(&BuildScalars::default(), None).unwrap_err();
        assert!(missing.contains("ph") && missing.contains("se.Env"));
    }

    #[test]
    fn envspec_projection_clamps_like_legacy_path() {
        let e = EnvSpec {
            ph: 99.0,
            temp_k: 0.0,
            mg_cl2_m: 5.0,
            ..Default::default()
        };
        let p = e.env_params();
        assert_eq!(p.ph, sane::PH_MAX);
        assert_eq!(p.temp_k, sane::TEMP_K_MIN);
        assert_eq!(p.mg_cl2_m, sane::DIVALENT_M_MAX);
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
