//! The high-level SPICE MD engine: wraps `crate::engine::md_core::MdState` with topology + env.

use std::collections::VecDeque;

use crate::engine::md_core::{ComputationDevice, MdState};
use lin_alg::f32::Vec3;

#[path = "engine/core/mod.rs"]
pub mod md_core;

use crate::env::EnvParams;
use crate::forcefield::{
    ComputationContent, ForceAtom, ForceBuffer, ForceFieldSelection, ForceFieldSystem,
};
use crate::topology::ProteinTopology;

/// Potential energy (kcal/mol) above which the system is treated as blown up.
const CRASH_ENERGY_KCAL: f64 = 1.0e8;

/// Maximum number of potential-energy samples kept for the `m1` variance metric.
const U_HISTORY_CAP: usize = 512;

/// Soft trend detector for LONG-MD environment validation (v2 protocol) and
/// RL-side fail-fast (v1.3.4). Moved from `domain.rs` onto the engine so every
/// step driver (FFI `step`/`step_md`/`step_fast`, `EngineWorker`, `equilibrate`,
/// scan loops) shares ONE early-abort path instead of each reimplementing a
/// termination check.
///
/// Short-MD asks "does the structure fold?"; long-MD asks "does the environment
/// let the folded structure STAY folded?". Instead of predicting a crash, we
/// detect whether the environment is driving the system in a bad direction —
/// potential energy rising, Rg expanding, secondary structure dissolving — via
/// the slope of a sliding window, z-scored against a thermal-noise floor. A
/// segment terminates when ≥2 of the 3 signals are significant (scoring, not
/// strict AND, so a single noisy metric cannot lock out a true termination).
#[derive(Debug, Clone)]
pub struct TrendConfig {
    /// Master switch.
    pub enabled: bool,
    /// Observations needed before a slope is fit (sliding window).
    pub window: usize,
    /// Steps between observations.
    pub check_every: usize,
    /// A signal counts as "bad" when its per-ps slope exceeds `z_threshold ×
    /// floor` (a z-score of `z_threshold` or more against the noise floor).
    pub z_threshold: f64,
    /// Thermal-noise floors: per-ps slope of each signal on a well-behaved
    /// reference run — the per-system part of the z-score. Calibrate these on
    /// the anchor reference. Units: kcal/mol/ps, Å/ps, fraction-of-ref/ps.
    pub energy_floor_ps: f64,
    pub rg_floor_ps: f64,
    pub ss_floor_ps: f64,
}

impl Default for TrendConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            window: 100,
            check_every: 10,
            z_threshold: 3.0,
            // Placeholders — calibrate on the reference run before trusting
            // long-MD verdicts.
            energy_floor_ps: 50.0,
            rg_floor_ps: 0.1,
            ss_floor_ps: 0.01,
        }
    }
}

impl TrendConfig {
    /// A tighter, LOUDER-tuned config for RL episodes (seconds, not ns). The
    /// window fills after `60 × 5 = 300` steps (0.6 ps at dt 2 fs) so a
    /// diverging rollout is killed before it reaches numerical blow-up
    /// (measured: the NPT hot-start detonation NaNs at step ~620–930).
    ///
    /// Because 0.6 ps is far shorter than the ns-scale scan window the default
    /// thresholds were eyeballed against, the OLS slope ESTIMATE is
    /// correspondingly noisier, so EVERY threshold is raised ~3× over the
    /// defaults (z 3→8, floors up). This is justified by headroom, not guess:
    /// the real divergence signals are orders of magnitude above these floors
    /// (e.g. the detonation ramps U ~1e8 kcal/ps vs the 400 kcal/ps alarm),
    /// while a healthy 324 K episode's thermal breathing sits far below. An
    /// empirical WT→WT mutant at 324 K false-tripped `ss_loss` under the old
    /// z=3/default-floor preset, so the tighter window REQUIRES the louder
    /// gate. The ≥2-of-3 vote is the primary safeguard (energy alone, even
    /// spiky, cannot terminate an episode).
    pub fn rl_fail_fast() -> Self {
        Self {
            enabled: true,
            window: 60,
            check_every: 5,
            z_threshold: 8.0,
            energy_floor_ps: 150.0,
            rg_floor_ps: 0.3,
            ss_floor_ps: 0.04,
        }
    }
}

/// Sliding-window trend state machine (see [`TrendConfig`]). Pure data + one
/// `observe` call per sample; knows nothing about where signals come from.
pub struct TrendDetector {
    cfg: TrendConfig,
    t_ps: VecDeque<f64>,
    energy: VecDeque<f64>,
    rg: VecDeque<f64>,
    ss: VecDeque<f64>,
}

impl Clone for TrendDetector {
    fn clone(&self) -> Self {
        Self {
            cfg: self.cfg.clone(),
            t_ps: self.t_ps.clone(),
            energy: self.energy.clone(),
            rg: self.rg.clone(),
            ss: self.ss.clone(),
        }
    }
}

impl TrendDetector {
    pub fn new(cfg: TrendConfig) -> Self {
        Self {
            cfg,
            t_ps: VecDeque::new(),
            energy: VecDeque::new(),
            rg: VecDeque::new(),
            ss: VecDeque::new(),
        }
    }

    /// Drop all window state (e.g. episode restart) but keep the config.
    pub fn reset(&mut self) {
        self.t_ps.clear();
        self.energy.clear();
        self.rg.clear();
        self.ss.clear();
    }

    pub fn cfg(&self) -> &TrendConfig {
        &self.cfg
    }

    /// Record one observation (time in ps, energy in kcal/mol, Rg in Å, and the
    /// fraction of reference secondary structure still present). Returns the
    /// triggering signal name once ≥2/3 trend slopes are significant in the bad
    /// direction (energy rising / Rg expanding / SS dissolving).
    pub fn observe(
        &mut self,
        t_ps: f64,
        energy: f64,
        rg: f64,
        ss_frac: f64,
    ) -> Option<&'static str> {
        if !self.cfg.enabled {
            return None;
        }
        let w = self.cfg.window.max(3);
        self.t_ps.push_back(t_ps);
        self.energy.push_back(energy);
        self.rg.push_back(rg);
        self.ss.push_back(ss_frac);
        while self.energy.len() > w {
            self.t_ps.pop_front();
            self.energy.pop_front();
            self.rg.pop_front();
            self.ss.pop_front();
        }
        if self.energy.len() < w {
            return None; // window not full yet — inert on short scans
        }

        let e_slope = slope_ps(&self.t_ps, &self.energy);
        let r_slope = slope_ps(&self.t_ps, &self.rg);
        let s_slope = slope_ps(&self.t_ps, &self.ss);

        let mut bad = 0u32;
        let mut reason = "trend";
        if e_slope > self.cfg.z_threshold * self.cfg.energy_floor_ps {
            bad += 1;
            reason = "energy_rise";
        }
        if r_slope > self.cfg.z_threshold * self.cfg.rg_floor_ps {
            bad += 1;
            reason = "rg_expand";
        }
        // SS dissolving = negative slope of the kept-SS fraction.
        if -s_slope > self.cfg.z_threshold * self.cfg.ss_floor_ps {
            bad += 1;
            reason = "ss_loss";
        }
        if bad >= 2 { Some(reason) } else { None }
    }
}

/// Least-squares slope of `y` vs time `t` (ps) over the window, in y-units/ps.
fn slope_ps(t: &VecDeque<f64>, y: &VecDeque<f64>) -> f64 {
    let n = t.len();
    if n < 2 {
        return 0.0;
    }
    let t_mean: f64 = t.iter().sum::<f64>() / n as f64;
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for i in 0..n {
        let dt = t[i] - t_mean;
        num += dt * (y[i] - y_mean);
        den += dt * dt;
    }
    if den.abs() < 1e-12 { 0.0 } else { num / den }
}

/// Engine-attached trend monitor: the [`TrendDetector`] window plus the
/// reference signals it needs — the build-time backbone H-bond network
/// (`ss_ref`, from the same DSSP-lite proxy as `m3`) sampled only on
/// `check_every`-step gates (both signals are O(solute), not O(N²)).
#[derive(Clone)]
pub struct TrendMonitor {
    det: TrendDetector,
    ss_ref: Vec<(usize, usize)>,
    hbond_n_o: f64,
    /// Steps observed since arming (window/phase counter; independent of
    /// `state.step_count`, which restarts can move in either direction).
    steps_since_arm: usize,
    /// Sampling is inert while `steps_since_arm <= skip_steps` — lets a scan
    /// re-arm per segment without observing its equilibration prefix.
    skip_steps: usize,
}

impl TrendMonitor {
    fn new(cfg: TrendConfig, ss_ref: Vec<(usize, usize)>, hbond_n_o: f64) -> Self {
        Self {
            det: TrendDetector::new(cfg),
            ss_ref,
            hbond_n_o,
            steps_since_arm: 0,
            skip_steps: 0,
        }
    }

    fn arm(&mut self, skip_steps: usize) {
        self.det.reset();
        self.steps_since_arm = 0;
        self.skip_steps = skip_steps;
    }
}

/// Result of one integration step.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Instantaneous potential energy, kcal/mol (dynamics native units).
    pub u_t_kcal: f64,
    /// Instantaneous potential energy, kJ/mol.
    pub u_t_kj: f64,
    /// Cα coordinates, `[L, 3]` Å, aligned with `topology.residues`.
    pub coords_ca: Vec<[f32; 3]>,
    pub step_count: usize,
    /// Simulation time, ps.
    pub time_ps: f64,
    pub crashed: bool,
    /// Reason of the crash, if crashed. When an attached trend monitor fires,
    /// `crashed` is also true and the reason carries the
    /// `trend_alarm:<signal>` prefix — RL loops that treat `crashed` as
    /// episode death get the soft early-abort fail-fast for free, and callers
    /// that care can distinguish a real numerical blow-up from a trend kill.
    pub crash_reason: Option<String>,
    /// Structured trend verdict: `Some(signal)` when the engine-attached
    /// monitor fired on this step (`energy_rise` / `rg_expand` / `ss_loss`).
    /// Implies `crashed == true`. `None` when no monitor is attached, the
    /// window is not full, or the step ended for any other reason.
    pub trend_alarm: Option<&'static str>,
}

/// SPICE's MD engine: one protein + solvent system plus its topology and env.
///
/// `Clone` gives each environment point a fully independent copy of a *pristine*
/// (built + minimized, never-run) system — used by the stability-domain scans so
/// that no point carries state over from another point's simulation.
#[derive(Clone)]
pub struct SpiceEngine {
    pub state: MdState,
    pub topology: ProteinTopology,
    pub env: EnvParams,
    pub dev: ComputationDevice,
    pub dt_ps: f32,
    /// SE-selected calculation content and existing force-field family.
    pub computation_content: ComputationContent,
    pub force_field: ForceFieldSelection,
    /// Recent potential-energy samples (kcal/mol), used by the `m1` metric.
    pub u_history: VecDeque<f64>,
    /// Optional fail-fast trend monitor (see [`TrendConfig`] and
    /// [`SpiceEngine::set_trend_monitor`]). `None` (the default, and what
    /// every build produces) keeps the step path bit-identical.
    pub trend: Option<TrendMonitor>,
    /// Running sum of Cα coordinates (Å) for time-averaged pseudo-labels.
    pub(crate) ca_acc: Vec<[f64; 3]>,
    /// Number of frames accumulated into `ca_acc`.
    pub(crate) ca_n: usize,
}

impl SpiceEngine {
    /// Advance one integration step. `external_force` is per-atom (indexed by
    /// `state.atoms` order) — the hook for SAC bias forces.
    pub fn step(&mut self, external_force: Option<Vec<Vec3>>) -> StepResult {
        self.step_borrowed(external_force.as_deref())
    }

    /// Borrowing external-force path used by RL to avoid cloning the full
    /// all-atom force vector on every action step.
    pub fn step_borrowed(&mut self, external_force: Option<&[Vec3]>) -> StepResult {
        self.state
            .step_with_external_force(&self.dev, self.dt_ps, external_force);

        let u_kcal = self.state.potential_energy;
        let mut crashed = !u_kcal.is_finite() || u_kcal > CRASH_ENERGY_KCAL;

        let mut crash_reason = None;
        if crashed {
            if !u_kcal.is_finite() {
                crash_reason = Some("potential_energy_nan".to_string());
                // Find first atom with non-finite position
                for (i, a) in self.state.atoms.iter().enumerate() {
                    if !a.posit.x.is_finite() || !a.posit.y.is_finite() || !a.posit.z.is_finite() {
                        let res_desc = self
                            .topology
                            .residues
                            .iter()
                            .find(|r| r.atom_indices.contains(&i))
                            .map(|r| format!("{} (seq_id: {})", r.one_letter, r.seq_id))
                            .unwrap_or_else(|| "unknown_residue".to_string());
                        crash_reason = Some(format!(
                            "nan_coordinates_at_atom_index_{}_in_residue_{}",
                            i, res_desc
                        ));
                        break;
                    }
                }
            } else if u_kcal > CRASH_ENERGY_KCAL {
                crash_reason = Some(format!(
                    "potential_energy_spike_exceeded_crash_threshold_{:.2e}_kcal",
                    u_kcal
                ));
            }
        }

        // RL-side fail-fast: the engine-attached trend monitor (opt-in via
        // `set_trend_monitor`) samples cheap signals on its own cadence and
        // terminates BEFORE the hard blow-up above can happen. Signals are
        // energy (free — every step), Rg and the kept-fraction of the
        // build-time backbone H-bond network (both O(solute), sampled only on
        // `check_every` gates; NOT the O(N²) m4/m5 metrics). A fired monitor
        // sets `crashed` + `trend_alarm` with a `trend_alarm:` reason so every
        // existing loop that treats `crashed` as episode death stops early with
        // no code changes, while reward-side code can read the structured
        // `trend_alarm` instead of parsing strings. Hard-crash always wins on
        // the same step (its diagnosis is the precise one).
        let mut trend_alarm: Option<&'static str> = None;
        if !crashed && u_kcal.is_finite() {
            let gate = match &self.trend {
                Some(m) => {
                    let since = m.steps_since_arm + 1;
                    let every = m.det.cfg().check_every.max(1);
                    Some((since, since > m.skip_steps && since % every == 0))
                }
                None => None,
            };
            if let Some((since, sampled)) = gate {
                if sampled {
                    let m = self.trend.as_ref().unwrap();
                    let rg = crate::metrics::radius_of_gyration(self);
                    let ss_frac = if m.ss_ref.is_empty() {
                        1.0
                    } else {
                        crate::metrics::ss_kept_count(self, &m.ss_ref, m.hbond_n_o) as f64
                            / m.ss_ref.len() as f64
                    };
                    let t_ps = self.state.time;
                    let m = self.trend.as_mut().unwrap();
                    m.steps_since_arm = since;
                    trend_alarm = m.det.observe(t_ps, u_kcal, rg, ss_frac);
                } else {
                    self.trend.as_mut().unwrap().steps_since_arm = since;
                }
            }
        }
        if trend_alarm.is_some() {
            crashed = true;
            crash_reason = Some(format!("trend_alarm:{}", trend_alarm.unwrap_or("trend")));
        }

        if u_kcal.is_finite() {
            self.u_history.push_back(u_kcal);
            if self.u_history.len() > U_HISTORY_CAP {
                self.u_history.pop_front();
            }
        }

        let coords_ca: Vec<[f32; 3]> = self
            .topology
            .ca_indices
            .iter()
            .map(|&i| {
                let p = self.state.atoms[i].posit;
                [p.x, p.y, p.z]
            })
            .collect();

        // Accumulate time-averaged Cα (pseudo-label source) on finite steps.
        if !crashed && self.ca_acc.len() == coords_ca.len() {
            for (acc, c) in self.ca_acc.iter_mut().zip(&coords_ca) {
                acc[0] += c[0] as f64;
                acc[1] += c[1] as f64;
                acc[2] += c[2] as f64;
            }
            self.ca_n += 1;
        }

        StepResult {
            u_t_kcal: u_kcal,
            u_t_kj: u_kcal * 4.184,
            coords_ca,
            step_count: self.state.step_count,
            time_ps: self.state.time,
            crashed,
            crash_reason,
            trend_alarm,
        }
    }

    /// Evaluate the currently selected SE force-field adapter on the current
    /// system. The MD step still uses dynamics' integrator until the low-level
    /// force boundary is split out.
    pub fn evaluate_selected_force_field(
        &self,
    ) -> Result<(ForceBuffer, crate::forcefield::EnergyVirial), String> {
        let atoms: Vec<ForceAtom> = self
            .state
            .atoms
            .iter()
            .map(|atom| ForceAtom {
                position: atom.posit,
                charge: atom.partial_charge,
                sigma: atom.lj_sigma,
                epsilon: atom.lj_eps,
            })
            .collect();
        let system = ForceFieldSystem {
            atoms: &atoms,
            bonds: &[],
        };
        let prepared = self
            .force_field
            .prepare_for_content(self.computation_content)
            .map_err(|e| e.to_string())?;
        let mut forces = ForceBuffer::zeros(atoms.len());
        let energy = prepared
            .evaluate(&system, &mut forces)
            .map_err(|e| e.to_string())?;
        Ok((forces, energy))
    }

    /// Live temperature change (environment perturbation, e.g. +ΔT).
    pub fn set_temperature(&mut self, k: f32) {
        self.state.cfg.temp_target = k;
    }

    /// Dual-bath annealing (v1.3.8): `Some(k)` gives the solute (protein +
    /// ions) its OWN Langevin setpoint while water keeps `temp_target`.
    /// Motivation (v8b): water owns ~96% of the heat capacity, so a single
    /// bath relaxes the solute slower than an anneal stage — the folding
    /// window never got its nominal temperature while the chain was still
    /// hot. `None` restores the single-bath behavior.
    pub fn set_solute_temperature(&mut self, k: Option<f32>) {
        self.state.cfg.temp_target_solute = k;
    }

    /// Dual-bath semantics (v1.3.8, calibrated today): the solite's OU
    /// noise fixes per-site variance, but intramolecular redistribution
    /// lags at production friction — measured solT/setpoint = 0.72-0.78 at
    /// γ=0.5 vs 1.0 at γ≥2. `Some(k)` therefore ALSO stiffens the solute
    /// coupling to γ=2.0 (folding-relevant timescale, Kramers high-γ; the
    /// hot bath still keeps waters fast), `None` restores single-bath
    /// behavior including the caller's γ.
    pub fn set_dual_bath(&mut self, solute_k: Option<f32>) {
        self.set_solute_temperature(solute_k);
        if solute_k.is_some() {
            self.set_langevin_gamma(2.0);
        }
    }

    /// Runtime Langevin friction (see ffi::set_langevin_gamma for why).
    pub fn set_langevin_gamma(&mut self, gamma: f32) {
        if let crate::engine::md_core::Integrator::LangevinMiddle { gamma: g } =
            &mut self.state.cfg.integrator
        {
            *g = gamma;
        }
    }

    /// Attach / replace the fail-fast trend monitor. Snapshots the current
    /// backbone H-bond network as the SS reference (same DSSP-lite proxy as
    /// `m3`), so call this AFTER the structure you want to protect against
    /// unfolding is in place (typically right after build + equilibration).
    /// From then on every `step`/`step_borrowed` feeds the detector on its
    /// cadence and a `crashed=true` with a `trend_alarm:` reason means the
    /// environment started driving the fold the wrong way — stop the episode.
    pub fn set_trend_monitor(&mut self, cfg: TrendConfig) {
        self.set_trend_monitor_skip(cfg, 0);
    }

    /// Like `set_trend_monitor` but leaves the first `skip_steps` steps
    /// unobserved — scans use this to skip their per-segment equilibration
    /// prefix so only production samples feed the window.
    pub fn set_trend_monitor_skip(&mut self, cfg: TrendConfig, skip_steps: usize) {
        let hbond_n_o = crate::metrics::MetricsConfig::default().hbond_n_o;
        let ss_ref = crate::metrics::backbone_hbonds(self, hbond_n_o);
        self.trend = Some(TrendMonitor::new(cfg, ss_ref, hbond_n_o));
        if let Some(m) = &mut self.trend {
            m.arm(skip_steps);
        }
    }

    /// Detach the trend monitor (no early termination).
    pub fn clear_trend_monitor(&mut self) {
        self.trend = None;
    }

    /// Whether a trend monitor is currently attached.
    pub fn has_trend_monitor(&self) -> bool {
        self.trend.is_some()
    }

    /// Empty the trend detector's sliding window without changing config or
    /// the SS reference — call at the START of each measured segment so the
    /// window only sees that segment (the equilibration→production handover
    /// transient must not bleed into the verdict). The monitor stays inert
    /// until `window` observations accumulate.
    pub fn reset_trend(&mut self) {
        if let Some(monitor) = &mut self.trend {
            monitor.arm(0);
        }
    }

    /// Re-zero all velocities (e.g. before a fresh run) — SOLUTE and SOLVENT.
    /// Water is included so a cold start is actually cold: leaving the rigid
    /// water at the build/solvent-init temperature (~420 K on 2LYZ) means the
    /// equilibrate ramp and production both start far above the target and the
    /// weak production thermostat (gamma=0.5) can't pull them down in a short
    /// window (measured: t_kin sits ~+65-75 K above target for hundreds of steps).
    pub fn reset_velocities(&mut self) {
        for a in &mut self.state.atoms {
            a.vel = Vec3::new_zero();
        }
        for w in &mut self.state.water {
            w.o.vel = Vec3::new_zero();
            w.h0.vel = Vec3::new_zero();
            w.h1.vel = Vec3::new_zero();
        }
    }

    /// Time-averaged Cα coordinates (Å), the pseudo-label source. Falls back
    /// to current coordinates when no steps have been accumulated.
    pub fn time_averaged_ca(&self) -> Vec<[f32; 3]> {
        if self.ca_n == 0 {
            // Fallback: return instantaneous coordinates of current state
            return self
                .topology
                .ca_indices
                .iter()
                .map(|&i| {
                    let p = self.state.atoms[i].posit;
                    [p.x, p.y, p.z]
                })
                .collect();
        }
        let inv = 1.0 / self.ca_n as f64;
        self.ca_acc
            .iter()
            .map(|a| {
                [
                    a[0] as f32 * inv as f32,
                    a[1] as f32 * inv as f32,
                    a[2] as f32 * inv as f32,
                ]
            })
            .collect()
    }

    /// Clear the potential-energy history and the pseudo-label accumulator, so
    /// metrics computed afterwards start from a clean (post-equilibration) state.
    pub fn reset_history(&mut self) {
        self.u_history.clear();
        for a in &mut self.ca_acc {
            *a = [0.0f64; 3];
        }
        self.ca_n = 0;
    }

    /// Reset the pseudo-label accumulator (start a fresh averaging window).
    pub fn reset_pseudo_labels(&mut self) {
        let n = self.topology.ca_indices.len();
        self.ca_acc = vec![[0.0f64; 3]; n];
        self.ca_n = 0;
    }

    /// Re-target an existing restraint (insertion order) instead of pushing a
    /// duplicate: the ramp of a steered-MD pull rewrites one entry per frame;
    /// `clear` is the RELEASE that makes quench-and-refold protocols expressible
    /// at all (pull with r0 ramping, then clear and watch unbiased re-formation).
    pub fn update_distance_restraint(&mut self, idx: usize, r0: f32, k: f32) -> bool {
        match self.state.distance_restraints.get_mut(idx) {
            Some(rs) => {
                rs.r0 = r0;
                rs.k = k;
                true
            }
            None => false,
        }
    }

    pub fn clear_distance_restraints(&mut self) {
        self.state.distance_restraints.clear();
    }

    /// Register a harmonic distance restraint between two atoms (e.g. for AlphaFold 3 ligand/ion coordination).
    pub fn add_distance_restraint(
        &mut self,
        atom_0_idx: usize,
        atom_1_idx: usize,
        r0: f32,
        k: f32,
    ) {
        self.state
            .distance_restraints
            .push(crate::engine::md_core::DistanceRestraint {
                atom_0_idx,
                atom_1_idx,
                r0,
                k,
            });
    }

    // ------------------------------------------------------------------
    // Observability: subset geometry on top of `md_core::analysis`.
    // Read-only diagnostics; the MD hot path and golden byte-identity are
    // untouched. ESP and SASA live on MdState (they need only state).
    // ------------------------------------------------------------------

    /// One PDB-style atom name per `state.atoms` entry. Protein atoms carry
    /// their mmCIF name ("CA", "NE2", "OG", "HH11", ...) via the topology;
    /// atoms outside the topology (ions, cosolvent molecules inserted at
    /// hydration time) fall back to their `force_field_type` label ("Na+",
    /// "Cl-", ...). This is the name space `select_atoms(names=...)` matches.
    /// Note: `force_field_type` is an Amber ATOM TYPE ("2C", "O2", ...), not
    /// a name - never use it as one.
    pub fn atom_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .state
            .atoms
            .iter()
            .map(|a| a.force_field_type.clone())
            .collect();
        for r in &self.topology.residues {
            for (&idx, nm) in r.atom_indices.iter().zip(&r.atom_names) {
                if idx < names.len() && !nm.is_empty() {
                    names[idx] = nm.clone();
                }
            }
        }
        names
    }

    /// Map residue sequence ids to `state.atoms` indices.
    /// - `res_seq`: residue ids (as in the source mmCIF; a residue appearing
    ///   in several chains matches each occurrence);
    /// - `names`: optional exact allow-list against true PDB-style atom names
    ///   (the space reported by `atom_names`);
    /// - `sidechain_heavy`: keep only non-hydrogen, non-backbone atoms
    ///   (backbone = N, CA, C, O, OXT). This is the "pull the sidechain heavy
    ///   atoms of residues 45-60" query catalytic-site work needs.
    /// Result preserves residue order. Water and ions are not in the
    /// topology, so `res_seq` can never select them (ions are still reachable
    /// via `names=["Na+"]`-style fallback only for atoms the topology owns;
    /// select ions through their force_field_type label, not this query).
    pub fn select_atoms(
        &self,
        res_seq: &[i32],
        names: Option<&[String]>,
        sidechain_heavy: bool,
    ) -> Vec<usize> {
        const BACKBONE: [&str; 5] = ["N", "CA", "C", "O", "OXT"];
        let n_atoms = self.state.atoms.len();
        let mut out = Vec::new();
        for r in &self.topology.residues {
            if !res_seq.contains(&r.seq_id) {
                continue;
            }
            for (k, &idx) in r.atom_indices.iter().enumerate() {
                if idx >= n_atoms {
                    continue; // defensive: stale topology cannot panic a probe
                }
                let name = &r.atom_names[k];
                if let Some(list) = names {
                    if !list.iter().any(|n| n == name) {
                        continue;
                    }
                }
                if sidechain_heavy {
                    if self.state.atoms[idx].element == na_seq::Element::Hydrogen {
                        continue;
                    }
                    if BACKBONE.contains(&name.as_str()) {
                        continue;
                    }
                }
                out.push(idx);
            }
        }
        out
    }

    /// Count pairs (a_i, b_j) of `state.atoms` closer than `cutoff` (Angstrom,
    /// minimum image). Indices are `state.atoms` slots (what `select_atoms`
    /// returns). Q_site contact retention = compare this before/after a
    /// mutation or trajectory window. Passing the same set twice counts each
    /// unordered pair once and skips self-pairs. Water is excluded by design:
    /// the query compares solute/ion subsets against each other.
    /// Panics on out-of-range indices (the FFI validates first).
    pub fn contact_count(&self, a: &[usize], b: &[usize], cutoff: f64) -> usize {
        let pos: Vec<_> = self.state.atoms.iter().map(|x| x.posit).collect();
        crate::engine::md_core::analysis::contact_count(a, b, &pos, self.state.cell.extent, cutoff)
    }

    /// Clearance profile (Angstrom) along a polyline: resampled every
    /// `spacing`, the distance from each sample to the nearest vdW SURFACE
    /// (minimum image). The bottleneck radius of a channel is
    /// `profile.iter().min()`: a sphere of radius r passes iff min >= r.
    /// `exclude` skips `state.atoms` indices (e.g. the substrate whose
    /// channel you are measuring). `include_water` makes solvent sites walls
    /// too (hydrated bottleneck); otherwise only solute+ion+cosolvent atoms
    /// line the path. Water sites shift frame to frame: for a stable number
    /// average over trajectory samples, or leave them out.
    pub fn bottleneck_profile(
        &self,
        path: &[[f64; 3]],
        spacing: f64,
        exclude: &[usize],
        include_water: bool,
    ) -> Vec<f64> {
        use crate::engine::md_core::analysis;
        let (pos, radii) = if include_water {
            self.state.real_sites_pos_radii()
        } else {
            self.state
                .atoms
                .iter()
                .map(|a| {
                    (
                        a.posit,
                        analysis::vdw_radius(&a.element, f64::from(a.lj_sigma)),
                    )
                })
                .unzip()
        };
        analysis::bottleneck_profile(
            path,
            spacing.max(0.05),
            &pos,
            &radii,
            self.state.cell.extent,
            exclude,
        )
    }
}

#[cfg(test)]
mod trend_tests {
    use super::*;

    // A config that fills the window quickly so tests stay tiny (5 obs).
    fn fast_cfg() -> TrendConfig {
        TrendConfig {
            enabled: true,
            window: 5,
            check_every: 1,
            z_threshold: 3.0,
            energy_floor_ps: 1.0,
            rg_floor_ps: 0.1,
            ss_floor_ps: 0.01,
        }
    }

    #[test]
    fn slope_is_zero_for_flat_and_linear_for_ramp() {
        let t: VecDeque<f64> = [0.0, 1.0, 2.0, 3.0].into();
        let flat: VecDeque<f64> = [5.0, 5.0, 5.0, 5.0].into();
        assert!((slope_ps(&t, &flat)).abs() < 1e-12);
        let ramp: VecDeque<f64> = [1.0, 3.0, 5.0, 7.0].into();
        assert!((slope_ps(&t, &ramp) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn inert_until_window_fills() {
        let mut d = TrendDetector::new(fast_cfg());
        // Window is 5; first 4 bad observations must not fire (not enough fit).
        for i in 0..4 {
            assert_eq!(
                d.observe(i as f64, i as f64 * 10.0, 1.0 + i as f64, 1.0),
                None
            );
        }
    }

    #[test]
    fn three_bad_signals_fire_and_report_reason() {
        let mut d = TrendDetector::new(fast_cfg());
        // Energy +Rg rising, SS falling — all three bad, strongly above floors.
        let mut fired = None;
        for i in 0..8 {
            let dt = i as f64;
            fired = d.observe(dt, dt * 1000.0, 1.0 + dt * 10.0, (1.0 - dt * 0.1).max(0.0));
            if fired.is_some() {
                break;
            }
        }
        // Any ≥2-of-3 reason is acceptable; it must be one of the named signals.
        let sig = fired.expect("trend should fire on uniformly-bad ramp");
        assert!(
            matches!(sig, "energy_rise" | "rg_expand" | "ss_loss"),
            "{sig}"
        );
    }

    #[test]
    fn single_bad_signal_does_not_fire() {
        let mut d = TrendDetector::new(fast_cfg());
        // Only energy rises; Rg flat, SS flat → 1 of 3 → no termination.
        for i in 0..10 {
            let dt = i as f64;
            assert_eq!(d.observe(dt, dt * 1000.0, 2.0, 0.5), None);
        }
    }

    #[test]
    fn reset_clears_window_back_to_inert() {
        let mut d = TrendDetector::new(fast_cfg());
        for i in 0..8 {
            d.observe(i as f64, i as f64 * 1000.0, 1.0 + i as f64, 1.0);
        }
        d.reset();
        // After reset the window is empty again → the next single observation
        // is inert (needs `window` samples before it can fit a slope).
        assert_eq!(d.observe(0.0, 0.0, 1.0, 1.0), None);
    }

    #[test]
    fn disabled_config_never_terminates() {
        let mut d = TrendDetector::new(TrendConfig {
            enabled: false,
            ..fast_cfg()
        });
        for i in 0..12 {
            let dt = i as f64;
            assert_eq!(
                d.observe(dt, dt * 1000.0, 1.0 + dt * 10.0, (1.0 - dt * 0.1).max(0.0)),
                None
            );
        }
    }

    #[test]
    fn rl_fail_fast_config_is_tight() {
        let c = TrendConfig::rl_fail_fast();
        assert!(c.enabled);
        // window × check_every ≈ 300 steps = 0.6 ps at dt = 2 fs — kills a
        // diverging rollout well before numerical blow-up.
        assert_eq!(c.window * c.check_every, 300);
    }
}
