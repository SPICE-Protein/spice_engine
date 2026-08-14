//! Stability-domain search — scan protein stability over an environment-parameter
//! grid (temperature / pH / ionic strength / pressure).
//!
//! For each environment point: build the system under that environment (pH sets
//! protonation, T sets the thermostat, ionic strength adds salt), run a short MD
//! window, and use the five physical metrics to decide whether the protein keeps
//! its native fold (no crash + no unfolding). Output is a `StabilityPoint` grid —
//! the protein's **stability domain** in multidimensional SPICE space.
//!
//! Parallelised with rayon: each environment point builds + simulates
//! independently (MdState is `Send`). `EnvGrid::mesh()` produces the Cartesian
//! product of the parameter space; `is_stable` gives the decision rule (reusable
//! as SAC reward / threshold).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use rayon::prelude::*;

use lin_alg::f32::Vec3;

use crate::builder::BuildOptions;
use crate::env::EnvParams;
use crate::metrics::{Metrics, MetricsConfig, MetricsResult};
use crate::structure::{StructureInput, build_from_input};

static PROGRESS_BAR: Mutex<Option<indicatif::ProgressBar>> = Mutex::new(None);

/// Environment-parameter grid (point sets per axis). `mesh()` takes the Cartesian product.
#[derive(Debug, Clone)]
pub struct EnvGrid {
    pub temps: Vec<f32>,
    pub phs: Vec<f32>,
    pub pressures: Vec<f32>,
    pub ionics: Vec<f32>,
}

impl Default for EnvGrid {
    /// A small default around physiological conditions.
    fn default() -> Self {
        Self {
            temps: vec![300.0, 310.0, 320.0],
            phs: vec![6.5, 7.0, 7.5],
            pressures: vec![1.0],
            ionics: vec![0.0],
        }
    }
}

impl EnvGrid {
    /// Evenly spaced points from `start` to `end` (inclusive) with `step`.
    ///
    /// Each SPICE axis gets its own resolution — e.g. fine temperature steps
    /// (5 K) but coarse pH steps (0.5–1.0 units, since protonation states are
    /// discrete and don't need high precision). `step <= 0` or `end < start`
    /// yields an empty axis.
    pub fn linspace(start: f32, end: f32, step: f32) -> Vec<f32> {
        if step <= 0.0 || end < start {
            return Vec::new();
        }
        let n = ((end - start) / step).floor() as usize + 1;
        (0..n).map(|i| start + i as f32 * step).collect()
    }

    /// Build a grid from per-axis `(start, end, step)` ranges, so each dimension
    /// can use a different resolution. `None` pressure/ionic keeps the default
    /// single point (1.0 bar / 0.0 M).
    pub fn from_ranges(
        temps: (f32, f32, f32),
        phs: (f32, f32, f32),
        pressures: Option<(f32, f32, f32)>,
        ionics: Option<(f32, f32, f32)>,
    ) -> Self {
        Self {
            temps: Self::linspace(temps.0, temps.1, temps.2),
            phs: Self::linspace(phs.0, phs.1, phs.2),
            pressures: pressures
                .map(|p| Self::linspace(p.0, p.1, p.2))
                .unwrap_or_else(|| vec![1.0]),
            ionics: ionics
                .map(|i| Self::linspace(i.0, i.1, i.2))
                .unwrap_or_else(|| vec![0.0]),
        }
    }

    pub fn mesh(&self) -> Vec<EnvParams> {
        let mut out = Vec::with_capacity(
            self.temps.len() * self.phs.len() * self.pressures.len() * self.ionics.len(),
        );
        for &temp in &self.temps {
            for &ph in &self.phs {
                for &pressure in &self.pressures {
                    for &ionic in &self.ionics {
                        out.push(EnvParams::new(ph, temp, pressure, ionic));
                    }
                }
            }
        }
        out
    }

    pub fn is_empty(&self) -> bool {
        self.temps.is_empty() || self.phs.is_empty() || self.pressures.is_empty() || self.ionics.is_empty()
    }
}

/// Stability-decision thresholds (used by `is_stable` and scans).
#[derive(Debug, Clone)]
pub struct StabilityConfig {
    /// MD steps (per environment point) over which the metrics are computed.
    pub n_steps: usize,
    /// Equilibration steps run (and discarded from `m1`) before the measured
    /// window — lets the initial build strain settle so `m1` is meaningful.
    pub equil_steps: usize,
    /// Independent MD segments per point; stability is decided by majority vote
    /// across segments. Guards against stochastic one-off crashes from residual
    /// strain (build once, reuse the same engine, reset velocities per segment).
    pub repeats: usize,
    /// Minimization iterations at build (`None` = `BuildOptions` default; `Some` overrides).
    pub relax_iters: Option<usize>,
    /// Minimization convergence tolerance kcal/(mol·Å).
    pub tolerance: f32,
    /// m2: max allowed Rg drift (relative).
    pub max_rg_drift: f64,
    /// m3: max allowed secondary-structure loss.
    pub max_ss_loss: f64,
    /// m4: max allowed clash fraction.
    pub max_clash: f64,
    /// m5: max allowed surface-charge mismatch.
    pub max_charge_mismatch: f64,
    /// m1: max allowed potential-energy fluctuation (guards against energy blow-up).
    pub max_energy_fluct: f64,
    /// Print a per-point progress line to stderr (`[stability] ...`). Useful for
    /// long scans; stderr is line-flushed, so lines show up live even when
    /// stdout is redirected to a file.
    pub progress: bool,
    /// Two-phase pruning for grid scans: each build-env group (one pH) is
    /// screened at the temperature closest to `anchor_temp` first; a column is
    /// worth a temperature walk only if the protein is STABLE (folded) there.
    /// Columns that fail to build, majority-crash, or simply unfold at the
    /// reference temperature are pruned — their remaining temperatures are
    /// marked `build_failed` and skipped, so no T-scan is wasted on a pH that
    /// doesn't hold its fold at the reference.
    pub prune_crashed: bool,
    /// Screening temperature for `prune_crashed`.
    pub anchor_temp: f32,
    /// B: early-abort a segment when potential energy exceeds this (kcal/mol).
    /// Normal U for a solvated protein is strongly negative; a large positive
    /// spike means divergence is underway, so cut the doomed segment short
    /// instead of running out the window to `CRASH_ENERGY`. `0` disables.
    pub early_abort_u: f64,
    /// C: adaptive repeats — run 1 segment; only run the remaining repeats when
    /// the first segment's verdict is borderline (a metric close to its
    /// threshold). Clearly stable/unstable points cost a single segment.
    pub adaptive_repeats: bool,
    /// C: clear-verdict margin (fraction of the threshold). A metric below
    /// `threshold × clear_margin` ⇒ clearly stable; above `threshold ÷
    /// clear_margin` ⇒ clearly unstable; in between ⇒ borderline.
    pub clear_margin: f64,
    /// A: monotonicity pruning in grid scans — within a build-env group, walk
    /// temperatures outward from the reference and stop at the first unstable
    /// point in each direction, pruning the temperatures beyond it. Assumes
    /// stability is monotonic in temperature (past the optimum, hotter is never
    /// more stable). `false` runs every temperature.
    pub monotonic_prune: bool,
    /// Growth factor for the adaptive coarse walk (radial scan): the outward
    /// step is multiplied by this each time it stays stable, jumping through
    /// the stable interior faster (bolder = larger).
    pub step_growth: f32,
    /// Long-MD environment-trend early termination (soft; complements the hard
    /// `early_abort_u` spike). Inert while windows are shorter than the trend
    /// window.
    pub trend: TrendConfig,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            n_steps: 20,
            equil_steps: 10,
            repeats: 3,
            relax_iters: Some(2_000),
            tolerance: 2.0,
            max_rg_drift: 0.15,
            // m3 baseline is ~0.43 even at physiological conditions (DSSP-lite
            // over short windows), so the old 0.40 threshold flagged every point
            // as unfolded. 0.55 gives headroom above the baseline.
            max_ss_loss: 0.55,
            max_clash: 0.05,
            max_charge_mismatch: 1.0,
            max_energy_fluct: 1e5,
            progress: true,
            prune_crashed: true,
            anchor_temp: 310.0,
            early_abort_u: 1e4,
            adaptive_repeats: true,
            clear_margin: 0.6,
            monotonic_prune: true,
            step_growth: 3.0,
            trend: TrendConfig::default(),
        }
    }
}

/// Decide whether an environment point is stable (native fold preserved).
pub fn is_stable(m: &MetricsResult, crashed: bool, cfg: &StabilityConfig) -> bool {
    !crashed
        && m.m1 < cfg.max_energy_fluct
        && m.m2 < cfg.max_rg_drift
        && m.m3 < cfg.max_ss_loss
        && m.m4 < cfg.max_clash
        && m.m5 < cfg.max_charge_mismatch
}

/// Confidence heuristics for adaptive repeats: a first-segment verdict is
/// "clear" when every metric sits comfortably away from its threshold.
fn is_clear_stable(m: &MetricsResult, cfg: &StabilityConfig) -> bool {
    m.m1 < cfg.max_energy_fluct * cfg.clear_margin
        && m.m2 < cfg.max_rg_drift * cfg.clear_margin
        && m.m3 < cfg.max_ss_loss * cfg.clear_margin
        && m.m4 < cfg.max_clash * cfg.clear_margin
        && m.m5 < cfg.max_charge_mismatch * cfg.clear_margin
}

/// Soft trend detector for LONG-MD environment validation (v2 protocol).
///
/// Short-MD asks "does the structure fold?"; long-MD asks "does the environment
/// let the folded structure STAY folded?". Instead of predicting a crash, we
/// detect whether the environment is driving the system in a bad direction —
/// potential energy rising, Rg expanding, secondary structure dissolving — via
/// the slope of a sliding window, z-scored against a thermal-noise floor. A
/// segment terminates when ≥2 of the 3 signals are significant (scoring, not
/// strict AND, so a single noisy metric cannot lock out a true termination).
///
/// Designed for ns-scale windows. With the current short scan windows it stays
/// inert (the window never fills), which is intentional — it is the long-MD
/// protocol that becomes live once windows are long enough (v2/v3).
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

struct TrendDetector {
    cfg: TrendConfig,
    t_ps: VecDeque<f64>,
    energy: VecDeque<f64>,
    rg: VecDeque<f64>,
    ss: VecDeque<f64>,
}

impl TrendDetector {
    fn new(cfg: TrendConfig) -> Self {
        Self {
            cfg,
            t_ps: VecDeque::new(),
            energy: VecDeque::new(),
            rg: VecDeque::new(),
            ss: VecDeque::new(),
        }
    }

    /// Record one observation (time in ps, energy in kcal/mol, Rg in Å, and the
    /// fraction of reference secondary structure still present). Returns the
    /// triggering signal name once ≥2/3 trend slopes are significant in the bad
    /// direction (energy rising / Rg expanding / SS dissolving).
    fn observe(&mut self, t_ps: f64, energy: f64, rg: f64, ss_frac: f64) -> Option<&'static str> {
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
        if bad >= 2 {
            Some(reason)
        } else {
            None
        }
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
    if den.abs() < 1e-12 {
        0.0
    } else {
        num / den
    }
}

/// Scan result for a single environment point.
#[derive(Debug, Clone)]
pub struct StabilityPoint {
    pub env: EnvParams,
    pub stable: bool,
    /// Whether the MD crashed during the run (energy blow-up).
    pub crashed: bool,
    /// Whether system build failed for this environment point (e.g. a pH that
    /// cannot be protonated) — distinct from "unstable".
    pub build_failed: bool,
    /// Why a segment was terminated early (`energy_spike` = hard early-abort;
    /// `energy_rise`/`rg_expand`/`ss_loss` = long-MD trend termination; `None`
    /// = ran the full window). First terminating segment wins.
    pub terminated_reason: Option<String>,
    /// Terminal metrics; `None` = build failed or crash before evaluation.
    pub metrics: Option<MetricsResult>,
}

impl StabilityPoint {
    fn build_failed(env: EnvParams) -> Self {
        Self {
            env,
            stable: false,
            crashed: false,
            build_failed: true,
            terminated_reason: None,
            metrics: None,
        }
    }
}

/// A built (solvated + minimized) engine plus its reference conformation,
/// reused across environment points that share build-time parameters (pH,
/// pressure on/off, ionic strength). Temperature is applied per point at
/// runtime via `set_temperature` + velocity re-sampling — the LAMMPS
/// "`velocity create` + `fix nvt`" sweep pattern — so solvent init and L-BFGS
/// minimization run once per distinct build environment, not once per point.
struct PointTemplate {
    engine: crate::engine::SpiceEngine,
    /// Minimized reference conformation (post-build), restored at the start of
    /// every segment / point.
    init_pos: Vec<Vec3>,
    /// Reference box (post-build). Restored alongside `init_pos` so every
    /// restart starts from the build-time volume — reusing only positions
    /// leaves the box at the previous temperature's volume, and the density
    /// mismatch crashes cold restarts (fresh builds at that T are stable).
    init_cell: dynamics::SimBox,
    /// Environment the engine was built with.
    env: EnvParams,
}

/// Build-env key: two points can share one template iff pH (protonation),
/// pressure on/off (barostat presence) and ionic strength (salt) match.
fn build_key(e: EnvParams) -> (u32, bool, u32) {
    (
        e.ph.to_bits(),
        e.pressure_bar > 0.0,
        e.ionic_strength_m.to_bits(),
    )
}

/// Build one environment point, run `n_steps`, and classify it (used by both
/// the grid scan and the radial boundary scan).
///
/// `template` is an optional reusable engine. When the point's build-time
/// parameters match the template's, the solvated + minimized engine is reused
/// and only temperature / velocities are reset (skipping the expensive
/// per-point solvent init + minimization). A `None` or mismatching template
/// triggers a fresh build.
fn probe_point(
    dev: &dynamics::ComputationDevice,
    param_set: &dynamics::params::FfParamSet,
    structure: &StructureInput,
    env: EnvParams,
    build_opts: &BuildOptions,
    cfg: &StabilityConfig,
    template: &mut Option<PointTemplate>,
) -> StabilityPoint {
    // Rebuild only when build-time parameters changed. `relax_iters: None`
    // must inherit the build default (Some(2000)) — passing it through disables
    // minimization and every point crashes.
    let needs_rebuild = template
        .as_ref()
        .map_or(true, |t| build_key(t.env) != build_key(env));
    if needs_rebuild {
        let opts = BuildOptions {
            env,
            relax_iters: cfg.relax_iters.or(build_opts.relax_iters),
            energy_minimization_tolerance: cfg.tolerance,
            ..build_opts.clone()
        };
        let engine = match build_from_input(dev, param_set, structure, &opts) {
            Ok(e) => e,
            Err(_) => return StabilityPoint::build_failed(env),
        };
        // Snapshot the minimized reference conformation + box so every
        // segment/point starts from the same clean structure.
        let init_pos: Vec<Vec3> = engine.state.atoms.iter().map(|a| a.posit).collect();
        let init_cell = engine.state.cell;
        *template = Some(PointTemplate {
            engine,
            init_pos,
            init_cell,
            env,
        });
    }
    // Clone a PRISTINE engine for THIS point. The template is never mutated, so
    // every point starts from an identical, truly independent post-build state
    // (positions / box / neighbor lists / SPME + PME caches / history all reset
    // to the build-time values). This is the root fix for the reused-template
    // false crashes — e.g. the 298 K row in the 2D map crashing after the same
    // engine had run at 328/343/373 K, while a fresh build at 298 K is stable.
    // A clone of the pristine build is exactly equivalent to that fresh build.
    let template = template.as_ref().expect("template built");
    let init_pos = &template.init_pos;
    let init_cell = &template.init_cell;
    let mut engine = template.engine.clone();
    // Keep `engine.env` consistent with the runtime env — Metrics normalizes m1
    // by `engine.env.temp_k`, so it must reflect this point's temperature.
    engine.env = env;

    // Majority vote over independent MD segments (same built engine, velocities
    // re-sampled per segment) so a single stochastic crash does not flip the
    // point's verdict. B + C keep this cheap: a segment is aborted early when
    // its energy diverges, and a single clear first segment is trusted.
    let mut n_stable = 0usize;
    let mut n_crash = 0usize;
    let mut metrics: Option<MetricsResult> = None;
    let mut stable = false;
    let mut decided = false;
    let mut terminated_reason: Option<String> = None;
    let seg_steps = cfg.equil_steps + cfg.n_steps;
    let n_repeats = cfg.repeats.max(1);
    for seg in 0..n_repeats {
        // Restore the clean reference state (box + positions + neighbor list +
        // all position/box-derived caches) AND re-sample Maxwell-Boltzmann
        // velocities at the target temperature — the OpenMM "setPositions() +
        // setVelocitiesToTemperature()" restart pattern. Restoring the box too
        // avoids density-mismatch crashes when a template built at one
        // temperature is reused at a colder one.
        engine.state.set_state_rebuild(dev, init_pos, init_cell);
        engine.state.initialize_velocities(env.temp_k, true);
        engine.reset_history();
        engine.set_temperature(env.temp_k);
        let mref = Metrics::new(
            &engine,
            MetricsConfig {
                u_skip: cfg.equil_steps,
                ..Default::default()
            },
        );

        // v2 long-MD environment-trend detector (soft early termination). It
        // complements the hard `early_abort_u` spike and is inert while the
        // window is shorter than `trend.window` (intended for short scans).
        let mut trend = TrendDetector::new(cfg.trend.clone());

        let mut crashed = false;
        let mut reason: Option<&'static str> = None;
        for step_i in 0..seg_steps {
            let r = engine.step(None);
            // B: early abort — a positive energy spike means divergence is
            // underway; cut the doomed segment short instead of running to
            // CRASH_ENERGY (normal U for a solvated protein is strongly
            // negative, so U > early_abort_u is unambiguous blow-up).
            if r.crashed || (cfg.early_abort_u > 0.0 && r.u_t_kcal > cfg.early_abort_u) {
                crashed = true;
                reason = Some("energy_spike");
                break;
            }
            // Trend detection: sample cheap signals (energy is free; Rg and SS
            // only on check steps) and terminate once the environment drives
            // the fold in a bad direction (≥2/3 signals significant).
            if cfg.trend.enabled
                && step_i >= cfg.equil_steps
                && (step_i + 1) % cfg.trend.check_every.max(1) == 0
            {
                let t_ps = engine.state.time;
                let rg = crate::metrics::radius_of_gyration(&engine);
                let ss_frac = if mref.ss_ref.is_empty() {
                    1.0
                } else {
                    crate::metrics::ss_kept_count(&engine, &mref.ss_ref, mref.config.hbond_n_o)
                        as f64
                        / mref.ss_ref.len() as f64
                };
                if let Some(sig) = trend.observe(t_ps, r.u_t_kcal, rg, ss_frac) {
                    crashed = true;
                    reason = Some(sig);
                    break;
                }
            }
        }
        if crashed {
            n_crash += 1;
            if terminated_reason.is_none() {
                terminated_reason = reason.map(|s| s.to_string());
            }
            // C: never decide "unstable" from a single crashed segment — a
            // marginal system (e.g. acidic pH) crashes stochastically. Run the
            // remaining repeats and let the majority vote decide.
            continue;
        }
        let m = mref.compute(&engine);
        if is_stable(&m, false, cfg) {
            n_stable += 1;
        }
        // C: trust the CLEAR-STABLE shortcut after a single segment (saves the
        // stable interior of the domain). Anything else — a crash, or a
        // borderline / apparently-unstable first segment — runs the full
        // repeats for the majority vote, so marginal systems are not misjudged
        // from one noisy segment. Checked before `m` is moved into `metrics`.
        let clear_stable = cfg.adaptive_repeats && seg == 0 && is_clear_stable(&m, cfg);
        metrics = Some(m);
        if clear_stable {
            stable = true;
            decided = true;
            break;
        }
    }
    if !decided {
        stable = n_stable > n_crash;
    }
    StabilityPoint {
        env,
        stable,
        crashed: n_crash > 0,
        build_failed: false,
        terminated_reason,
        metrics,
    }
}

/// Evaluate one point in a `scan_stability` group (probe + progress line) with
/// template reuse. `tag` is appended to the progress line (e.g. `[screen]`).
fn log_progress(msg: String, inc: bool) {
    let mut printed = false;
    if let Some(pb) = &*PROGRESS_BAR.lock().unwrap() {
        if !pb.is_hidden() {
            pb.println(&msg);
            printed = true;
        }
        if inc {
            pb.inc(1);
        }
    }
    if !printed {
        crate::log_eprint(msg);
    }
}

fn eval_group_point(
    dev: &dynamics::ComputationDevice,
    param_set: &dynamics::params::FfParamSet,
    structure: &StructureInput,
    env: EnvParams,
    build_opts: &BuildOptions,
    cfg: &StabilityConfig,
    template: &mut Option<PointTemplate>,
    done: &AtomicUsize,
    tag: &str,
) -> StabilityPoint {
    let pt = probe_point(dev, param_set, structure, env, build_opts, cfg, template);
    if cfg.progress {
        // `done` counts ACTUAL MD simulations (builds + measurements) — not
        // output-grid cells — so the user sees real computational progress.
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        let verdict = if pt.build_failed {
            "build_failed"
        } else if pt.stable {
            "stable"
        } else {
            "unstable"
        };
        let log_line = format!(
            "[stability] eval {n:>3}: T={:.0} pH={:.1}: {verdict} (crashed={}){tag}",
            env.temp_k,
            env.ph,
            pt.crashed,
        );
        log_progress(log_line, true);
    }
    pt
}

/// A skipped (pruned) point: not simulated, marked `build_failed`.
fn pruned_group_point(env: EnvParams, cfg: &StabilityConfig, reason: &str) -> StabilityPoint {
    if cfg.progress {
        let log_line = format!(
            "[stability] PRUNED: T={:.0} pH={:.1}: {reason}",
            env.temp_k,
            env.ph,
        );
        log_progress(log_line, true);
    }
    StabilityPoint {
        env,
        stable: false,
        crashed: true,
        build_failed: true,
        terminated_reason: None,
        metrics: None,
    }
}

/// A grid point inferred stable (not simulated): monotonicity places it inside
/// the stable interval between two measured-stable temperatures, so it is
/// reported stable without running MD. Metrics are `None` (not measured).
fn inferred_stable(env: EnvParams) -> StabilityPoint {
    StabilityPoint {
        env,
        stable: true,
        crashed: false,
        build_failed: false,
        terminated_reason: None,
        metrics: None,
    }
}

/// Bold dynamic-step walk of a temperature-ordered index sequence: jump through
/// the (assumed monotonic) stable interior with DOUBLING index steps, inferring
/// the skipped interior points as stable without running them, then bisect near
/// the boundary and prune everything beyond the first unstable point.
/// `indices` must be ordered by temperature in the direction of the walk
/// (ascending for "above", descending for "below").
fn bold_walk(
    dev: &dynamics::ComputationDevice,
    param_set: &dynamics::params::FfParamSet,
    structure: &StructureInput,
    envs: &[EnvParams],
    indices: &[usize],
    build_opts: &BuildOptions,
    cfg: &StabilityConfig,
    template: &mut Option<PointTemplate>,
    done: &AtomicUsize,
    results: &mut [Option<StabilityPoint>],
) {
    let mut ls: Option<usize> = None; // index in `indices` of last measured stable
    let mut fu: Option<usize> = None; // index in `indices` of first unstable
    let mut i = 0usize;
    // Coarse doubling: 0, 1, 3, 7, 15...
    while i < indices.len() {
        let idx = indices[i];
        let pt = eval_group_point(
            dev, param_set, structure, envs[idx], build_opts, cfg, template, done, "",
        );
        let stable = pt.stable;
        results[idx] = Some(pt);
        if !stable {
            fu = Some(i);
            break;
        }
        // Infer skipped interior (between the previous stable and this one) as
        // stable — monotonicity guarantees they lie inside the stable interval.
        if let Some(prev) = ls {
            for j in (prev + 1)..i {
                let jidx = indices[j];
                results[jidx] = Some(inferred_stable(envs[jidx]));
            }
        }
        ls = Some(i);
        i = i * 2 + 1;
    }
    // Bisect between the last stable and first unstable.
    if let (Some(mut l), Some(mut f)) = (ls, fu) {
        while f - l > 1 {
            let mid = (l + f) / 2;
            let idx = indices[mid];
            let pt = eval_group_point(
                dev, param_set, structure, envs[idx], build_opts, cfg, template, done, "",
            );
            let stable = pt.stable;
            results[idx] = Some(pt);
            if stable {
                for j in (l + 1)..mid {
                    let jidx = indices[j];
                    results[jidx] = Some(inferred_stable(envs[jidx]));
                }
                l = mid;
            } else {
                f = mid;
            }
        }
        // Prune everything beyond the first unstable.
        for j in (f + 1)..indices.len() {
            let jidx = indices[j];
            results[jidx] = Some(pruned_group_point(
                envs[jidx], cfg, "monotonic: beyond boundary",
            ));
        }
    } else if let Some(l) = ls {
        // Never hit unstable — infer the remaining tail as stable.
        for j in (l + 1)..indices.len() {
            let jidx = indices[j];
            results[jidx] = Some(inferred_stable(envs[jidx]));
        }
    } else if let Some(f) = fu {
        // The very first point in this direction was already unstable — there is
        // no stable interval, so everything beyond it is pruned (this path was
        // previously missed, leaving group points unfilled → panic).
        for j in (f + 1)..indices.len() {
            let jidx = indices[j];
            results[jidx] = Some(pruned_group_point(
                envs[jidx], cfg, "monotonic: beyond boundary",
            ));
        }
    }
}

/// strategy — no T-scan is wasted on a pH that cannot even build.
pub fn scan_stability(
    dev: &dynamics::ComputationDevice,
    param_set: &dynamics::params::FfParamSet,
    structure: &StructureInput,
    grid: &EnvGrid,
    build_opts: &BuildOptions,
    cfg: &StabilityConfig,
) -> Vec<StabilityPoint> {
    if grid.is_empty() {
        return Vec::new();
    }
    let mesh = grid.mesh();
    let total = mesh.len();
    let done = AtomicUsize::new(0);

    if cfg.progress {
        let pb = indicatif::ProgressBar::new(total as u64);
        pb.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.green/blue} {pos}/{len} {msg} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message("Scanning grid stability...");
        *PROGRESS_BAR.lock().unwrap() = Some(pb);
    }

    if cfg.progress {
        log_progress(format!(
            "[stability] grid = {total} cells; progress below reports ACTUAL MD simulations as \"eval #\""
        ), false);
    }
    // Partition mesh points by build-env key; each group reuses one solvated +
    // minimized template (the LAMMPS velocity-create sweep pattern), so solvent
    // init + L-BFGS run once per distinct build env instead of per point.
    let mut groups: HashMap<(u32, bool, u32), Vec<EnvParams>> = HashMap::new();
    for env in &mesh {
        groups.entry(build_key(*env)).or_default().push(*env);
    }

    // ---- Stage 1 (parallel): screen each column at its reference temperature.
    if cfg.progress {
        log_progress(format!(
            "[stability] stage 1: screening {} columns at T≈{:.0} K (each = 1 build + ref MD eval)",
            groups.len(),
            cfg.anchor_temp,
        ), false);
    }
    let screened: Vec<(
        (u32, bool, u32),
        usize,
        StabilityPoint,
        Vec<EnvParams>,
        Option<PointTemplate>,
    )> = groups
        .into_par_iter()
        .map(|(k, envs)| {
            let mut template: Option<PointTemplate> = None;
            // Screening reference: the temperature closest to `anchor_temp`.
            let ref_idx = envs
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (a.temp_k - cfg.anchor_temp)
                        .abs()
                        .partial_cmp(&(b.temp_k - cfg.anchor_temp).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            let ref_env = envs[ref_idx];
            let ref_pt = eval_group_point(
                dev,
                param_set,
                structure,
                ref_env,
                build_opts,
                cfg,
                &mut template,
                &done,
                " [screen]",
            );
            (k, ref_idx, ref_pt, envs, template)
        })
        .collect();

    if cfg.progress {
        let stable_cols = screened.iter().filter(|(_, _, p, _, _)| p.stable).count();
        log_progress(format!(
            "[stability] stage 2: {stable_cols}/{} columns stable at ref — walking T (\"eval #\" = actual MD sims)",
            screened.len(),
        ), false);
    }

    // ---- Stage 2 (parallel): walk T within each column, reusing its template.
    let by_key: HashMap<(u32, bool, u32), Vec<StabilityPoint>> = screened
        .into_par_iter()
        .map(|(k, ref_idx, ref_pt, envs, mut template)| {
            let n = envs.len();
            let mut results: Vec<Option<StabilityPoint>> = vec![None; n];
            // Stage 1 is a "stable pH" screen: a column is worth a T-scan only
            // if the protein is FOLDED at the reference temperature. Build
            // failures, majority crashes, or simply unfolded columns are pruned
            // — no T-walk is wasted on a pH that doesn't hold its fold at the
            // reference (the user-specified two-stage design).
            let pruned = cfg.prune_crashed && !ref_pt.stable;
            results[ref_idx] = Some(ref_pt);
            if pruned {
                for (i, env) in envs.iter().enumerate() {
                    if i != ref_idx {
                        results[i] = Some(pruned_group_point(
                            *env,
                            cfg,
                            "pH not stable at ref",
                        ));
                    }
                }
            } else if cfg.monotonic_prune && n > 1 {
                // A (bold): doubling index jumps through the stable interior
                // (inferring the skipped points stable), bisect near the
                // boundary, prune beyond it.
                let ref_temp = envs[ref_idx].temp_k;
                let mut order: Vec<usize> = (0..n).filter(|&i| i != ref_idx).collect();
                order.sort_by(|&a, &b| {
                    envs[a]
                        .temp_k
                        .partial_cmp(&envs[b].temp_k)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let above: Vec<usize> = order
                    .iter()
                    .copied()
                    .filter(|&i| envs[i].temp_k > ref_temp)
                    .collect();
                let mut below: Vec<usize> = order
                    .iter()
                    .copied()
                    .filter(|&i| envs[i].temp_k < ref_temp)
                    .collect();
                below.reverse(); // walk away below (descending temperatures)
                bold_walk(
                    dev, param_set, structure, &envs, &above, build_opts, cfg,
                    &mut template, &done, &mut results,
                );
                bold_walk(
                    dev, param_set, structure, &envs, &below, build_opts, cfg,
                    &mut template, &done, &mut results,
                );
            } else {
                for (i, env) in envs.iter().enumerate() {
                    if i != ref_idx {
                        results[i] = Some(eval_group_point(
                            dev,
                            param_set,
                            structure,
                            *env,
                            build_opts,
                            cfg,
                            &mut template,
                            &done,
                            "",
                        ));
                    }
                }
            }
            (k, results.into_iter().map(|o| o.expect("group point filled")).collect())
        })
        .collect();

    if let Some(pb) = PROGRESS_BAR.lock().unwrap().take() {
        pb.finish_with_message("Stability scan completed.");
    }

    // Re-assemble in the original mesh order (each group preserves relative
    // order; pop from the front).
    let mut by_key = by_key;
    mesh.iter()
        .map(|env| {
            let pts = by_key.get_mut(&build_key(*env)).expect("group exists");
            pts.remove(0)
        })
        .collect()
}

/// Which environment axis a radial probe walks along.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    Temp,
    Ph,
    Pressure,
    Ionic,
}

impl Axis {
    pub fn name(&self) -> &'static str {
        match self {
            Axis::Temp => "temp",
            Axis::Ph => "ph",
            Axis::Pressure => "pressure",
            Axis::Ionic => "ionic",
        }
    }

    /// Step `env` along this axis by `step × sign(direction)`, clamped into the
    /// biologically sensible ranges.
    pub fn step(&self, env: EnvParams, direction: Direction, step: f32) -> EnvParams {
        let mut e = env;
        let s = match direction {
            Direction::Positive => step,
            Direction::Negative => -step,
        };
        match self {
            Axis::Temp => e.temp_k += s,
            Axis::Ph => e.ph += s,
            Axis::Pressure => e.pressure_bar += s,
            Axis::Ionic => e.ionic_strength_m += s,
        }
        EnvParams::new(e.ph, e.temp_k, e.pressure_bar, e.ionic_strength_m)
    }

    /// Value of `env` along this axis.
    pub fn value(&self, env: EnvParams) -> f32 {
        match self {
            Axis::Temp => env.temp_k,
            Axis::Ph => env.ph,
            Axis::Pressure => env.pressure_bar,
            Axis::Ionic => env.ionic_strength_m,
        }
    }

    /// Distance between two envs along this axis (for bisection termination).
    fn gap(&self, a: EnvParams, b: EnvParams) -> f32 {
        (self.value(a) - self.value(b)).abs()
    }

    /// Midpoint of two envs along this axis (other axes taken from `a`).
    fn midpoint(&self, a: EnvParams, b: EnvParams) -> EnvParams {
        let mid = 0.5 * (self.value(a) + self.value(b));
        match self {
            Axis::Temp => EnvParams::new(a.ph, mid, a.pressure_bar, a.ionic_strength_m),
            Axis::Ph => EnvParams::new(mid, a.temp_k, a.pressure_bar, a.ionic_strength_m),
            Axis::Pressure => EnvParams::new(a.ph, a.temp_k, mid, a.ionic_strength_m),
            Axis::Ionic => EnvParams::new(a.ph, a.temp_k, a.pressure_bar, mid),
        }
    }

    /// Sane limit along this axis in `direction` (from `env::sane`).
    fn bound(&self, direction: Direction) -> f32 {
        use crate::env::sane;
        match (self, direction) {
            (Axis::Temp, Direction::Positive) => sane::TEMP_K_MAX,
            (Axis::Temp, Direction::Negative) => sane::TEMP_K_MIN,
            (Axis::Ph, Direction::Positive) => sane::PH_MAX,
            (Axis::Ph, Direction::Negative) => sane::PH_MIN,
            (Axis::Pressure, Direction::Positive) => sane::PRESSURE_BAR_MAX,
            (Axis::Pressure, Direction::Negative) => sane::PRESSURE_BAR_MIN,
            (Axis::Ionic, Direction::Positive) => sane::IONIC_M_MAX,
            (Axis::Ionic, Direction::Negative) => sane::IONIC_M_MIN,
        }
    }
}

/// Probe direction relative to the anchor point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Positive,
    Negative,
}

/// One axis to probe bidirectionally from the anchor.
#[derive(Debug, Clone, Copy)]
pub struct AxisProbe {
    pub axis: Axis,
    /// Step magnitude along the axis (same for + and −).
    pub step: f32,
    /// Max steps in each direction before giving up.
    pub max_steps: usize,
    /// Boundary-location precision. `Some(p)` enables the adaptive
    /// coarse-doubling + bisection walk (refines the boundary to within `p`);
    /// `None` keeps the legacy uniform step-by-step walk.
    pub precision: Option<f32>,
}

/// Result of probing one ray (one axis × one direction) outward from the anchor.
#[derive(Debug, Clone)]
pub struct RadialResult {
    pub axis: Axis,
    pub direction: Direction,
    pub anchor: EnvParams,
    /// Points probed outward from the anchor (index 0 = first step).
    pub points: Vec<StabilityPoint>,
}

impl RadialResult {
    /// The last stable point on this ray — the stability boundary seen from the
    /// stable side. `None` if the very first step is already unstable.
    pub fn boundary_stable(&self) -> Option<&StabilityPoint> {
        self.points.iter().filter(|p| p.stable).last()
    }

    /// The first point beyond the stable region on this ray. `None` if every
    /// probed point stayed stable (stability extends past `max_steps`).
    pub fn first_unstable(&self) -> Option<&StabilityPoint> {
        self.points.iter().find(|p| !p.stable)
    }
}

/// Emit one per-point progress line to stderr (line-flushed, so it appears
/// live even when stdout is redirected). Thread-safe via an atomic counter.
fn report_point(
    cfg: &StabilityConfig,
    done: &AtomicUsize,
    total: usize,
    pt: &StabilityPoint,
    env: EnvParams,
    axis: Axis,
    direction: Direction,
) {
    if !cfg.progress {
        return;
    }
    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
    let verdict = if pt.build_failed {
        "build_failed"
    } else if pt.stable {
        "stable"
    } else {
        "unstable"
    };
    let m1 = pt.metrics.as_ref().map_or(f64::NAN, |m| m.m1);
    let log_line = format!(
        "[stability] {:>3}/{total} {axis:?}/{direction:?} T={:.0} pH={:.1}: {verdict} (crashed={}, m1={m1:.0})",
        n,
        env.temp_k,
        env.ph,
        pt.crashed,
    );
    log_progress(log_line, true);
}

/// Adaptive (coarse-to-fine) probe of one ray — the DL-style "big step first,
/// then refine" idea. Phase 1 walks outward with a DOUBLING step to bracket the
/// stable/unstable boundary cheaply; Phase 2 bisects that interval down to
/// `probe.precision`. Samples far fewer points than a uniform walk and locates
/// the boundary more tightly.
fn probe_ray_adaptive(
    dev: &dynamics::ComputationDevice,
    param_set: &dynamics::params::FfParamSet,
    structure: &StructureInput,
    anchor: EnvParams,
    axis: Axis,
    direction: Direction,
    probe: &AxisProbe,
    build_opts: &BuildOptions,
    cfg: &StabilityConfig,
    template: &mut Option<PointTemplate>,
    done: &AtomicUsize,
    total: usize,
) -> Vec<StabilityPoint> {
    let precision = probe.precision.unwrap_or(probe.step);
    let bound = axis.bound(direction);
    let mut points: Vec<StabilityPoint> = Vec::new();
    // The anchor is the (assumed stable) reference; if even the first step is
    // unstable we bisect between the anchor and that point.
    let mut lo: Option<EnvParams> = Some(anchor); // last stable
    let mut hi: Option<EnvParams> = None; // first unstable

    // Phase 1: coarse outward walk, doubling the step each time.
    let mut step = probe.step;
    loop {
        if points.len() >= probe.max_steps {
            break;
        }
        let env = axis.step(anchor, direction, step);
        let at_bound = if direction == Direction::Positive {
            axis.value(env) >= bound
        } else {
            axis.value(env) <= bound
        };
        let pt = probe_point(dev, param_set, structure, env, build_opts, cfg, template);
        report_point(cfg, done, total, &pt, env, axis, direction);
        let stable = pt.stable;
        points.push(pt);
        if !stable {
            hi = Some(env);
            break;
        }
        lo = Some(env);
        if at_bound {
            break; // stable all the way to the sane limit
        }
        step *= cfg.step_growth;
    }

    // Phase 2: bisect the [stable, unstable] interval down to `precision`.
    if let (Some(mut l), Some(mut h)) = (lo, hi) {
        while points.len() < probe.max_steps && axis.gap(l, h) > precision {
            let mid = axis.midpoint(l, h);
            if axis.gap(mid, l) <= f32::EPSILON || axis.gap(mid, h) <= f32::EPSILON {
                break; // no further progress possible
            }
            let pt = probe_point(dev, param_set, structure, mid, build_opts, cfg, template);
            report_point(cfg, done, total, &pt, mid, axis, direction);
            let stable = pt.stable;
            points.push(pt);
            if stable {
                l = mid;
            } else {
                h = mid;
            }
        }
    }
    points
}

/// Bidirectional boundary scan: start from the (assumed stable) `anchor` and
/// walk outward along each probe axis in both + and − directions until the
/// system is judged unstable, a build fails, or `max_steps` is reached. Each
/// ray is an independent build+simulate, so all rays run in parallel via rayon.
pub fn scan_radial(
    dev: &dynamics::ComputationDevice,
    param_set: &dynamics::params::FfParamSet,
    structure: &StructureInput,
    anchor: EnvParams,
    probes: &[AxisProbe],
    build_opts: &BuildOptions,
    cfg: &StabilityConfig,
) -> Vec<RadialResult> {
    let jobs: Vec<(Axis, Direction)> = probes
        .iter()
        .flat_map(|p| [(p.axis, Direction::Positive), (p.axis, Direction::Negative)])
        .collect();
    // Upper bound on candidate points (rays stop early at the first unstable).
    let total: usize = probes.iter().map(|p| 2 * p.max_steps).sum();
    let done = AtomicUsize::new(0);

    if cfg.progress {
        let pb = indicatif::ProgressBar::new(total as u64);
        pb.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.green/blue} {pos}/{len} {msg} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message("Scanning radial stability...");
        *PROGRESS_BAR.lock().unwrap() = Some(pb);
    }

    if cfg.progress {
        log_progress(format!(
            "[stability] scanning up to {total} candidate points across {} rays",
            jobs.len()
        ), false);
    }

    let results: Vec<RadialResult> = jobs.par_iter()
        .map(|&(axis, direction)| {
            let probe = probes
                .iter()
                .find(|p| p.axis == axis)
                .expect("probe for axis");
            let env = axis.step(anchor, direction, probe.step);
            if cfg.progress {
                log_progress(format!(
                    "[stability] ray {axis:?}/{direction:?}: T={:.0} pH={:.1} (up to {} pts, {})",
                    env.temp_k,
                    env.ph,
                    probe.max_steps,
                    if axis == Axis::Ph {
                        "rebuild per pt"
                    } else {
                        "template reuse"
                    },
                ), false);
            }
            // One reusable template per ray: temp / pressure / ionic rays share
            // the same build across points (only the runtime T changes); ph rays
            // rebuild per point because protonation changes the system.
            let mut template: Option<PointTemplate> = None;
            // Adaptive (coarse-doubling + bisection) when a precision is set —
            // the default for boundary-location scans. Falls back to a uniform
            // walk when `precision` is None (legacy behaviour).
            let points = if probe.precision.is_some() {
                probe_ray_adaptive(
                    dev,
                    param_set,
                    structure,
                    anchor,
                    axis,
                    direction,
                    probe,
                    build_opts,
                    cfg,
                    &mut template,
                    &done,
                    total,
                )
            } else {
                let mut pts = Vec::with_capacity(probe.max_steps);
                let mut env = axis.step(anchor, direction, probe.step);
                for _ in 0..probe.max_steps {
                    let pt = probe_point(
                        dev,
                        param_set,
                        structure,
                        env,
                        build_opts,
                        cfg,
                        &mut template,
                    );
                    report_point(cfg, &done, total, &pt, env, axis, direction);
                    let unstable = !pt.stable;
                    pts.push(pt);
                    if unstable {
                        break;
                    }
                    env = axis.step(env, direction, probe.step);
                }
                pts
            };
            RadialResult {
                axis,
                direction,
                anchor,
                points,
            }
        })
        .collect();

    if let Some(pb) = PROGRESS_BAR.lock().unwrap().take() {
        pb.finish_with_message("Radial scan completed.");
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metrics(m1: f64, m2: f64, m3: f64, m4: f64, m5: f64) -> MetricsResult {
        MetricsResult {
            m1,
            m2,
            m3,
            m4,
            m5,
            u_t_kcal: -1.0,
            rg: 14.0,
            n_ss_ref: 290,
            n_ss_kept: 290,
            n_surface_charged: 17,
        }
    }

    #[test]
    fn grid_mesh() {
        let g = EnvGrid::default();
        assert_eq!(g.mesh().len(), 3 * 3 * 1 * 1);
        let first = g.mesh()[0];
        assert_eq!(first.temp_k, 300.0);
        assert_eq!(first.ph, 6.5);
    }

    #[test]
    fn linspace_per_axis_resolution() {
        // fine temp steps, coarse pH steps
        let t = EnvGrid::linspace(290.0, 330.0, 10.0);
        assert_eq!(t, vec![290.0, 300.0, 310.0, 320.0, 330.0]);
        let ph = EnvGrid::linspace(6.0, 8.0, 1.0);
        assert_eq!(ph, vec![6.0, 7.0, 8.0]);
        // degenerate ranges -> empty axis
        assert!(EnvGrid::linspace(8.0, 6.0, 1.0).is_empty());
        assert!(EnvGrid::linspace(6.0, 8.0, 0.0).is_empty());
    }

    #[test]
    fn from_ranges_builds_mesh() {
        let g = EnvGrid::from_ranges((290.0, 310.0, 10.0), (6.0, 8.0, 1.0), None, None);
        assert_eq!(g.temps.len(), 3);
        assert_eq!(g.phs.len(), 3);
        assert_eq!(g.pressures, vec![1.0]);
        assert_eq!(g.ionics, vec![0.0]);
        assert_eq!(g.mesh().len(), 9);
    }

    #[test]
    fn stability_judgement() {
        let cfg = StabilityConfig::default();
        // native fold preserved: stable
        let ok = sample_metrics(100.0, 0.02, 0.05, 0.001, 0.2);
        assert!(is_stable(&ok, false, &cfg));
        // crashed
        assert!(!is_stable(&ok, true, &cfg));
        // Rg drift too large (unfolding)
        let unfolded = sample_metrics(100.0, 0.5, 0.1, 0.001, 0.2);
        assert!(!is_stable(&unfolded, false, &cfg));
        // too much SS loss
        let ss_lost = sample_metrics(100.0, 0.02, 0.9, 0.001, 0.2);
        assert!(!is_stable(&ss_lost, false, &cfg));
        // severe clash
        let clashed = sample_metrics(100.0, 0.02, 0.05, 0.3, 0.2);
        assert!(!is_stable(&clashed, false, &cfg));
        // energy blow-up
        let exploded = sample_metrics(1e9, 0.02, 0.05, 0.001, 0.2);
        assert!(!is_stable(&exploded, false, &cfg));
    }

    #[test]
    fn axis_stepping() {
        let a = EnvParams::new(7.0, 310.0, 1.0, 0.0);
        assert_eq!(Axis::Temp.step(a, Direction::Positive, 10.0).temp_k, 320.0);
        assert_eq!(Axis::Temp.step(a, Direction::Negative, 10.0).temp_k, 300.0);
        assert_eq!(Axis::Ph.step(a, Direction::Positive, 0.5).ph, 7.5);
        assert_eq!(Axis::Ph.step(a, Direction::Negative, 0.5).ph, 6.5);
        // stepping clamps into the biologically sensible ranges
        assert_eq!(Axis::Ph.step(a, Direction::Negative, 99.0).ph, crate::env::sane::PH_MIN);
        assert_eq!(
            Axis::Temp.step(a, Direction::Positive, 999.0).temp_k,
            crate::env::sane::TEMP_K_MAX
        );
    }

    #[test]
    fn radial_boundary_helpers() {
        let anchor = EnvParams::new(7.0, 310.0, 1.0, 0.0);
        let pt = |stable: bool| StabilityPoint {
            env: anchor,
            stable,
            crashed: false,
            build_failed: false,
            terminated_reason: None,
            metrics: None,
        };
        // stable, stable, unstable → boundary on the stable side + first outside
        let r = RadialResult {
            axis: Axis::Temp,
            direction: Direction::Positive,
            anchor,
            points: vec![pt(true), pt(true), pt(false)],
        };
        assert_eq!(r.boundary_stable().map(|p| p.stable), Some(true));
        assert_eq!(r.first_unstable().map(|p| p.stable), Some(false));

        // all stable → stability extends past max_steps
        let all = RadialResult {
            axis: Axis::Ph,
            direction: Direction::Negative,
            anchor,
            points: vec![pt(true), pt(true)],
        };
        assert!(all.first_unstable().is_none());
        assert!(all.boundary_stable().is_some());

        // first step already unstable → no boundary on the stable side
        let bad = RadialResult {
            axis: Axis::Ionic,
            direction: Direction::Positive,
            anchor,
            points: vec![pt(false)],
        };
        assert!(bad.boundary_stable().is_none());
        assert!(bad.first_unstable().is_some());
    }
}
