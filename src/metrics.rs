//! Five physical metrics — SPICE's reward/state vector `M` (RL decision input).
//!
//! Each metric is normalised so "closer to equilibrium / native state is better":
//! ```text
//!   m1 = MAD(U) / (k_B·T)         potential-energy fluctuation (spike-robust, unitless)
//!   m2 = |Rg − Rg_ref| / Rg_ref   radius-of-gyration drift (relative)
//!   m3 = 1 − SS_kept / SS_ref     secondary-structure loss (DSSP-lite backbone H-bonds)
//!   m4 = Clash score              fraction of heavy-atom pairs d/(vdW sum) < threshold
//!   m5 = surface-charge mismatch  surface ionizable residues' actual vs pH-ideal charge
//! ```
//!
//! Notes: m3 uses a simplified DSSP-lite (backbone C=O(i)···N(j) distance < cutoff
//! as the H-bond test, no full DSSP energy criterion); m5 approximates "surface"
//! by burial count (non-self heavy atoms within Cα radius 8 Å ≤ threshold), with
//! actual charge summed from atomic partial charges.

use na_seq::Element;

use crate::engine::SpiceEngine;

/// Boltzmann constant, kcal/(mol·K).
pub const KB_KCAL_MOL_K: f64 = 0.0019872;
/// `√332.06` — the dynamics crate's charge-unit scaler (q_internal = q_e × scaler).
pub const CHARGE_UNIT_SCALER: f32 = 18.2223;

/// Van der Waals radii by element (Å), used by the clash metric.
#[derive(Debug, Clone)]
pub struct VdwRadii {
    pub c: f64,
    pub n: f64,
    pub o: f64,
    pub s: f64,
    pub p: f64,
    pub h: f64,
}

impl VdwRadii {
    pub fn radius(&self, elm: Element) -> f64 {
        use Element::*;
        match elm {
            Carbon => self.c,
            Nitrogen => self.n,
            Oxygen => self.o,
            Sulfur => self.s,
            Phosphorus => self.p,
            Hydrogen => self.h,
            _ => 1.7,
        }
    }
}

impl Default for VdwRadii {
    fn default() -> Self {
        Self {
            c: 1.70,
            n: 1.55,
            o: 1.52,
            s: 1.80,
            p: 1.80,
            h: 1.20,
        }
    }
}

/// Simplified pKa table for surface-charge matching (m5).
#[derive(Debug, Clone)]
pub struct PkaTable {
    pub asp: f64,
    pub glu: f64,
    pub his: f64,
    pub lys: f64,
    pub arg: f64,
    pub nterm: f64,
    pub cterm: f64,
}

impl Default for PkaTable {
    fn default() -> Self {
        Self {
            asp: 3.9,
            glu: 4.3,
            his: 6.0,
            lys: 10.5,
            arg: 12.5,
            nterm: 8.0,
            cterm: 3.6,
        }
    }
}

/// Tunables for the metric computation.
#[derive(Debug, Clone)]
pub struct MetricsConfig {
    /// `m1` variance window (# of U samples).
    pub u_window: usize,
    /// Skip this many *earliest* U samples when computing `m1` — drops the
    /// initial equilibration spike so `m1` reflects the equilibrated system.
    pub u_skip: usize,
    /// `m3` DSSP-lite hydrogen-bond cutoff: C=O(i)···N(j) distance (Å).
    pub hbond_n_o: f64,
    /// `m4` clash ratio threshold: d / (vdW sum) < ratio counts as a clash.
    pub clash_ratio: f64,
    /// `m5` surface detection: burial radius (Å) around each Cα.
    pub surface_radius: f64,
    /// `m5` a residue is "surface" if it has at most this many non-self heavy
    /// atoms within `surface_radius` of its Cα.
    pub surface_max_neighbors: usize,
    pub vdw: VdwRadii,
    pub pka: PkaTable,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            u_window: 100,
            u_skip: 0,
            hbond_n_o: 3.5,
            clash_ratio: 0.6,
            surface_radius: 6.0,
            surface_max_neighbors: 30,
            vdw: VdwRadii::default(),
            pka: PkaTable::default(),
        }
    }
}

/// One computed metrics vector (and some diagnostics for debugging).
#[derive(Debug, Clone)]
pub struct MetricsResult {
    pub m1: f64,
    pub m2: f64,
    pub m3: f64,
    pub m4: f64,
    pub m5: f64,
    /// Instantaneous potential energy, kcal/mol.
    pub u_t_kcal: f64,
    /// Current radius of gyration, Å.
    pub rg: f64,
    /// Number of reference secondary-structure hydrogen bonds.
    pub n_ss_ref: usize,
    /// Of those, how many are still present.
    pub n_ss_kept: usize,
    /// Number of charge-capable surface residues considered by m5.
    pub n_surface_charged: usize,
}

/// Metrics evaluated against a *reference* (native) structure — the structure the
/// engine was built from. Construct once at build time; call `compute` each step.
pub struct Metrics {
    pub config: MetricsConfig,
    /// Radius of gyration of the reference structure, Å.
    pub rg_ref: f64,
    /// Reference secondary-structure hydrogen bonds: `(donor O res, acceptor N res)`.
    pub ss_ref: Vec<(usize, usize)>,
}

fn dist3(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    let dz = a.2 - b.2;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn pos(engine: &SpiceEngine, i: usize) -> (f64, f64, f64) {
    let p = engine.state.atoms[i].posit;
    (p.x as f64, p.y as f64, p.z as f64)
}

/// Radius of gyration over protein heavy atoms (mass-weighted).
pub fn radius_of_gyration(engine: &SpiceEngine) -> f64 {
    let heavy = &engine.topology.heavy_indices;
    if heavy.is_empty() {
        return 0.0;
    }
    let mut com = (0.0f64, 0.0f64, 0.0f64);
    let mut m_sum = 0.0f64;
    for &i in heavy {
        let m = engine.state.atoms[i].mass as f64;
        let (x, y, z) = pos(engine, i);
        com.0 += m * x;
        com.1 += m * y;
        com.2 += m * z;
        m_sum += m;
    }
    if m_sum <= 0.0 {
        return 0.0;
    }
    com.0 /= m_sum;
    com.1 /= m_sum;
    com.2 /= m_sum;

    let mut acc = 0.0f64;
    for &i in heavy {
        let m = engine.state.atoms[i].mass as f64;
        let d2 = dist3(pos(engine, i), com);
        acc += m * d2 * d2;
    }
    (acc / m_sum).sqrt()
}

/// DSSP-lite: find main-chain C=O(i)···N(j) hydrogen bonds. Returns `(donor O res,
/// acceptor N res)` pairs with O–N distance under the cutoff.
fn backbone_hbonds(engine: &SpiceEngine, cutoff: f64) -> Vec<(usize, usize)> {
    let n = engine.topology.sequence.len();
    let mut out = Vec::new();
    for i in 0..n {
        let Some(&oi) = engine.topology.o_indices.get(i) else { continue };
        let po = pos(engine, oi);
        for j in 0..n {
            if i == j {
                continue;
            }
            let Some(&nj) = engine.topology.n_indices.get(j) else { continue };
            if dist3(po, pos(engine, nj)) < cutoff {
                out.push((i, j));
            }
        }
    }
    out
}

/// Cheap SS proxy for the long-MD trend detector: how many of the reference
/// main-chain H-bonds are still present right now. O(res²) per call — call at
/// the trend sampling frequency, not every step.
pub(crate) fn ss_kept_count(engine: &SpiceEngine, refs: &[(usize, usize)], cutoff: f64) -> usize {
    if refs.is_empty() {
        return 0;
    }
    let cur = backbone_hbonds(engine, cutoff);
    refs.iter().filter(|hb| cur.contains(hb)).count()
}

impl Metrics {
    /// Compute reference quantities from the freshly built engine (native structure).
    pub fn new(engine: &SpiceEngine, config: MetricsConfig) -> Self {
        Self {
            rg_ref: radius_of_gyration(engine),
            ss_ref: backbone_hbonds(engine, config.hbond_n_o),
            config,
        }
    }

    /// Ideal protonation charge (in e) of a residue at a given pH. Returns `None`
    /// for residues that cannot carry a net titratable charge.
    fn ideal_charge(
        one: char,
        ph: f64,
        pka: &PkaTable,
        is_nterm: bool,
        is_cterm: bool,
    ) -> Option<f64> {
        if is_nterm {
            return Some(1.0);
        }
        if is_cterm {
            return Some(-1.0);
        }
        match one {
            'D' => Some(if ph < pka.asp { 0.0 } else { -1.0 }),
            'E' => Some(if ph < pka.glu { 0.0 } else { -1.0 }),
            'H' => Some(if ph < pka.his { 1.0 } else { 0.0 }),
            'K' => Some(1.0),
            'R' => Some(1.0),
            _ => None,
        }
    }

    fn actual_residue_charge(engine: &SpiceEngine, res: usize) -> f64 {
        let mut q = 0.0f64;
        for &ai in &engine.topology.residues[res].atom_indices {
            q += engine.state.atoms[ai].partial_charge as f64 / CHARGE_UNIT_SCALER as f64;
        }
        q
    }

    /// Clash fraction among protein heavy atoms (O(N²) over heavy atoms only).
    fn clash_fraction(engine: &SpiceEngine, config: &MetricsConfig) -> f64 {
        let heavy = &engine.topology.heavy_indices;
        let mut clashes = 0usize;
        let mut pairs = 0usize;
        for a in 0..heavy.len() {
            let ia = heavy[a];
            let pa = pos(engine, ia);
            let ra = config.vdw.radius(engine.state.atoms[ia].element);
            for b in (a + 1)..heavy.len() {
                let ib = heavy[b];
                pairs += 1;
                if dist3(pa, pos(engine, ib)) < (ra + config.vdw.radius(engine.state.atoms[ib].element)) * config.clash_ratio {
                    clashes += 1;
                }
            }
        }
        if pairs == 0 {
            0.0
        } else {
            clashes as f64 / pairs as f64
        }
    }

    /// Surface residues via burial counting: a residue is "surface" when it has at
    /// most `surface_max_neighbors` non-self heavy atoms within `surface_radius` of
    /// its Cα (a cheap solvent-accessibility proxy).
    fn surface_residues(&self, engine: &SpiceEngine) -> Vec<bool> {
        let n = engine.topology.sequence.len();
        let heavy = &engine.topology.heavy_indices;
        let mut surface = vec![false; n];
        for i in 0..n {
            let Some(&cai) = engine.topology.ca_indices.get(i) else { continue };
            let pc = pos(engine, cai);
            let self_atoms = &engine.topology.residues[i].atom_indices;
            let mut count = 0usize;
            for &hj in heavy {
                if self_atoms.contains(&hj) {
                    continue;
                }
                if dist3(pc, pos(engine, hj)) < self.config.surface_radius {
                    count += 1;
                }
            }
            surface[i] = count <= self.config.surface_max_neighbors;
        }
        surface
    }

    /// Surface charge mismatch (m5): mean |actual − ideal| over surface,
    /// charge-capable residues. 0 when none.
    fn surface_charge_mismatch(&self, engine: &SpiceEngine) -> (f64, usize) {
        let ph = engine.env.ph as f64;
        let n = engine.topology.sequence.len();
        let surface = self.surface_residues(engine);
        let mut acc = 0.0f64;
        let mut count = 0usize;
        for i in 0..n {
            if !surface[i] {
                continue;
            }
            let one = engine.topology.residues[i].one_letter;
            let is_nterm = i == 0;
            let is_cterm = i + 1 == n;
            let Some(ideal) = Self::ideal_charge(one, ph, &self.config.pka, is_nterm, is_cterm) else {
                continue;
            };
            let actual = Self::actual_residue_charge(engine, i);
            acc += (actual - ideal).abs();
            count += 1;
        }
        let m5 = if count == 0 { 0.0 } else { acc / count as f64 };
        (m5, count)
    }

    /// Evaluate all five metrics for the current engine state.
    pub fn compute(&self, engine: &SpiceEngine) -> MetricsResult {
        // m1: MAD(U)/(k_B T) — median absolute deviation (scaled like a σ) instead
        // of variance, so rare numerical spikes (accel clamps from residual build
        // strain) do not dominate the fluctuation estimate.
        let temp_k = engine.env.temp_k as f64;
        let hist: Vec<f64> = engine
            .u_history
            .iter()
            .skip(self.config.u_skip)
            .rev()
            .take(self.config.u_window)
            .copied()
            .collect();
        let m1 = if hist.len() >= 2 {
            let mut sorted: Vec<f64> = hist.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = sorted[sorted.len() / 2];
            let mut abs_dev: Vec<f64> = sorted.iter().map(|u| (u - med).abs()).collect();
            abs_dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mad = abs_dev[abs_dev.len() / 2];
            // 1.4826 × MAD ≈ σ for normally distributed data.
            let sigma = mad * 1.4826;
            if temp_k > 0.0 {
                sigma / (KB_KCAL_MOL_K * temp_k)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // m2: Rg drift
        let rg = radius_of_gyration(engine);
        let m2 = if self.rg_ref.abs() > 1e-9 {
            (rg - self.rg_ref).abs() / self.rg_ref.abs()
        } else {
            0.0
        };

        // m3: SS loss
        let cur = backbone_hbonds(engine, self.config.hbond_n_o);
        let mut kept = 0usize;
        for &(i, j) in &self.ss_ref {
            if cur.contains(&(i, j)) {
                kept += 1;
            }
        }
        let m3 = if self.ss_ref.is_empty() {
            0.0
        } else {
            1.0 - kept as f64 / self.ss_ref.len() as f64
        };

        // m4: clash fraction
        let m4 = Self::clash_fraction(engine, &self.config);

        // m5: surface charge mismatch
        let (m5, n_surface_charged) = self.surface_charge_mismatch(engine);

        MetricsResult {
            m1,
            m2,
            m3,
            m4,
            m5,
            u_t_kcal: engine.state.potential_energy,
            rg,
            n_ss_ref: self.ss_ref.len(),
            n_ss_kept: kept,
            n_surface_charged,
        }
    }
}
