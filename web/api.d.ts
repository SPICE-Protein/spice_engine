// Type declarations for spice_engine's browser Web API (api.mjs).
// Mirrors the extern-"C" wasm build (v1.3.9; v1.3.10 FFI-parity surface);
// see docs/web_demo.md for the ABI.

/** Build-time environment; every key optional. Maps to Engine.build params.
 *  `soluteK` is runtime-only: when given with `tempK`, the dual bath engages. */
export interface Env {
  ph?: number;
  tempK?: number;
  soluteK?: number;
  pressureBar?: number; // <=0 => NVT (barostat off)
  ionicStrengthM?: number; // NaCl background, not total I
  relaxIters?: number;
  tolerance?: number;
  strictIncomplete?: boolean;
  boxPadA?: number; // 0 => builder default (10 A)
  mgCl2M?: number;
  caCl2M?: number;
  srCl2M?: number;
  baCl2M?: number;
  redoxReducing?: number; // [0,1] seeded CYX->CYS
  cosolventsJson?: string;
  saltsJson?: string;
  /** Static/oscillating external field, kcal/(mol·e·A); clamped to ±5 per axis. */
  efieldJson?: [number, number, number];
  efieldOmega?: number; // rad/ps, 0 = static
}

export interface SystemInfo {
  handle: number;
  nAtoms: number;
  nResidues: number;
  nWater: number;
  nSites: number;
  netChargeE: number;
  effectiveIonicM: number;
}

/** Step metrics: observables dict + step-result extras (exact wasm ABI keys). */
export interface Metrics {
  step_count: number;
  time_ps: number;
  dt_ps: number;
  u_total_kcal: number;
  u_nonbonded_kcal: number;
  u_bonded_kcal: number;
  u_t_kcal: number; // potential energy as returned by the step itself
  ke_kcal: number;
  t_kin: number; // total kinetic temperature K
  pressure_bar: number;
  n_clamped: number;
  max_accel_clamped: number;
  solute_t_k: number;
  water_t_k: number;
  rg: number;
  crashed: boolean;
  crash_reason?: string | null;
  trend_alarm?: string | null;
  [key: string]: unknown;
}

export interface Observables {
  u_total_kcal?: number;
  solute_t_k?: number;
  water_t_k?: number;
  pressure_bar?: number;
  rg_a?: number;
  cell_a?: number;
  [key: string]: unknown;
}

/** coords: [x,y,z]*nSites angstrom; roles: 0 solute heavy, 1 solute H,
 *  2 water O, 3 water H. Fresh copies — safe to keep across steps. */
export interface Snapshot {
  coords: Float32Array;
  roles: Int32Array;
  nSites: number;
  metrics: Metrics | null;
  observables?: Observables;
}

export interface Animation {
  stop(): void;
}

/** Points: array of [x,y,z] triples or a flat Float64Array/number[]. */
export type ProbePoints = Float64Array | number[] | [number, number, number][];

export interface SelectionSpec {
  resSeq: number[];
  names?: string[];
  sidechainHeavy?: boolean;
}

export interface BottleneckResult {
  profile: number[];
  bottleneck: number;
}

/** Python `Engine.metrics()` parity: the five metrics + structural extras. */
export interface MetricsReport {
  m1: number; m2: number; m3: number; m4: number; m5: number;
  rg: number;
  u_t_kcal: number;
  n_ss_ref: number;
  n_ss_kept: number;
  n_surface_charged: number;
  stability_margin: number;
  rmsf: number;
}

export interface ForceRow {
  element: string;
  residue: string;
  seq_id: number;
  serial: number;
  force: number;
  min_d: number;
}

export declare class Sim {
  readonly spice: unknown; // raw loader.mjs Spice runtime (power users)
  readonly engine: unknown; // raw loader.mjs SpiceEngine handle
  info: SystemInfo | null;
  last: Metrics | null;

  static create(opts?: { wasm?: string; log?: (msg: string) => void }): Promise<Sim>;
  static load(cifText: string, opts?: { env?: Env; wasm?: string; log?: (msg: string) => void }): Promise<Sim>;

  build(cifText: string, env?: Env): this;
  step(n?: number): Metrics;
  stepAction(forces: Float32Array): Metrics; // length = nAtoms * 3
  set(ctrl: {
    tempK?: number;
    soluteK?: number | null;
    gamma?: number;
    timestepPs?: number;
    pressureBar?: number;
  }): this;
  setForcesOff(f?: { bonded?: boolean; coulomb?: boolean; lj?: boolean; longRange?: boolean }): this;
  resetVelocities(): this;
  snapshot(opts?: { observables?: boolean }): Snapshot;
  observables(): Observables;
  animate(onFrame: (sim: Sim, metrics: Metrics) => void, opts?: { stepsPerFrame?: number; fps?: number }): Animation;

  // -- v1.3.10 analysis + control (Python-FFI parity) ------------------------
  atomNames(): string[]; // one per solute atom (state.atoms order)
  atomLabels(): { element: string; residue: string; seq_id: number; serial: number }[];
  sequence(): string;
  select(spec: SelectionSpec): number[];
  contacts(a: number[], b: number[], cutoff?: number): number;
  bottleneck(spec: { path: [number, number, number][]; spacing?: number; exclude?: number[]; includeWater?: boolean }): BottleneckResult;
  sasa(opts?: { probeRadius?: number; nSphere?: number }): Float64Array;
  esp(points: ProbePoints): Float64Array; // gauge-arbitrary absolute — use differences
  field(points: ProbePoints, opts?: { positions?: Float64Array | null }): Float64Array; // 3 per point
  pmePositions(): Float64Array; // 3 per PME site: solute (wrapped) + M/H0/H1 per water (O excluded)
  coordsCa(): Float64Array; // 3 per residue
  pseudoLabels(): Float64Array;
  resetPseudoLabels(): this;
  perResidueMaxForce(): Float64Array;
  metrics(): MetricsReport;
  energyTerms(): { total: number; nonbonded: number; bonded: number };
  speciesTemperatures(): {
    solute_ke_kcal: number; water_ke_kcal: number;
    solute_dof: number; water_dof: number;
    solute_t_k: number; water_t_k: number;
  };
  thermoInfo(): Record<string, number>;
  waterRigidSplit(): Record<string, number>;
  envInfo(): Record<string, unknown>;
  exclusionDiagnostics(): Record<string, number>;
  debugStateDump(): Record<string, unknown>; // LARGE — analysis, not per-frame
  computationTime(): Record<string, number>;
  clashReport(minForce?: number): ForceRow[];
  forceReport(minForce?: number): ForceRow[];
  rigidScaleProbe(lam: number): { u_kcal: number; virial_kcal: number; pressure_bar: number }; // MUTATES forces
  setIntegrator(mode: "langevin_middle" | "langevin_strong" | "nve"): this;
  setTrend(cfg: "rl_fail_fast" | "default" | Record<string, unknown>): this;
  resetTrend(): this;
  clearTrend(): this;
  hasTrend(): boolean;
  setSkipWaterThermostat(on: boolean): this;
  addRestraint(i0: number, i1: number, r0: number, k: number): this;
  updateRestraint(idx: number, r0: number, k: number): boolean;
  clearRestraints(): this;
  equilibrate(opts?: {
    rampSteps?: number; tStartK?: number; kRestraint?: number;
    holdSteps?: number; restrainHydrogens?: boolean; frictionGamma?: number;
  }): Metrics;

  dispose(): void;
}

export declare function createSim(opts?: { wasm?: string; log?: (msg: string) => void }): Promise<Sim>;
