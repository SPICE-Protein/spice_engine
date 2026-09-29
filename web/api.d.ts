// Type declarations for spice_engine's browser Web API (api.mjs).
// Mirrors the extern-"C" wasm build (v1.3.9); see docs/web_demo.md for the ABI.

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
  dispose(): void;
}

export declare function createSim(opts?: { wasm?: string; log?: (msg: string) => void }): Promise<Sim>;
