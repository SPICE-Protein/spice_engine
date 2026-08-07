//! EnginePool — several workers each holding an independent `SpiceEngine`, for
//! path A/B parallelism and a mutant pool (16–32).
//!
//! `MdState` is not shared across threads (each worker owns one), so every
//! worker holds its own `SpiceEngine` + `ForceAction` + `Metrics`. `step_all`
//! advances each worker inside a rayon parallel iterator; `metrics_all` collects
//! the five metrics.

use rayon::prelude::*;

use crate::actions::ForceAction;
use crate::engine::{SpiceEngine, StepResult};
use crate::metrics::{Metrics, MetricsConfig, MetricsResult};

// Compile-time guarantee that a worker (and its MdState) is `Send`, which the
// rayon `par_iter_mut` below relies on. (The `Barostat` in the dynamics fork now
// uses `StdRng` instead of the thread-local RNG so this holds.)
#[allow(dead_code)]
fn _assert_worker_send() {
    fn f<T: Send>() {}
    f::<EngineWorker>();
}

/// An independent worker: engine + action interface + metrics evaluator.
pub struct EngineWorker {
    pub engine: SpiceEngine,
    pub force: ForceAction,
    pub metrics: Metrics,
}

impl EngineWorker {
    pub fn new(engine: SpiceEngine) -> Self {
        let n_res = engine.topology.sequence.len();
        let metrics = Metrics::new(&engine, MetricsConfig::default());
        let force = ForceAction::new(n_res, 16, 0.5, 20);
        Self {
            engine,
            force,
            metrics,
        }
    }

    /// Step forward once (including the bias-force action).
    pub fn step(&mut self, action: &[f32]) -> StepResult {
        self.force.step(&mut self.engine, action)
    }

    pub fn metrics(&self) -> MetricsResult {
        self.metrics.compute(&self.engine)
    }
}

/// Parallel worker pool.
pub struct EnginePool {
    workers: Vec<EngineWorker>,
}

impl EnginePool {
    /// Build a pool from a set of engines (each gets a default ForceAction + Metrics).
    pub fn new(engines: Vec<SpiceEngine>) -> Self {
        Self {
            workers: engines.into_iter().map(EngineWorker::new).collect(),
        }
    }

    pub fn from_workers(workers: Vec<EngineWorker>) -> Self {
        Self { workers }
    }

    pub fn len(&self) -> usize {
        self.workers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    pub fn worker(&self, i: usize) -> &EngineWorker {
        &self.workers[i]
    }

    pub fn worker_mut(&mut self, i: usize) -> &mut EngineWorker {
        &mut self.workers[i]
    }

    pub fn workers(&self) -> &[EngineWorker] {
        &self.workers
    }

    pub fn workers_mut(&mut self) -> &mut [EngineWorker] {
        &mut self.workers
    }

    /// Advance all workers in parallel; `actions[i]` is the action vector for the
    /// i-th worker. Returns each worker's `StepResult` (aligned with the input).
    pub fn step_all(&mut self, actions: &[Vec<f32>]) -> Result<Vec<StepResult>, String> {
        if actions.len() != self.workers.len() {
            return Err(format!(
                "actions len {} != worker count {}",
                actions.len(),
                self.workers.len()
            ));
        }
        Ok(self
            .workers
            .par_iter_mut()
            .zip(actions)
            .map(|(w, a)| w.force.step(&mut w.engine, a))
            .collect())
    }

    /// Collect the five metrics across all workers (aligned with `workers`).
    pub fn metrics_all(&self) -> Vec<MetricsResult> {
        self.workers.iter().map(|w| w.metrics.compute(&w.engine)).collect()
    }

    /// Hot-switch temperature on all workers.
    pub fn set_temperature_all(&mut self, temp_k: f32) {
        for w in &mut self.workers {
            w.engine.set_temperature(temp_k);
        }
    }
}
